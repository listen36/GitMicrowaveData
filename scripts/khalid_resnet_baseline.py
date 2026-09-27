"""Khalid-style residual CNN evaluated with phantom-disjoint splitting.

The cited ISCAS paper describes four concatenated residual blocks containing
2, 1, 3, and 4 residual layers; every residual layer uses two convolutions,
batch normalization, ReLU, and a shortcut.  It does not publish all filter,
kernel, optimizer, or UM-BMID preprocessing details, so this script records the
remaining choices and reports every seed instead of claiming exact replication.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from umbmid_validation.core import (
    assert_group_disjoint,
    choose_threshold,
    evaluate_scores,
    ifft_envelope,
    load_aligned_npz,
    load_fixed_split,
    save_json,
)


def validation_rank(run: dict) -> float:
    """Use only validation AUC; iteration order resolves 12-decimal ties."""
    return round(float(run["validation_auc"]), 12)


def main() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    data = load_aligned_npz(args.dataset)
    split = load_fixed_split(args.split, len(data.y))
    assert_group_disjoint(split, data.groups)
    # Khalid et al. report S11-only time-domain input.  Retain the same early
    # range gate used by the manuscript so the comparison changes the model,
    # not the available acquisition window.
    array = ifft_envelope(data.x11).astype(np.float32)[:, None, :, :]
    mean = float(array[split["train"]].mean())
    std = float(array[split["train"]].std())
    array = (array - mean) / max(std, 1e-6)

    class Augmented(Dataset):
        def __init__(self, indices, augment, seed):
            self.indices, self.augment = np.asarray(indices), augment
            self.rng = np.random.default_rng(seed)
        def __len__(self): return len(self.indices)
        def __getitem__(self, item):
            idx = self.indices[item]
            value = array[idx].copy()
            if self.augment:
                # Antenna-origin augmentation preserves circular scan geometry.
                value = np.roll(value, int(self.rng.integers(value.shape[-1])), axis=-1)
                value += self.rng.normal(0.0, 0.01, value.shape).astype(np.float32)
            return torch.from_numpy(value.copy()), torch.tensor(float(data.y[idx]))

    class ResidualUnit(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.activation = nn.ReLU(inplace=True)
        def forward(self, value): return self.activation(value + self.body(value))

    class FourBlockResNet(nn.Module):
        def __init__(self, base, dropout):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1, base, 5, padding=2, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
            depths = (2, 1, 3, 4)
            channels = (base, base * 2, base * 3, base * 4)
            self.blocks = nn.ModuleList()
            self.transitions = nn.ModuleList()
            for i, (depth, width) in enumerate(zip(depths, channels)):
                if i:
                    self.transitions.append(nn.Sequential(nn.Conv2d(channels[i - 1], width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)))
                self.blocks.append(nn.Sequential(*[ResidualUnit(width) for _ in range(depth)]))
            total = sum(channels)
            self.head = nn.Sequential(nn.Linear(total, 2 * base), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(2 * base, 1))
        def forward(self, value):
            value = self.stem(value)
            pooled = []
            for i, block in enumerate(self.blocks):
                if i: value = self.transitions[i - 1](value)
                value = block(value)
                pooled.append(value.mean(dim=(-2, -1)))
            return self.head(torch.cat(pooled, dim=1)).squeeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        {"base_channels": 8, "dropout": 0.30, "learning_rate": 1e-3},
        {"base_channels": 8, "dropout": 0.50, "learning_rate": 3e-4},
        {"base_channels": 12, "dropout": 0.30, "learning_rate": 3e-4},
        {"base_channels": 16, "dropout": 0.50, "learning_rate": 1e-4},
    ]
    runs = []
    best = None
    for seed_offset in range(args.seeds):
        run_seed = args.seed + seed_offset
        for config in configs:
            torch.manual_seed(run_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(run_seed)
            model = FourBlockResNet(config["base_channels"], config["dropout"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
            loss_fn = nn.BCEWithLogitsLoss()
            generator = torch.Generator().manual_seed(run_seed)
            loader = DataLoader(Augmented(split["train"], True, run_seed), batch_size=16, shuffle=True, generator=generator)
            best_key, state, stale, used_epochs = (-np.inf, -np.inf), None, 0, 0
            for epoch in range(args.epochs):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(xb.to(device)), yb.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(array[split["val"]]).to(device)
                    logits = model(xv)
                    val_loss = float(loss_fn(logits, torch.from_numpy(data.y[split["val"]].astype(np.float32)).to(device)))
                    val_scores = torch.sigmoid(logits).cpu().numpy()
                val_auc = float(roc_auc_score(data.y[split["val"]], val_scores))
                scheduler.step(val_auc)
                key = (val_auc, -val_loss)
                used_epochs = epoch + 1
                if key > best_key:
                    best_key, stale, state = key, 0, copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                if stale >= args.patience: break
            model.load_state_dict(state); model.eval()
            with torch.no_grad():
                val_scores = torch.sigmoid(model(torch.from_numpy(array[split["val"]]).to(device))).cpu().numpy()
                test_scores = torch.sigmoid(model(torch.from_numpy(array[split["test"]]).to(device))).cpu().numpy()
            threshold = choose_threshold(data.y[split["val"]], val_scores)
            record = {"seed": run_seed, "config": config, "epochs": used_epochs, "validation_auc": float(roc_auc_score(data.y[split["val"]], val_scores)), "test": evaluate_scores(data.y[split["test"]], test_scores, threshold), "test_scores": test_scores}
            runs.append(record)
            # Selection uses validation AUC only. A near-tie (within floating
            # point tolerance) keeps the first prespecified seed/configuration
            # in loop order; the held-out test metric is never a tie-breaker.
            rank = validation_rank(record)
            if best is None or rank > best[0]: best = (rank, record)
            print(f"seed={run_seed} config={config} val_auc={record['validation_auc']:.3f} test_auc={record['test'].roc_auc:.3f}", flush=True)

    aucs = np.array([run["test"].roc_auc for run in runs])
    selected = best[1]
    save_json(args.output, {
        "description": "Khalid-style four-block residual S11 CNN under fixed phantom-disjoint evaluation; partial reproduction because unpublished choices remain",
        "published_structure_used": {"block_depths": [2, 1, 3, 4], "convolutions_per_residual_layer": 2, "batch_normalization": True, "shortcut_connections": True, "block_outputs_concatenated": True},
        "preprocessing": {"port": "S11", "domain": "IFFT magnitude", "gate": [5, 40], "train_mean": mean, "train_std": std, "augmentation": "random circular antenna-origin shift plus Gaussian noise"},
        "device": str(device), "runs": runs, "selected_by_validation": selected,
        "selection_rule": "maximum validation_auc; ties within 12 decimal places retain the first prespecified seed/configuration in loop order; test metrics are never used",
        "all_run_test_auc": {"mean": float(aucs.mean()), "std": float(aucs.std()), "min": float(aucs.min()), "max": float(aucs.max())},
    })
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
