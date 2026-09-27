"""Optional Khalid-inspired lightweight CNN under the manuscript's fixed split.

This is an architectural approximation, not a claim of exact reproduction:
the conference paper does not publish all preprocessing/training details needed
for bitwise replication.  The script uses the same 35 x 72 early-time inputs,
phantom-disjoint split, validation-only model selection, and frozen threshold as
the other manuscript baselines.
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
    cnn_tensor,
    evaluate_scores,
    load_aligned_npz,
    load_fixed_split,
    save_json,
)


def main() -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("PyTorch is optional and not installed. Install torch, then rerun this script.") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_aligned_npz(args.dataset)
    split = load_fixed_split(args.split, len(data.y))
    assert_group_disjoint(split, data.groups)
    x = torch.from_numpy(cnn_tensor(data.x11, data.x21))
    y = torch.from_numpy(data.y.astype(np.float32))

    class Net(nn.Module):
        def __init__(self, channels: int, dropout: float):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(2, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(channels, 2 * channels, 3, padding=1), nn.BatchNorm2d(2 * channels), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(2 * channels, 4 * channels, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * channels, 1))
        def forward(self, value):
            return self.head(self.features(value)).squeeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(x[split["train"]], y[split["train"]]), batch_size=16, shuffle=True)
    best_run = None
    for channels in (8, 16):
        for dropout in (0.25, 0.5):
            for lr in (1e-3, 3e-4):
                model = Net(channels, dropout).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
                loss_fn = nn.BCEWithLogitsLoss()
                best_auc, stale, state = -np.inf, 0, None
                for _ in range(args.epochs):
                    model.train()
                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        loss = loss_fn(model(xb.to(device)), yb.to(device))
                        loss.backward(); optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        val_scores = torch.sigmoid(model(x[split["val"]].to(device))).cpu().numpy()
                    auc = roc_auc_score(data.y[split["val"]], val_scores)
                    if auc > best_auc + 1e-6:
                        best_auc, stale, state = auc, 0, copy.deepcopy(model.state_dict())
                    else:
                        stale += 1
                    if stale >= args.patience:
                        break
                candidate = (best_auc, channels, dropout, lr, state)
                if best_run is None or candidate[:4] > best_run[:4]:
                    best_run = candidate

    val_auc, channels, dropout, lr, state = best_run
    model = Net(channels, dropout).to(device); model.load_state_dict(state); model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(x[split["val"]].to(device))).cpu().numpy()
        test_scores = torch.sigmoid(model(x[split["test"]].to(device))).cpu().numpy()
    threshold = choose_threshold(data.y[split["val"]], val_scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"description": "Khalid-inspired lightweight CNN; not an exact reproduction", "device": str(device), "params": {"channels": channels, "dropout": dropout, "learning_rate": lr}, "validation_auc": val_auc, "test": evaluate_scores(data.y[split["test"]], test_scores, threshold), "test_scores": test_scores})
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
