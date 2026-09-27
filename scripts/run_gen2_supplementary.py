"""Run the reviewer-requested Gen2 supplementary experiments and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from umbmid_validation.core import (
    DEFAULT_C_GRID,
    assert_group_disjoint,
    bootstrap_auc,
    choose_threshold,
    evaluate_scores,
    extract_physics_features,
    ifft_envelope,
    load_aligned_npz,
    load_fixed_split,
    permute_angles,
    probability,
    save_json,
    time_axis_ns,
    validation_search,
)


def logistic_pipeline(seed: int = 20260910) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed)),
    ])


def plot_bootstrap(values: np.ndarray, point: float, out: Path) -> None:
    lo, hi = np.quantile(values, [0.025, 0.975])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.hist(values, bins=36, color="#4472C4", alpha=0.85, edgecolor="white")
    ax.axvline(point, color="black", lw=2, label=f"Observed AUC = {point:.3f}")
    ax.axvspan(lo, hi, color="#ED7D31", alpha=0.22, label=f"95% CI {lo:.3f}-{hi:.3f}")
    ax.set(xlabel="Bootstrap ROC AUC", ylabel="Count", title=f"Held-out test AUC bootstrap distribution ({len(values):,} resamples)")
    ax.legend(frameon=False)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def grouped_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> list[dict]:
    rows: list[dict] = []
    outer = GroupKFold(n_splits=5)
    for fold, (dev_idx, test_idx) in enumerate(outer.split(x, y, groups), 1):
        # Calibration uses out-of-fold predictions inside the outer development set.
        inner = GroupKFold(n_splits=4)
        best_c, best_auc = None, -np.inf
        for c in DEFAULT_C_GRID:
            oof = np.full(len(dev_idx), np.nan)
            for inner_train, inner_val in inner.split(x[dev_idx], y[dev_idx], groups[dev_idx]):
                model = logistic_pipeline(seed + fold).set_params(model__C=c)
                model.fit(x[dev_idx][inner_train], y[dev_idx][inner_train])
                oof[inner_val] = probability(model, x[dev_idx][inner_val])
            auc = float(roc_auc_score(y[dev_idx], oof))
            if auc > best_auc:
                best_c, best_auc = c, auc
        calibration = np.full(len(dev_idx), np.nan)
        for inner_train, inner_val in inner.split(x[dev_idx], y[dev_idx], groups[dev_idx]):
            model = logistic_pipeline(seed + fold).set_params(model__C=best_c)
            model.fit(x[dev_idx][inner_train], y[dev_idx][inner_train])
            calibration[inner_val] = probability(model, x[dev_idx][inner_val])
        threshold = choose_threshold(y[dev_idx], calibration)
        final = logistic_pipeline(seed + fold).set_params(model__C=best_c)
        final.fit(x[dev_idx], y[dev_idx])
        metrics = evaluate_scores(y[test_idx], probability(final, x[test_idx]), threshold)
        rows.append({"fold": fold, "test_groups": sorted(set(groups[test_idx])), "C": best_c, **metrics.__dict__})
    return rows


def plot_fold_specificity(rows: list[dict], out: Path) -> None:
    values = np.array([row["specificity"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.scatter(range(1, 6), values, s=70, color="#4472C4", zorder=3)
    ax.plot(range(1, 6), values, color="#A5A5A5", lw=1, zorder=2)
    ax.axhline(values.mean(), color="#ED7D31", ls="--", label=f"Mean {values.mean():.3f} +/- {values.std(ddof=0):.3f}")
    ax.set(xticks=range(1, 6), ylim=(-0.05, 1.05), xlabel="GroupKFold fold", ylabel="Specificity", title="Specificity at development-calibrated thresholds")
    ax.legend(frameon=False)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_time_structure(data, out: Path, f_min_hz: float, f_max_hz: float) -> dict:
    times, dt = time_axis_ns(data.x11.shape[1], f_min_hz, f_max_hz)
    gate = (5, 40)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)
    for ax, x, name in zip(axes, (data.x11, data.x21), ("S11", "S21")):
        env = np.abs(np.fft.ifft(x, axis=1))
        env /= np.maximum(env.max(axis=(1, 2), keepdims=True), np.finfo(float).eps)
        for label, color in ((0, "#4472C4"), (1, "#C00000")):
            mean = env[data.y == label].mean(axis=(0, 2))
            ax.plot(times[:80], mean[:80], color=color, label="Lesion" if label else "No lesion")
        ax.axvspan(times[gate[0]], times[gate[1]], color="#70AD47", alpha=0.16, label="[5,40) gate")
        ax.set(ylabel=f"{name} mean envelope")
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set(xlabel="IFFT delay (ns)")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return {
        "frequency_range_hz": [f_min_hz, f_max_hz],
        "frequency_spacing_hz": (f_max_hz - f_min_hz) / (data.x11.shape[1] - 1),
        "ifft_spacing_ns": dt,
        "gate_bins_half_open": list(gate),
        "gate_start_ns": float(times[gate[0]]),
        "gate_stop_edge_ns": float(times[gate[1]]),
        "last_included_sample_ns": float(times[gate[1] - 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--quick", action="store_true", help="Small grids/resample counts for smoke tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_aligned_npz(args.dataset)
    split = load_fixed_split(args.split, len(data.y))
    assert_group_disjoint(split, data.groups)
    x = extract_physics_features(data.x11, data.x21)

    models = {
        "logistic_physics": (
            logistic_pipeline(args.seed), {"model__C": DEFAULT_C_GRID if not args.quick else (0.1, 1.0)}
        ),
        "tuned_random_forest": (
            RandomForestClassifier(random_state=args.seed, class_weight="balanced", n_jobs=-1),
            {"n_estimators": (400, 800) if not args.quick else (80,), "max_depth": (3, 6, None) if not args.quick else (3,), "min_samples_leaf": (2, 5, 10) if not args.quick else (5,), "max_features": ("sqrt", 0.5) if not args.quick else ("sqrt",)},
        ),
        "tuned_hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=args.seed, early_stopping=False),
            {"learning_rate": (0.03, 0.1) if not args.quick else (0.1,), "max_iter": (100, 250) if not args.quick else (50,), "max_leaf_nodes": (7, 15, 31) if not args.quick else (7,), "l2_regularization": (0.0, 1.0, 10.0) if not args.quick else (1.0,)},
        ),
    }
    holdout: dict[str, dict] = {}
    fitted = {}
    for name, (estimator, grid) in models.items():
        model, params, threshold, val_auc = validation_search(estimator, grid, x, data.y, split["train"], split["val"])
        scores = probability(model, x[split["test"]])
        holdout[name] = {"validation_auc": val_auc, "params": params, "test": evaluate_scores(data.y[split["test"]], scores, threshold), "test_scores": scores}
        fitted[name] = model

    primary = holdout["logistic_physics"]
    n_boot = 200 if args.quick else args.bootstrap
    boots = bootstrap_auc(data.y[split["test"]], primary["test_scores"], n_boot, args.seed)
    np.save(args.output / "bootstrap_auc_values.npy", boots)
    plot_bootstrap(boots, primary["test"].roc_auc, args.output / "bootstrap_auc_distribution.png")

    folds = grouped_cv(x, data.y, data.groups, args.seed)
    plot_fold_specificity(folds, args.output / "groupkfold_specificity_points.png")

    # Signal-destruction ablations: remove the early gate, destroy angular coherence,
    # and form a within-phantom label-permutation null distribution.
    ablations: dict[str, dict] = {}
    for name, xa in (
        ("late_time_gate_40_75", extract_physics_features(data.x11, data.x21, gate=(40, 75))),
        ("angular_order_shuffled", extract_physics_features(*permute_angles(data.x11, data.x21, args.seed))),
    ):
        model, params, threshold, val_auc = validation_search(logistic_pipeline(args.seed), {"model__C": DEFAULT_C_GRID if not args.quick else (0.1, 1.0)}, xa, data.y, split["train"], split["val"])
        scores = probability(model, xa[split["test"]])
        ablations[name] = {"validation_auc": val_auc, "params": params, "test": evaluate_scores(data.y[split["test"]], scores, threshold)}

    rng = np.random.default_rng(args.seed)
    n_perm = 20 if args.quick else args.permutations
    null_auc = []
    for _ in range(n_perm):
        shuffled = data.y.copy()
        for group in np.unique(data.groups):
            idx = np.flatnonzero(data.groups == group)
            shuffled[idx] = rng.permutation(shuffled[idx])
        # Repeat the same regularization search used for the observed-label
        # primary model.  The permutation null must include the complete model-
        # selection procedure; otherwise the observed and null AUCs are not
        # generated under comparable search rules.
        model, _, threshold, _ = validation_search(
            logistic_pipeline(args.seed),
            {"model__C": DEFAULT_C_GRID if not args.quick else (0.1, 1.0)},
            x,
            shuffled,
            split["train"],
            split["val"],
        )
        scores = probability(model, x[split["test"]])
        null_auc.append(evaluate_scores(shuffled[split["test"]], scores, threshold).roc_auc)
    ablations["within_phantom_label_permutation"] = {"n": n_perm, "auc_values": null_auc, "mean_auc": float(np.mean(null_auc)), "p_one_sided": float((1 + np.sum(np.asarray(null_auc) >= primary["test"].roc_auc)) / (n_perm + 1))}

    time_info = plot_time_structure(data, args.output / "time_envelope_ns.png", 1e9, 8e9)
    summary = {
        "dataset": {"n": len(data.y), "groups": len(np.unique(data.groups)), "class_counts": np.bincount(data.y).tolist()},
        "split_groups": {name: sorted(set(data.groups[idx])) for name, idx in split.items()},
        "feature_count": x.shape[1],
        "holdout_models": holdout,
        "bootstrap_auc": {"n": n_boot, "mean": float(boots.mean()), "quantiles_2.5_50_97.5": np.quantile(boots, [0.025, 0.5, 0.975])},
        "groupkfold": folds,
        "groupkfold_summary": {key: {"mean": float(np.mean([r[key] for r in folds])), "std_population": float(np.std([r[key] for r in folds]))} for key in ("roc_auc", "sensitivity", "specificity")},
        "ablations": ablations,
        "time_axis": time_info,
    }
    save_json(args.output / "supplementary_results.json", summary)
    print(f"Wrote results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
