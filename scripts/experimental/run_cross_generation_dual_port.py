"""Cross-hardware Gen1 <-> Gen2 validation with explicit harmonization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from umbmid_validation.core import (
    DEFAULT_C_GRID,
    choose_threshold,
    evaluate_scores,
    extract_physics_features,
    load_aligned_npz,
    probability,
    save_json,
)


def interpolate_frequency(x: np.ndarray, source_hz: np.ndarray, target_hz: np.ndarray) -> np.ndarray:
    out = np.empty((len(x), len(target_hz), x.shape[2]), dtype=np.complex128)
    for sample in range(len(x)):
        for angle in range(x.shape[2]):
            out[sample, :, angle] = np.interp(target_hz, source_hz, x[sample, :, angle].real) + 1j * np.interp(target_hz, source_hz, x[sample, :, angle].imag)
    return out


def harmonize(source, target, source_band: tuple[float, float], target_band: tuple[float, float]):
    low, high = max(source_band[0], target_band[0]), min(source_band[1], target_band[1])
    if low >= high:
        raise ValueError("The generations have no overlapping frequency range")
    n_freq = min(source.x11.shape[1], target.x11.shape[1])
    target_hz = np.linspace(low, high, n_freq)
    def one(data, band):
        grid = np.linspace(band[0], band[1], data.x11.shape[1])
        x11 = interpolate_frequency(data.x11, grid, target_hz)
        x21 = interpolate_frequency(data.x21, grid, target_hz)
        n_angle = min(source.x11.shape[2], target.x11.shape[2])
        return resample(x11, n_angle, axis=2), resample(x21, n_angle, axis=2)
    return one(source, source_band), one(target, target_band), target_hz


def source_oof_calibration(x, y, groups, seed):
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    best_c, best_auc, best_scores = None, -np.inf, None
    from sklearn.metrics import roc_auc_score
    for c in DEFAULT_C_GRID:
        scores = np.full(len(y), np.nan)
        for train, val in cv.split(x, y, groups):
            model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, max_iter=5000, solver="liblinear", random_state=seed))])
            model.fit(x[train], y[train]); scores[val] = probability(model, x[val])
        auc = roc_auc_score(y, scores)
        if auc > best_auc:
            best_c, best_auc, best_scores = c, auc, scores
    return best_c, best_auc, choose_threshold(y, best_scores)


def direction(source, target, source_band, target_band, seed):
    (s11, s21), (t11, t21), grid = harmonize(source, target, source_band, target_band)
    xs, xt = extract_physics_features(s11, s21), extract_physics_features(t11, t21)
    c, cv_auc, threshold = source_oof_calibration(xs, source.y, source.groups, seed)
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, max_iter=5000, solver="liblinear", random_state=seed))])
    model.fit(xs, source.y)
    return {"source_group_cv_auc": cv_auc, "C": c, "threshold_from_source_oof": threshold, "common_frequency_hz": [float(grid[0]), float(grid[-1]), len(grid)], "common_angles": s11.shape[2], "target": evaluate_scores(target.y, probability(model, xt), threshold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen1", type=Path, required=True, help="Aligned Gen1 NPZ with the same schema as Gen2")
    parser.add_argument("--gen2", type=Path, required=True)
    parser.add_argument("--gen1-band-ghz", nargs=2, type=float, required=True)
    parser.add_argument("--gen2-band-ghz", nargs=2, type=float, default=(1.0, 8.0))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if not args.gen1.exists():
        raise SystemExit(f"Gen1 aligned data not found: {args.gen1}. Build/provide it before cross-hardware evaluation.")
    gen1, gen2 = load_aligned_npz(args.gen1), load_aligned_npz(args.gen2)
    b1, b2 = tuple(np.asarray(args.gen1_band_ghz) * 1e9), tuple(np.asarray(args.gen2_band_ghz) * 1e9)
    save_json(args.output, {"Gen1_to_Gen2": direction(gen1, gen2, b1, b2, args.seed), "Gen2_to_Gen1": direction(gen2, gen1, b2, b1, args.seed), "note": "Frequency interpolation uses only the common band; circular Fourier resampling harmonizes antenna count; calibration is learned only from source-generation OOF predictions."})
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
