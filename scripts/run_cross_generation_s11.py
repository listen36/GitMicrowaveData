"""Leakage-controlled S11-only Gen1 <-> Gen2 cross-hardware evaluation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from umbmid_validation.core import (
    DEFAULT_C_GRID,
    choose_threshold,
    evaluate_scores,
    extract_single_port_features,
    probability,
    save_json,
)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        required = {"X11", "y", "group_ids", "sample_ids", "frequency_hz", "angle_deg", "calibration"}
        if missing := required.difference(z.files): raise ValueError(f"Missing {sorted(missing)} in {path}")
        x = np.asarray(z["X11"])
        frequency_hz = np.asarray(z["frequency_hz"], dtype=float)
        angle_deg = np.asarray(z["angle_deg"], dtype=float)
        if frequency_hz.shape != (x.shape[1],) or angle_deg.shape != (x.shape[2],):
            raise ValueError(f"Stored acquisition axes do not match signal shape in {path}")
        if not np.isfinite(frequency_hz).all() or not np.all(np.diff(frequency_hz) > 0):
            raise ValueError(f"frequency_hz must be finite and strictly increasing in {path}")
        if not np.isfinite(angle_deg).all() or not np.all(np.diff(angle_deg) > 0):
            raise ValueError(f"angle_deg must be finite and strictly increasing in {path}")
        return {"x": x, "y": np.asarray(z["y"], int), "groups": np.asarray(z["group_ids"]).astype(str),
                "frequency_hz": frequency_hz, "angle_deg": angle_deg, "calibration": str(z["calibration"])}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate(x, source_hz, target_hz):
    out = np.empty((len(x), len(target_hz), x.shape[2]), np.complex128)
    for sample in range(len(x)):
        for angle in range(x.shape[2]):
            out[sample, :, angle] = np.interp(target_hz, source_hz, x[sample, :, angle].real) + 1j * np.interp(target_hz, source_hz, x[sample, :, angle].imag)
    return out


def harmonize(a, b, band_a, band_b):
    low = max(band_a[0], band_b[0], a["frequency_hz"][0], b["frequency_hz"][0])
    high = min(band_a[1], band_b[1], a["frequency_hz"][-1], b["frequency_hz"][-1])
    if low >= high: raise ValueError("No overlapping frequency band")
    n_freq = min(np.count_nonzero((a["frequency_hz"] >= low) & (a["frequency_hz"] <= high)),
                 np.count_nonzero((b["frequency_hz"] >= low) & (b["frequency_hz"] <= high)))
    if n_freq < 2:
        raise ValueError("Common frequency band contains fewer than two measured samples")
    n_angle = min(a["x"].shape[2], b["x"].shape[2])
    common = np.linspace(low, high, n_freq)
    def one(dataset):
        return resample(interpolate(dataset["x"], dataset["frequency_hz"], common), n_angle, axis=2)
    # Angular samples are resampled by scan index. The reported experiment did
    # not physically rotate the different Gen1/Gen2 antenna origins into one
    # frame, so its result cannot identify a particular hardware mechanism.
    return one(a), one(b), common


def calibrate(x, y, groups, seed):
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2: raise ValueError("At least two source phantom groups are required")
    cv = GroupKFold(n_splits=n_splits)
    best = None
    for c in DEFAULT_C_GRID:
        scores = np.full(len(y), np.nan)
        for train, val in cv.split(x, y, groups):
            model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, max_iter=5000, solver="liblinear", random_state=seed))])
            model.fit(x[train], y[train]); scores[val] = probability(model, x[val])
        auc = float(roc_auc_score(y, scores))
        if best is None or auc > best[0]: best = (auc, c, scores)
    return best[1], best[0], choose_threshold(y, best[2])


def direction(source, target, source_x, target_x, common, seed):
    xs, xt = extract_single_port_features(source_x), extract_single_port_features(target_x)
    c, source_auc, threshold = calibrate(xs, source["y"], source["groups"], seed)
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, max_iter=5000, solver="liblinear", random_state=seed))])
    model.fit(xs, source["y"])
    return {"source_n": len(source["y"]), "source_groups": len(np.unique(source["groups"])), "target_n": len(target["y"]), "target_groups": len(np.unique(target["groups"])), "source_group_oof_auc": source_auc, "C": c, "source_oof_threshold": threshold, "common_frequency_hz": [float(common[0]), float(common[-1]), len(common)], "common_angles": source_x.shape[2], "target_metrics": evaluate_scores(target["y"], probability(model, xt), threshold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen1", type=Path, required=True)
    parser.add_argument("--gen2", type=Path, required=True)
    parser.add_argument("--gen1-band-ghz", nargs=2, type=float, default=(1.0, 8.0))
    parser.add_argument("--gen2-band-ghz", nargs=2, type=float, default=(1.0, 8.0))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    gen1, gen2 = load(args.gen1), load(args.gen2)
    b1, b2 = tuple(np.asarray(args.gen1_band_ghz) * 1e9), tuple(np.asarray(args.gen2_band_ghz) * 1e9)
    x1, x2, common = harmonize(gen1, gen2, b1, b2)
    save_json(args.output, {"port": "S11 (the only S-parameter acquired in Gen1)",
        "input_npz_sha256": {"gen1": sha256(args.gen1), "gen2": sha256(args.gen2)},
        "native_axes": {name: {"frequency_hz": [float(data["frequency_hz"][0]), float(data["frequency_hz"][-1]), len(data["frequency_hz"])],
                               "angle_deg": [float(data["angle_deg"][0]), float(data["angle_deg"][-1]), len(data["angle_deg"])],
                               "calibration": data["calibration"]} for name, data in (("gen1", gen1), ("gen2", gen2))},
        "harmonization": "stored native frequency axes interpolated to the common band; circular Fourier resampling by angle index; per-sample envelope normalization in feature extraction; no physical antenna-origin rotation applied",
        "interpretation_limit": "Observed acquisition-domain shift only; this run cannot isolate a hardware mechanism",
        "Gen1_to_Gen2": direction(gen1, gen2, x1, x2, common, args.seed),
        "Gen2_to_Gen1": direction(gen2, gen1, x2, x1, common, args.seed)})
    print("wrote", args.output.resolve())


if __name__ == "__main__": main()
