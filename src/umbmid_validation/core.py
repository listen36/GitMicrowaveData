"""Shared data, feature, validation, and reporting utilities.

The functions in this module deliberately keep phantom groups out of one
another's development and evaluation partitions.  They operate on the aligned
NPZ format produced by ``scripts/build_gen2_aligned.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, ParameterGrid


DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass
class Dataset:
    x11: np.ndarray
    x21: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_ids: np.ndarray
    tumor_params: np.ndarray | None = None


@dataclass
class Metrics:
    threshold: float
    roc_auc: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def load_aligned_npz(path: str | Path) -> Dataset:
    """Load and validate an aligned dual-port dataset."""
    with np.load(path, allow_pickle=False) as z:
        required = {"X11", "X21", "y", "group_ids", "sample_ids"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
        data = Dataset(
            x11=np.asarray(z["X11"]),
            x21=np.asarray(z["X21"]),
            y=np.asarray(z["y"], dtype=int),
            groups=np.asarray(z["group_ids"]).astype(str),
            sample_ids=np.asarray(z["sample_ids"]).astype(str),
            tumor_params=np.asarray(z["tumor_params"])
            if "tumor_params" in z.files
            else None,
        )
    n = len(data.y)
    if data.x11.shape != data.x21.shape or data.x11.shape[0] != n:
        raise ValueError("S11, S21, and label dimensions are inconsistent")
    if data.x11.ndim != 3:
        raise ValueError("Expected scattering arrays shaped samples x frequency x angle")
    if set(np.unique(data.y)) != {0, 1}:
        raise ValueError("Binary labels must contain both 0 and 1")
    return data


def load_fixed_split(path: str | Path, n_samples: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        split = {name: np.asarray(z[f"{name}_idx"], dtype=int) for name in ("train", "val", "test")}
    joined = np.concatenate(list(split.values()))
    if len(joined) != n_samples or sorted(joined.tolist()) != list(range(n_samples)):
        raise ValueError("Split indices must partition every sample exactly once")
    return split


def assert_group_disjoint(split: dict[str, np.ndarray], groups: np.ndarray) -> None:
    group_sets = {name: set(groups[idx]) for name, idx in split.items()}
    names = list(group_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = group_sets[left].intersection(group_sets[right])
            if overlap:
                raise ValueError(f"Group leakage between {left} and {right}: {sorted(overlap)}")


def ifft_envelope(x: np.ndarray, gate: tuple[int, int] = (5, 40)) -> np.ndarray:
    """Return per-sample max-normalized IFFT magnitude in a half-open gate."""
    start, stop = gate
    if not (0 <= start < stop <= x.shape[1]):
        raise ValueError(f"Invalid gate {gate} for {x.shape[1]} frequency samples")
    env = np.abs(np.fft.ifft(x, axis=1))[:, start:stop, :]
    scale = np.max(env, axis=(1, 2), keepdims=True)
    return env / np.maximum(scale, np.finfo(float).eps)


def _port_features(env: np.ndarray, window_bins: int = 5, harmonics: int = 8) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n, n_time, n_angle = env.shape
    if n_time % window_bins:
        raise ValueError("The time gate must divide evenly into contiguous windows")
    if harmonics >= n_angle // 2:
        raise ValueError("Too many angular harmonics for the number of angles")
    rows: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    harmonic_blocks: list[np.ndarray] = []
    for start in range(0, n_time, window_bins):
        block = env[:, start : start + window_bins, :]
        profile = block.mean(axis=1)
        mean = profile.mean(axis=1)
        # The manuscript's "mean energy" is the mean normalized envelope
        # amplitude in the window (not squared power).
        energy = mean
        cv = profile.std(axis=1) / np.maximum(mean, np.finfo(float).eps)
        dynamic_range = profile.max(axis=1) - profile.min(axis=1)
        spectrum = np.abs(np.fft.rfft(profile, axis=1))
        # Normalize angular magnitudes by total spectral L2 energy.  This is
        # stable when the DC term is small and matches the reported Gen2
        # holdout baseline (AUC 0.925) under the fixed split.
        spectral_norm = np.sqrt(np.sum(spectrum**2, axis=1, keepdims=True))
        harmonic = spectrum[:, 1 : harmonics + 1] / np.maximum(spectral_norm, np.finfo(float).eps)
        rows.append(np.column_stack((energy, cv, dynamic_range, harmonic)))
        energies.append(energy)
        harmonic_blocks.append(harmonic)
    return np.concatenate(rows, axis=1), {
        "energy": np.column_stack(energies),
        "harmonics": np.stack(harmonic_blocks, axis=1),
    }


def extract_physics_features(
    x11: np.ndarray,
    x21: np.ndarray,
    gate: tuple[int, int] = (5, 40),
    window_bins: int = 5,
    harmonics: int = 8,
) -> np.ndarray:
    """Build S11, S21, and cross-port energy/harmonic fusion features."""
    env11, env21 = ifft_envelope(x11, gate), ifft_envelope(x21, gate)
    f11, aux11 = _port_features(env11, window_bins, harmonics)
    f21, aux21 = _port_features(env21, window_bins, harmonics)
    energy_delta = np.abs(aux21["energy"] - aux11["energy"])
    harmonic_delta = np.abs(aux21["harmonics"] - aux11["harmonics"]).reshape(len(x11), -1)
    return np.concatenate((f11, f21, energy_delta, harmonic_delta), axis=1)


def extract_single_port_features(
    x: np.ndarray,
    gate: tuple[int, int] = (5, 40),
    window_bins: int = 5,
    harmonics: int = 8,
) -> np.ndarray:
    """Build the same physics representation for one available S-parameter."""
    features, _ = _port_features(ifft_envelope(x, gate), window_bins, harmonics)
    return features


def raw_gate_features(x11: np.ndarray, x21: np.ndarray, gate: tuple[int, int] = (5, 40)) -> np.ndarray:
    env = np.stack((ifft_envelope(x11, gate), ifft_envelope(x21, gate)), axis=1)
    return env.reshape(len(env), -1)


def cnn_tensor(x11: np.ndarray, x21: np.ndarray, gate: tuple[int, int] = (5, 40)) -> np.ndarray:
    return np.stack((ifft_envelope(x11, gate), ifft_envelope(x21, gate)), axis=1).astype(np.float32)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select the balanced-accuracy threshold using the legacy 1,000-point grid.

    The recovered UM-BMID analysis code evaluates thresholds from 0 to 1 and
    retains the first maximizer.  Keeping that convention makes operating-point
    results comparable while AUC remains threshold independent.
    """
    candidates = np.linspace(0.0, 1.0, 1000)
    predictions = scores[None, :] >= candidates[:, None]
    positive = y_true == 1
    negative = ~positive
    sensitivity = predictions[:, positive].mean(axis=1)
    specificity = (~predictions[:, negative]).mean(axis=1)
    values = 0.5 * (sensitivity + specificity)
    return float(candidates[int(np.argmax(values))])


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Metrics:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return Metrics(
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, scores)),
        accuracy=float(accuracy_score(y_true, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
        sensitivity=float(recall_score(y_true, pred, zero_division=0)),
        specificity=float(tn / (tn + fp)) if tn + fp else float("nan"),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def probability(model: Any, x: np.ndarray) -> np.ndarray:
    scores = model.predict_proba(x)
    return np.asarray(scores[:, 1] if scores.ndim == 2 else scores, dtype=float)


def validation_search(
    estimator: Any,
    param_grid: dict[str, Iterable[Any]],
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[Any, dict[str, Any], float, float]:
    """Tune solely on the fixed validation split and return its threshold."""
    best_key: tuple[float, str] | None = None
    best: tuple[Any, dict[str, Any], float, float] | None = None
    for params in ParameterGrid(param_grid):
        candidate = clone(estimator).set_params(**params)
        candidate.fit(x[train_idx], y[train_idx])
        scores = probability(candidate, x[val_idx])
        auc = float(roc_auc_score(y[val_idx], scores))
        key = (auc, json.dumps(params, sort_keys=True))
        if best_key is None or key > best_key:
            best_key = key
            best = (candidate, params, choose_threshold(y[val_idx], scores), auc)
    assert best is not None
    return best


def bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 3000,
    seed: int = 20260910,
) -> np.ndarray:
    """Stratified non-parametric sample bootstrap of ROC AUC."""
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == c) for c in (0, 1)]
    values = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = np.concatenate([rng.choice(part, len(part), replace=True) for part in class_indices])
        values[i] = roc_auc_score(y_true[idx], scores[idx])
    return values


def json_ready(value: Any) -> Any:
    if isinstance(value, Metrics):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_ready), encoding="utf-8")


def time_axis_ns(n_frequency: int, f_min_hz: float = 1e9, f_max_hz: float = 8e9) -> tuple[np.ndarray, float]:
    """Return non-negative IFFT sample times and spacing for a linear sweep."""
    df = (f_max_hz - f_min_hz) / (n_frequency - 1)
    dt_s = 1.0 / (n_frequency * df)
    return np.arange(n_frequency) * dt_s * 1e9, dt_s * 1e9


def permute_angles(x11: np.ndarray, x21: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Destroy angular coherence while retaining per-sample marginals and port pairing."""
    rng = np.random.default_rng(seed)
    out11, out21 = np.empty_like(x11), np.empty_like(x21)
    for i in range(len(x11)):
        order = rng.permutation(x11.shape[2])
        # Index the 2-D sample first so NumPy does not move the advanced-index
        # dimension to the front (which would yield angle x frequency).
        out11[i] = x11[i][:, order]
        out21[i] = x21[i][:, order]
    return out11, out21
