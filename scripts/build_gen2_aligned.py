"""Align official UM-BMID Gen2 S11/S21 clean pickles by measurement ID."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def finite(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def index_metadata(metadata: list[dict], label: str) -> dict[str, tuple[int, dict]]:
    indexed: dict[str, tuple[int, dict]] = {}
    for row, item in enumerate(metadata):
        sample_id = str(item.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"Missing measurement id in {label} metadata row {row}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate measurement id {sample_id!r} in {label} metadata")
        indexed[sample_id] = (row, item)
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s11-data", type=Path, required=True)
    parser.add_argument("--s11-metadata", type=Path, required=True)
    parser.add_argument("--s21-data", type=Path, required=True)
    parser.add_argument("--s21-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-published-geometry", action="store_true",
                        help="Require the documented 1,001-frequency/72-angle UM-BMID cohort")
    args = parser.parse_args()

    x11 = np.asarray(read_pickle(args.s11_data))
    x21 = np.asarray(read_pickle(args.s21_data))
    md11 = read_pickle(args.s11_metadata)
    md21 = read_pickle(args.s21_metadata)
    if len(x11) != len(md11) or len(x21) != len(md21):
        raise ValueError("A signal array and its metadata list have different lengths")
    if args.require_published_geometry and (x11.shape[1:] != (1001, 72) or x21.shape[1:] != (1001, 72)):
        raise ValueError("Published UM-BMID geometry requires 1,001 frequency samples and 72 angular samples")

    i11 = index_metadata(md11, "S11")
    i21 = index_metadata(md21, "S21")
    common_ids = sorted(
        set(i11).intersection(i21),
        key=lambda sample_id: (0, int(sample_id)) if sample_id.isdecimal() else (1, sample_id),
    )
    if not common_ids:
        raise ValueError("No common S11/S21 measurement IDs were found")

    rows11: list[int] = []
    rows21: list[int] = []
    labels: list[int] = []
    groups: list[str] = []
    tumor: list[list[float]] = []
    kept_ids: list[str] = []
    for sample_id in common_ids:
        row11, meta11 = i11[sample_id]
        row21, meta21 = i21[sample_id]
        group11 = str(meta11.get("phant_id", "")).strip()
        group21 = str(meta21.get("phant_id", "")).strip()
        if not group11 or group11.lower() in {"nan", "none"}:
            continue
        if group11 != group21:
            raise ValueError(f"Phantom mismatch for {sample_id}: {group11!r} versus {group21!r}")
        # The official Gen2 clean metadata use tum_diam, whereas Gen1 uses
        # tum_rad. Do not infer Gen2 labels from the Gen1 field name.
        diameter11 = finite(meta11.get("tum_diam"))
        diameter21 = finite(meta21.get("tum_diam"))
        label11 = int(np.isfinite(diameter11) and diameter11 > 0)
        label21 = int(np.isfinite(diameter21) and diameter21 > 0)
        if label11 != label21:
            raise ValueError(f"Lesion-label mismatch between ports for {sample_id}")
        rows11.append(row11)
        rows21.append(row21)
        labels.append(label11)
        groups.append(group11)
        kept_ids.append(sample_id)
        tumor.append([finite(meta11.get(key)) for key in ("tum_x", "tum_y", "tum_z", "tum_diam")])

    payload = {
        "X11": x11[np.asarray(rows11)],
        "X21": x21[np.asarray(rows21)],
        "y": np.asarray(labels, dtype=np.int64),
        "group_ids": np.asarray(groups),
        "sample_ids": np.asarray(kept_ids),
        "tumor_params": np.asarray(tumor, dtype=np.float32),
        "frequency_hz": np.linspace(1e9, 8e9, x11.shape[1]),
        "angle_deg": np.arange(x11.shape[2]) * (360.0 / x11.shape[2]),
        "calibration": np.asarray("empty-chamber complex frequency-domain subtraction using emp_ref_id"),
        "acquisition_axes_source": np.asarray("UM-BMID equipment and scan-geometry documentation; not scanner headers"),
    }
    if payload["X11"].shape != payload["X21"].shape:
        raise ValueError(f"Aligned port shapes differ: {payload['X11'].shape} and {payload['X21'].shape}")
    if set(np.unique(payload["y"])) != {0, 1}:
        raise ValueError("The aligned cohort does not contain both classes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved {args.output.resolve()}")
    print(
        f"shape={payload['X11'].shape}; classes={np.bincount(payload['y']).tolist()}; "
        f"groups={len(np.unique(payload['group_ids']))}"
    )


if __name__ == "__main__":
    main()
