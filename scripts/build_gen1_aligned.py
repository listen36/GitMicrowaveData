"""Build an S11-only aligned Gen1 NPZ from official UM-BMID clean pickles."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def finite(value) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-published-geometry", action="store_true",
                        help="Require the documented 1,001-frequency/72-angle UM-BMID cohort")
    args = parser.parse_args()
    with args.data.open("rb") as f: x11 = np.asarray(pickle.load(f))
    with args.metadata.open("rb") as f: metadata = pickle.load(f)
    if len(x11) != len(metadata): raise ValueError("Data and metadata lengths differ")
    if args.require_published_geometry and x11.shape[1:] != (1001, 72):
        raise ValueError("Published UM-BMID geometry requires 1,001 frequency samples and 72 angular samples")
    keep, labels, groups, ids, tumor = [], [], [], [], []
    for i, md in enumerate(metadata):
        group = str(md.get("phant_id", "")).strip()
        # Clean files should already omit empty-chamber measurements. Retain
        # physical breast-phantom scans and reject only missing phantom IDs.
        if not group or group.lower() in {"nan", "none"}: continue
        radius = finite(md.get("tum_rad"))
        keep.append(i); labels.append(int(np.isfinite(radius) and radius > 0))
        groups.append(group); ids.append(str(md.get("id", i)))
        tumor.append([finite(md.get(k)) for k in ("tum_x", "tum_y", "tum_z", "tum_rad")])
    payload = {
        "X11": x11[np.asarray(keep)], "y": np.asarray(labels, dtype=np.int64),
        "group_ids": np.asarray(groups), "sample_ids": np.asarray(ids),
        "tumor_params": np.asarray(tumor, dtype=np.float32),
        "frequency_hz": np.linspace(1e9, 8e9, x11.shape[1]),
        "angle_deg": np.arange(x11.shape[2]) * (360.0 / x11.shape[2]),
        "calibration": np.asarray("empty-chamber complex frequency-domain subtraction using emp_ref_id"),
        "acquisition_axes_source": np.asarray("UM-BMID equipment and scan-geometry documentation; not scanner headers"),
    }
    if len(np.unique(payload["y"])) != 2: raise ValueError("Gen1 selection does not contain both classes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print("saved", args.output.resolve())
    print("shape", payload["X11"].shape, "classes", np.bincount(payload["y"]).tolist(), "groups", len(np.unique(payload["group_ids"])))


if __name__ == "__main__": main()
