"""Validate the aligned NPZ schema and, optionally, a fixed split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from umbmid_validation.core import assert_group_disjoint, load_aligned_npz, load_fixed_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    args = parser.parse_args()
    data = load_aligned_npz(args.dataset)
    print(f"samples={len(data.y)}")
    print(f"signal_shape={data.x11.shape}")
    print(f"classes={np.bincount(data.y).tolist()}")
    print(f"phantom_groups={len(np.unique(data.groups))}")
    print(f"unique_sample_ids={len(np.unique(data.sample_ids))}")
    if args.split:
        split = load_fixed_split(args.split, len(data.y))
        assert_group_disjoint(split, data.groups)
        for name, idx in split.items():
            print(
                f"{name}: samples={len(idx)}, classes={np.bincount(data.y[idx], minlength=2).tolist()}, "
                f"groups={sorted(set(data.groups[idx]))}"
            )


if __name__ == "__main__":
    main()

