"""Create sample-index arrays from the transparent manuscript group split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/fixed_split_groups.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as archive:
        if "group_ids" not in archive.files:
            raise ValueError("Dataset is missing group_ids")
        groups = np.asarray(archive["group_ids"]).astype(str)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    requested = {name: set(config[name]) for name in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = requested[left].intersection(requested[right])
        if overlap:
            raise ValueError(f"Group overlap between {left} and {right}: {sorted(overlap)}")
    observed = set(groups)
    configured = set().union(*requested.values())
    if observed != configured:
        raise ValueError(
            f"Dataset/config group mismatch; missing in config={sorted(observed-configured)}, "
            f"missing in dataset={sorted(configured-observed)}"
        )
    split = {name: np.flatnonzero(np.isin(groups, sorted(values))) for name, values in requested.items()}
    joined = np.concatenate(list(split.values()))
    if len(joined) != len(groups) or len(np.unique(joined)) != len(groups):
        raise ValueError("The configured groups do not partition every sample exactly once")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{f"{name}_idx": idx for name, idx in split.items()})
    print(f"saved {args.output.resolve()}")
    for name, idx in split.items():
        print(f"{name}: n={len(idx)}, groups={len(np.unique(groups[idx]))}")


if __name__ == "__main__":
    main()

