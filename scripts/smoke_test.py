"""Data-free smoke test of the public feature and split contracts."""

from __future__ import annotations

import numpy as np

from umbmid_validation.core import assert_group_disjoint, extract_physics_features, time_axis_ns


def main() -> None:
    rng = np.random.default_rng(20260910)
    shape = (6, 80, 72)
    x11 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    x21 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    features = extract_physics_features(x11, x21)
    assert features.shape == (6, 217) and np.isfinite(features).all()
    groups = np.array(["A", "A", "B", "B", "C", "C"])
    assert_group_disjoint(
        {"train": np.array([0, 1]), "val": np.array([2, 3]), "test": np.array([4, 5])},
        groups,
    )
    times, _ = time_axis_ns(1001, 1e9, 8e9)
    assert np.isclose(times[5], 0.7135721421)
    print("PASS: synthetic dual-port features, phantom-disjoint split, and ns axis")


if __name__ == "__main__":
    main()
