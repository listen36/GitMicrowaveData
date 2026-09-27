from __future__ import annotations

import numpy as np
import pytest

from umbmid_validation.core import (
    assert_group_disjoint,
    choose_threshold,
    extract_physics_features,
    load_fixed_split,
    permute_angles,
    time_axis_ns,
)


def test_physics_feature_count_is_217() -> None:
    rng = np.random.default_rng(7)
    shape = (4, 80, 72)
    x11 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    x21 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    features = extract_physics_features(x11, x21)
    assert features.shape == (4, 217)
    assert np.isfinite(features).all()


def test_angle_permutation_preserves_port_pairing_and_values() -> None:
    base = np.arange(2 * 45 * 12).reshape(2, 45, 12)
    x11 = base.astype(complex)
    x21 = (base + 10_000).astype(complex)
    out11, out21 = permute_angles(x11, x21, seed=11)
    np.testing.assert_array_equal(out21 - out11, np.full_like(out11, 10_000))
    for sample in range(len(base)):
        np.testing.assert_array_equal(np.sort(out11[sample], axis=1), np.sort(x11[sample], axis=1))


def test_fixed_split_validation_and_leakage_detection(tmp_path) -> None:
    split_path = tmp_path / "split.npz"
    np.savez(split_path, train_idx=[0, 1], val_idx=[2], test_idx=[3])
    split = load_fixed_split(split_path, 4)
    assert_group_disjoint(split, np.asarray(["a", "a", "b", "c"]))
    with pytest.raises(ValueError, match="Group leakage"):
        assert_group_disjoint(split, np.asarray(["a", "a", "b", "a"]))


def test_time_axis_matches_reported_gen2_spacing() -> None:
    times, spacing = time_axis_ns(1001, 1e9, 8e9)
    assert spacing == pytest.approx(0.1427144284)
    assert times[5] == pytest.approx(0.7135721421)
    assert times[40] == pytest.approx(5.708577136)


def test_threshold_is_first_balanced_accuracy_maximizer() -> None:
    y = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold = choose_threshold(y, scores)
    assert 0.2 < threshold <= 0.8

