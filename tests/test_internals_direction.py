"""M2 proofs for the steering direction (docs/PLAN-internals.md). Pure numpy/sklearn."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from leakproof.direction import diff_of_means, holdout_split, projection_auroc, random_units  # noqa: E402
from tests.test_internals_probe import synthetic  # noqa: E402


def test_vhat_is_unit_length_per_layer():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4, 16))
    y = np.array([0, 1] * 25)
    v = diff_of_means(X, y)
    assert v.shape == (4, 16)
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0)


def test_planted_mean_difference_is_recovered():
    # 400 rows, 64 dims, shift 4: sampling noise in the means has norm ~0.8, so the
    # expected cosine is ~0.98 (in 256 dims it would only be ~0.93).
    X, y, _, direction = synthetic(dim=64)
    assert diff_of_means(X, y) @ direction > 0.95


def test_vhat_points_from_safe_to_leak():
    X, y, _, _ = synthetic()
    v = diff_of_means(X, y)
    assert (X[y == 1] @ v).mean() > (X[y == 0] @ v).mean()


def test_heldout_projection_beats_random_directions():
    X, y, groups, _ = synthetic()
    train, test = holdout_split(y, groups)
    assert not set(groups[train]) & set(groups[test])
    assert len(train) + len(test) == len(y)
    assert 0.3 < len(set(groups[test])) / len(set(groups)) < 0.5
    real = projection_auroc(X[test], y[test], diff_of_means(X[train], y[train]))
    rand = [projection_auroc(X[test], y[test], u) for u in random_units(X.shape[1])]
    assert real > 0.95
    assert abs(np.mean(rand) - 0.5) < 0.1


def test_random_units_are_unit_and_reproducible():
    a, b = random_units(32, 5, seed=1), random_units(32, 5, seed=1)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
