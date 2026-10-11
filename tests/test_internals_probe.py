"""M1 proofs for the probe (docs/PLAN-internals.md). Pure numpy/sklearn: runs anywhere."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from leakproof.probe import make_probe, oof_scores, pick_C, shuffled, splits, template_of, threshold_for_false_alarm  # noqa: E402


def synthetic(n_templates=25, per_template=16, dim=256, shift=4.0, seed=0):
    """Rows grouped by template; leak rows are shifted `shift` std devs along one planted
    direction (shift=4 -> the best possible AUROC is ~0.998)."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"t{i}" for i in range(n_templates)], per_template)
    y = rng.integers(0, 2, len(groups))
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    X = rng.normal(size=(len(groups), dim)) + shift * y[:, None] * direction
    return X.astype(np.float32), y, groups, direction


def test_template_of():
    assert template_of("direct-salary@e002") == "direct-salary"


def test_no_template_on_both_sides_of_any_split():
    _, y, groups, _ = synthetic()
    folds = splits(y, groups)
    assert len(folds) == 5
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
    # every row is tested exactly once
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested.tolist()) == list(range(len(y)))


def test_planted_direction_is_recovered():
    X, y, groups, direction = synthetic()
    _, auc = oof_scores(X, y, groups, C=0.1)
    assert np.mean(auc) > 0.95
    probe = make_probe(0.1).fit(X, y)
    w = probe[1].coef_[0] / probe[0].scale_  # weights back in input units
    assert np.dot(w, direction) / np.linalg.norm(w) > 0.8


def test_shuffled_labels_give_chance():
    X, y, groups, _ = synthetic()
    _, auc = oof_scores(X, shuffled(y), groups, C=0.1)
    assert abs(np.mean(auc) - 0.5) < 0.1


def test_no_signal_gives_chance():
    X, y, groups, _ = synthetic(shift=0.0)
    _, auc = oof_scores(X, y, groups, C=0.1)
    assert abs(np.mean(auc) - 0.5) < 0.1


def test_reproducible_with_fixed_seed():
    X, y, groups, _ = synthetic()
    a = oof_scores(X, y, groups, C=0.1)
    b = oof_scores(X, y, groups, C=0.1)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_threshold_for_false_alarm():
    benign = np.linspace(0, 1, 101)
    thr = threshold_for_false_alarm(benign, 0.05)
    assert (benign > thr).mean() <= 0.05
    assert (benign > thr - 0.02).mean() > 0.05  # and it's the tightest such threshold


def test_pick_C_prefers_strongest_regularization_within_tolerance():
    assert pick_C({0.001: 0.9726, 0.01: 0.9756, 0.1: 0.976, 1.0: 0.978}) == 0.01
    assert pick_C({0.001: 0.90, 0.01: 0.95, 0.1: 0.99, 1.0: 0.991}) == 0.1
