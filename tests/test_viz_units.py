"""V1 proofs, pure logic: the probe's linear readout and the risk units (docs/PLAN-viz.md).

Mac: `uv run --with numpy --with scikit-learn pytest tests/test_viz_units.py`
"""

import numpy as np
import pytest

from leakproof.viz import Probe, Units, running_mean


def random_probe(dim=16, seed=0, layer=18):
    r = np.random.default_rng(seed)
    return Probe(layer, r.normal(size=dim).astype(np.float32), 0.3, r.normal(size=dim).astype(np.float32),
                 r.uniform(0.5, 2, size=dim).astype(np.float32), threshold=0.739, k=2)


def gate_logit(p: Probe, X):
    """The gate's own formula (GateProbe.risk before the sigmoid)."""
    return ((X - p.mean) / p.scale) @ p.w + p.b


def test_readout_equals_gate_formula():
    p = random_probe()
    X = np.random.default_rng(1).normal(size=(50, 16)).astype(np.float32) * 10
    assert np.allclose(p.logit(X), gate_logit(p, X.astype(np.float64)), atol=1e-9)


def test_readout_is_linear_so_running_mean_commutes():
    """Probe on the running mean == running mean of per-token logits minus... i.e. the readout of
    the mean is the mean of the readouts (what makes the decomposition exact)."""
    p = random_probe()
    w, b = p.readout()
    H = np.random.default_rng(2).normal(size=(30, 16))
    lhs = p.logit(np.cumsum(H, 0) / np.arange(1, 31)[:, None])
    rhs = running_mean(H @ w) + b
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_units_send_medians_to_zero_and_one():
    r = np.random.default_rng(3)
    z_safe, z_leak = r.normal(-3, 1, 101), r.normal(4, 2, 77)
    u = Units.fit(z_safe, z_leak)
    assert u(np.median(z_safe)) == pytest.approx(0, abs=1e-12)
    assert u(np.median(z_leak)) == pytest.approx(1, abs=1e-12)


def test_threshold_in_units_gives_same_decisions():
    r = np.random.default_rng(4)
    u = Units.fit(r.normal(-3, 1, 50), r.normal(4, 2, 50))
    z = r.normal(0, 5, 10_000)
    for p_thr in (0.1, 0.5, 0.739, 0.9):
        p = 1 / (1 + np.exp(-z))
        assert np.array_equal(u(z) > u.from_prob(p_thr), p > p_thr)


def test_units_reject_inverted_medians():
    with pytest.raises(ValueError):
        Units.fit(np.array([2.0, 3.0]), np.array([0.0, 1.0]))


def test_probe_save_load_round_trip(tmp_path):
    p = random_probe()
    np.savez(tmp_path / "g.npz", layer=p.layer, w=p.w, b=p.b, mean=p.mean, scale=p.scale,
             threshold=p.threshold, k=p.k, train_templates=np.array(["a", "b"]))
    q = Probe.load(tmp_path / "g.npz")
    assert q.version() == p.version() and q.threshold == p.threshold and q.k == p.k
    assert random_probe(seed=9).version() != p.version()
