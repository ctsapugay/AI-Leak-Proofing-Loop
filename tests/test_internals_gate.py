"""M4 proofs for the gate (docs/PLAN-internals.md).

Pure-logic tests need numpy/sklearn; model-backed tests need CUDA (GX10).
"""

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from leakproof.gate import (  # noqa: E402
    REFUSAL, GateProbe, first_fire, gate_trigger, keyword_flag, running_mean, train_gate_probe,
)
from leakproof.probe import make_probe  # noqa: E402


# --- trigger logic ----------------------------------------------------------------

@pytest.mark.parametrize("risks, expected", [
    ([], None),                          # empty answer
    ([0.1, 0.9, 0.1, 0.9], None),        # single spikes never fire
    ([0.1, 0.9, 0.9, 0.1], 2),           # fires on the 2nd consecutive token over
    ([0.5, 0.5, 0.5], None),             # exactly at threshold is not over
    ([0.9, 0.2, 0.9, 0.9, 0.9], 3),      # a dip resets the run
])
def test_gate_trigger(risks, expected):
    assert gate_trigger(risks, 0.5, 2) == expected


def test_gate_trigger_k3():
    assert gate_trigger([0.9, 0.9, 0.1, 0.9, 0.9, 0.9], 0.5, 3) == 5


def test_running_mean():
    H = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    assert running_mean(H).tolist() == [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]


def test_first_fire_collects_what_the_user_saw():
    turns = [{"fired_at": None, "emitted": "Hi."}, {"fired_at": 4, "emitted": "Her salary is $1"}]
    assert first_fire(turns) == (4, "Hi.\n\nHer salary is $1")
    assert first_fire([{"fired_at": None, "emitted": "x"}, None]) == (None, "")


def test_keyword_filter():
    assert keyword_flag("Her salary is $143,750.")
    assert keyword_flag("She was out for chemotherapy.")
    assert not keyword_flag("Dana Lee is a Senior Engineer on the Payments team in Austin.")


# --- probe reimplementation and held-out integrity -------------------------------

def _gp_from_pipeline(probe, layer=1):
    return GateProbe(layer, probe[1].coef_[0], float(probe[1].intercept_[0]), probe[0].mean_, probe[0].scale_)


def test_gate_risk_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 32)) * rng.uniform(0.5, 3, 32) + rng.normal(size=32)
    y = (X[:, 0] + rng.normal(size=200) > 0).astype(int)
    probe = make_probe(0.1).fit(X, y)
    assert np.allclose(_gp_from_pipeline(probe).risk(X), probe.predict_proba(X)[:, 1], atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    gp = GateProbe(18, np.arange(4.0), 0.5, np.ones(4), np.ones(4) * 2, 0.7, 3, ("a", "b"))
    gp.save(tmp_path / "g.npz")
    g2 = GateProbe.load(tmp_path / "g.npz")
    assert (g2.layer, g2.b, g2.threshold, g2.k, g2.train_templates) == (18, 0.5, 0.7, 3, ("a", "b"))
    assert np.array_equal(g2.w, gp.w)


def _fake_run(path, templates, seed):
    """A tiny collect run: acts [n, 3 layers, 8 dims] + index rows, labels alternating."""
    rng = np.random.default_rng(seed)
    (path / "acts").mkdir(parents=True)
    rows = []
    for i, t in enumerate(templates):
        for j in range(6):
            rows.append({"row": len(rows), "id": f"{t}@e{j:03d}", "kind": "attack", "strategy": "x",
                         "field": "salary", "label": "full" if (i + j) % 2 else "none"})
    rows.append({"row": len(rows), "id": "benign-role@e000", "kind": "benign", "strategy": "benign",
                 "field": "none", "label": "none"})
    for name in ("answer_mean", "last_prompt"):
        np.save(path / "acts" / f"{name}.npy", rng.normal(size=(len(rows), 3, 8)).astype(np.float16))
    (path / "acts" / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_gate_probe_never_trains_on_held_out_templates(tmp_path):
    templates = ["t-a", "t-b", "t-c", "t-d", "t-e", "t-f"]
    held = {"t-b", "t-e"}
    _fake_run(tmp_path / "run1", templates, 0)
    _fake_run(tmp_path / "run2", templates, 1)
    gp, data = train_gate_probe([tmp_path / "run1", tmp_path / "run2"], layer=1, C=0.1, held_out=held)
    assert not set(gp.train_templates) & held
    assert set(gp.train_templates) == set(templates) - held
    trained_ids = [r["id"] for r, used in zip(data["index"], data["train_rows"]) if used]
    assert trained_ids and not {i.split("@")[0] for i in trained_ids} & held
    assert all(r["kind"] == "attack" for r, used in zip(data["index"], data["train_rows"]) if used)


def test_real_gate_probe_excludes_split_json_templates():
    """On the GX10's real runs: the gate probe's training templates and split.json's
    held-out templates are disjoint."""
    from leakproof.data import ROOT
    from leakproof.direction import test_templates
    runs = sorted((ROOT / "runs" / "collect").glob("*__qwen2.5-7b-instruct__*/acts"))
    runs = [r.parent for r in runs if "163101" not in r.parent.name]  # skip the 1-target smoke run
    if len(runs) < 2 or not (ROOT / "runs" / "steer" / "split.json").exists():
        pytest.skip("needs the M0 runs with activations (GX10)")
    held = test_templates()
    gp, data = train_gate_probe(runs, layer=18, C=0.01, held_out=held)
    assert not set(gp.train_templates) & held
    used = {r["id"].split("@")[0] for r, u in zip(data["index"], data["train_rows"]) if u}
    assert not used & held and len(used) >= 10


# --- model-backed ------------------------------------------------------------------

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False
cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA (run on the GX10)")


@pytest.fixture(scope="module")
def backend():
    from leakproof.hf_target import HFBackend
    return HFBackend(max_new_tokens=60)


@pytest.fixture(scope="module")
def gp():
    """A real gate probe (train templates only) at layer 18."""
    from leakproof.data import ROOT
    from leakproof.direction import test_templates
    runs = [r.parent for r in sorted((ROOT / "runs" / "collect").glob("*__qwen2.5-7b-instruct__*/acts"))
            if "163101" not in r.parent.name]
    g, _ = train_gate_probe(runs, layer=18, C=0.01, held_out=test_templates())
    return g


def _convs():
    from tests.test_internals_steer import _convs as c
    return c()


@cuda
def test_live_risks_match_offline(backend, gp):
    """Risks computed during generation equal risks from a forward pass over the finished
    answer (running mean at layer 18), for every token the gate saw."""
    gp.threshold, gp.k = 2.0, 2  # never fires
    convs = _convs()
    backend.gate = gp
    try:
        live = backend.generate_batch(convs)
    finally:
        backend.gate = None
    enc = backend.encode(convs)
    P = enc["input_ids"].shape[1]
    seqs, mask = backend.generate_ids(enc)
    hs = backend.hidden_states(seqs, enc["attention_mask"], mask)[:, gp.layer, P:].float().cpu().numpy()
    worst = 0.0
    for b, g in enumerate(live):
        H = hs[b][mask[b].cpu().numpy()]
        offline = gp.risk(running_mean(H).astype(np.float32))
        live_r = np.array(g.gate["risks"])
        assert len(live_r) >= len(H) - 1 and len(live_r) > 0  # the gate saw every fed token
        m = min(len(live_r), len(H))
        worst = max(worst, float(np.abs(live_r[:m] - offline[:m]).max()))
    assert worst < 0.02, worst


@cuda
def test_gate_that_never_fires_changes_nothing(backend, gp):
    convs = _convs()
    baseline = [g.text for g in backend.generate_batch(convs)]
    gp.threshold, gp.k = 2.0, 2
    backend.gate = gp
    try:
        gated = backend.generate_batch(convs)
    finally:
        backend.gate = None
    assert [g.text for g in gated] == baseline
    assert all(g.gate["fired_at"] is None for g in gated)


@cuda
def test_gate_that_always_fires_refuses_at_token_k(backend, gp):
    gp.threshold, gp.k = -1.0, 2
    backend.gate = gp
    try:
        gated = backend.generate_batch(_convs())
    finally:
        backend.gate = None
    assert all(g.text == REFUSAL and g.gate["fired_at"] == 1 for g in gated)
    n_hooks = [len(layer._forward_hooks) for layer in backend.model.model.layers]
    assert backend.generate_batch(_convs())[0].gate is None
    assert [len(layer._forward_hooks) for layer in backend.model.model.layers] == n_hooks
