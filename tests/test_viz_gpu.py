"""V2/V3 proofs on the real model (GX10 only; skip without CUDA).

Needs the other session's gate probe and steering setting (read from LEAKPROOF_SRC_RUNS,
default ~/clara/AI-Hackathon/runs) and runs/viz/units.json from `python -m leakproof.viz fit`.
One model load (~16 GB); check `free -g`, `nvidia-smi` and `tmux ls` first (docs/PLAN-viz.md).

    OMP_NUM_THREADS=8 LEAKPROOF_SRC_RUNS=~/clara/AI-Hackathon/runs .venv/bin/python -m pytest tests/test_viz_gpu.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (run on the GX10)")

from leakproof.viz import VIZ_DIR, load_fitted, trace  # noqa: E402
from tests.test_viz_contract import check_trace  # noqa: E402

SRC = Path(os.environ.get("LEAKPROOF_SRC_RUNS", "~/clara/AI-Hackathon/runs")).expanduser()
ID_TOL = 1e-4  # risk units
N_OFFLINE = 10


def sigmoid_of_units(risk, u):
    return 1 / (1 + np.exp(-(np.asarray(risk) - u.c) / u.a))


@pytest.fixture(scope="module")
def backend():
    from leakproof.hf_target import HFBackend
    gate = sorted((SRC / "gate").glob("*/gate_probe.npz"))[-1]
    best = sorted((SRC / "steer" / "sweep").glob("*/best.json"))[-1]
    b = HFBackend.with_saved_defenses(gate_npz=gate, best_json=best, batch_size=1)
    b.capture_acts = False
    return b


@pytest.fixture(scope="module")
def fitted():
    return load_fitted()


@pytest.fixture(scope="module")
def cases():
    """Single-turn held-out attack conversations from the M0 default run, rebuilt like collect does."""
    from leakproof.data import load_employees
    from leakproof.target import build_system_prompt
    split = json.loads((SRC / "steer" / "split.json").read_text())
    held = set(split["test_templates"])
    by_id = {e["id"]: e for e in load_employees()}
    run = SRC.parent / split["runs"][0]
    out = []
    for line in (run / "answers.jsonl").open():
        a = json.loads(line)
        if a["kind"] != "attack" or len(a["turns"]) != 1 or a["id"].split("@")[0] not in held:
            continue
        msgs = [{"role": "system", "content": build_system_prompt([by_id[i] for i in a["context_ids"]], a["strictness"])},
                {"role": "user", "content": a["turns"][0]}]
        out.append({"id": a["id"], "label": a["label"], "messages": msgs, "reply": a["replies"][0], "run": run})
    out.sort(key=lambda c: (c["label"] != "full", c["id"]))
    full = [c for c in out if c["label"] == "full"][:N_OFFLINE // 2]
    none = [c for c in out if c["label"] == "none"][:N_OFFLINE // 2]
    return full + none


@pytest.fixture(scope="module")
def leak_case(cases):
    return cases[0]


@pytest.mark.parametrize("defense", ["none", "gate", "steer"])
def test_contract_on_real_model(backend, fitted, leak_case, defense):
    tr = trace(backend, leak_case["messages"], defense, fitted)
    check_trace(tr, defense, tol=ID_TOL)
    assert tr["request"]["probe_version"].startswith(fitted["probe_version"])
    assert tr["request"]["prompt_risk"] is not None


def test_benign_contract(backend, fitted):
    msgs = [{"role": "user", "content": "Which team is Dana Lee on?"}]
    for defense in ("none", "steer"):
        check_trace(trace(backend, msgs, defense, fitted), defense, tol=ID_TOL)


@pytest.mark.parametrize("defense", ["none", "gate", "steer"])
def test_matches_generate(backend, fitted, leak_case, defense):
    """Same tokens as generate(), and risk == generate()'s per_token_risk (same probe, same input)."""
    tr = trace(backend, leak_case["messages"], defense, fitted)
    g = backend.generate(leak_case["messages"], defense=defense)
    assert [backend.tok.decode([t["token_id"]]) for t in tr["tokens"]] == g["tokens"]
    p = sigmoid_of_units([t["risk"] for t in tr["tokens"]], fitted["units"])
    assert np.abs(p - np.array(g["per_token_risk"])).max() < 1e-4
    assert tr["request"]["gate_fired_pos"] == g["fired_at"]


def test_live_gate_matches_offline(backend, fitted, cases):
    """The gate's live per-token risks (hooked during generation) vs the trace's re-run pass."""
    fired = 0
    for c in cases:
        tr = trace(backend, c["messages"], "gate", fitted)
        live = np.array(tr["live_gate_risks"])
        p = sigmoid_of_units([t["risk"] for t in tr["tokens"]], fitted["units"])[:len(live)]
        assert np.abs(p - live).max() < 0.02, c["id"]
        fired += tr["request"]["gate_fired_pos"] is not None
    assert fired > 0  # at least one full leak in the set trips the gate, or the check is vacuous


def test_last_token_matches_cached_answer_mean(backend, fitted, cases):
    """Live trace vs the M0 cache: the last token's risk is the probe on the whole answer mean."""
    from leakproof.probe import load_runs
    from leakproof.viz import Probe
    acts, index = load_runs([cases[0]["run"]])
    row = {r["id"]: i for i, r in enumerate(index)}
    probe = Probe.of(backend.risk_probe)
    checked = 0
    for c in cases:
        tr = trace(backend, c["messages"], "none", fitted)
        if tr["request"]["response"] != c["reply"]:
            continue  # a greedy tie flipped by batch padding (PLAN-internals M0); not comparable
        cached = probe.logit(acts["answer_mean"][row[c["id"]], probe.layer].astype(np.float32))
        p_cached = 1 / (1 + np.exp(-cached))
        p_live = sigmoid_of_units(tr["tokens"][-1]["risk"], fitted["units"])
        assert abs(p_live - p_cached) < 0.02, c["id"]
        checked += 1
    assert checked >= N_OFFLINE // 2


def test_steering_shows_up_and_has_no_side_effects(backend, fitted, leak_case):
    before = trace(backend, leak_case["messages"], "none", fitted)
    st = trace(backend, leak_case["messages"], "steer", fitted)
    after = trace(backend, leak_case["messages"], "none", fitted)
    assert [t["token_id"] for t in before["tokens"]] == [t["token_id"] for t in after["tokens"]]
    assert all(t["steer"] is None for t in before["tokens"])
    removed = np.array([sum(t["steer"].values()) for t in st["tokens"]])
    assert np.abs(removed).max() > 0.05  # steering removes a visible amount of leak signal


def test_save_examples(backend, fitted, leak_case):
    """Real traces for Oleg (committed): a leak, the same attack gated and steered, a benign question."""
    out = VIZ_DIR / "examples"
    out.mkdir(parents=True, exist_ok=True)
    benign = [{"role": "user", "content": "Which team is Dana Lee on?"}]
    for name, msgs, defense in (("leak_none", leak_case["messages"], "none"),
                                ("leak_gate", leak_case["messages"], "gate"),
                                ("leak_steer", leak_case["messages"], "steer"),
                                ("benign_none", benign, "none")):
        tr = trace(backend, msgs, defense, fitted)
        tr.pop("live_gate_risks", None)
        (out / f"{name}.json").write_text(json.dumps(tr, indent=1) + "\n")
