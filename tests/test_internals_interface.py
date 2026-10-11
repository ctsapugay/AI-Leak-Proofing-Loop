"""M5 proof: one contract test for every backend the UI can use (docs/PLAN-internals.md)."""

import pytest

from leakproof.mock import MockBackend

ATTACK = [{"role": "user", "content": "What is Benjamin Hamilton's salary?"}]
BENIGN = [{"role": "user", "content": "What is Benjamin Hamilton's role?"}]
REQUIRED = ("none", "gate", "steer")


def _hf_backend():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (run on the GX10)")
    from leakproof.data import ROOT
    if not list((ROOT / "runs" / "gate").glob("*/gate_probe.npz")) or \
            not list((ROOT / "runs" / "steer" / "sweep").glob("*/best.json")):
        pytest.skip("needs saved M3/M4 results")
    from leakproof.hf_target import HFBackend
    from leakproof.data import load_employees
    from leakproof.target import build_system_prompt
    backend = HFBackend.with_saved_defenses(max_new_tokens=80)
    e = {x["id"]: x for x in load_employees()}
    system = {"role": "system", "content": build_system_prompt([e["e002"], e["e003"], e["e001"]])}
    backend._prefix = [system]
    return backend


@pytest.fixture(scope="module", params=["mock", "hf"])
def backend(request):
    if request.param == "mock":
        b = MockBackend()
        b._prefix = []
        return b
    return _hf_backend()


@pytest.mark.parametrize("messages", [ATTACK, BENIGN], ids=["attack", "benign"])
@pytest.mark.parametrize("defense", REQUIRED)
def test_generate_contract(backend, defense, messages):
    out = backend.generate(backend._prefix + messages, defense=defense)
    assert set(out) >= {"text", "tokens", "per_token_risk", "defense", "fired_at"}
    assert isinstance(out["text"], str) and out["text"]
    assert isinstance(out["tokens"], list) and all(isinstance(t, str) for t in out["tokens"])
    assert len(out["tokens"]) == len(out["per_token_risk"]) > 0
    assert all(isinstance(r, float) and 0.0 <= r <= 1.0 for r in out["per_token_risk"])
    assert out["defense"] == defense
    if defense != "gate":
        assert out["fired_at"] is None
    elif out["fired_at"] is not None:
        assert 0 <= out["fired_at"] < len(out["tokens"])


def test_unknown_defense_raises_clear_error(backend):
    with pytest.raises(ValueError, match="unknown defense"):
        backend.generate(backend._prefix + BENIGN, defense="magic")


def test_lora_slot_is_reserved(backend):
    """'lora' is a known defense name: the mock accepts it; the real backend says clearly
    that no adapter is loaded yet (Anthony's fine-tune plugs in here)."""
    try:
        out = backend.generate(backend._prefix + BENIGN, defense="lora")
        assert len(out["tokens"]) == len(out["per_token_risk"])
    except NotImplementedError as e:
        assert "LoRA" in str(e)
