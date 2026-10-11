"""V2 proofs on a tiny randomly initialized Qwen2 (CPU, same decoder code as Qwen2.5-7B).

The decomposition must sum exactly to the risk, the risk must equal the probe on the running
mean of hidden_states[L] (the gate's input), the base must equal an independent readout of
the embeddings, and the steering edit must land in `steer`, not in `mlp_contrib`.

Mac: `uv run --with torch --with transformers --with numpy --with scikit-learn pytest tests/test_viz_decompose.py`
"""

from contextlib import contextmanager

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from leakproof.viz import Probe, Units, capturing, decompose, identity_residual, running_mean, stack  # noqa: E402

N_LAYERS, HIDDEN, L = 6, 32, 4  # probe at hidden_states[4] = output of decoder layer 3
P, T = 7, 9  # prompt length, answer length


try:
    from leakproof.steer import steering  # the real hook, once the other session's M3 is merged
except ImportError:
    @contextmanager
    def steering(model, directions, alpha):
        """Same as leakproof.steer.steering (prepended forward hook, h - alpha (h.v) v)."""
        handles = []
        try:
            for Lh, v in directions.items():
                def hook(_m, _i, out, v=v):
                    h = out[0] if isinstance(out, tuple) else out
                    hf, vf = h.float(), v.float().to(h.device)
                    new = (hf - alpha * (hf @ vf).unsqueeze(-1) * vf).to(h.dtype)
                    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
                handles.append(model.model.layers[Lh - 1].register_forward_hook(hook, prepend=True))
            yield
        finally:
            for h in handles:
                h.remove()


@pytest.fixture(scope="module", params=["float32", "bfloat16"])
def tiny(request):
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=101, hidden_size=HIDDEN, intermediate_size=64,
                                   num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=64)
    model = transformers.Qwen2ForCausalLM(cfg).to(getattr(torch, request.param)).eval()
    ids = torch.randint(0, 101, (1, P + T))
    r = np.random.default_rng(0)
    probe = Probe(L, r.normal(size=HIDDEN).astype(np.float32), 0.2, r.normal(size=HIDDEN).astype(np.float32) * 0.1,
                  r.uniform(0.5, 2, HIDDEN).astype(np.float32))
    units = Units(a=0.3, c=0.1, z_safe=-1 / 3, z_leak=3)
    v = {Lh: torch.nn.functional.normalize(torch.randn(HIDDEN), dim=0) for Lh in range(1, N_LAYERS + 1)}
    return model, ids, probe, units, v  # one direction per layer, like v-hat


def run(model, ids, probe, units, steer=None, alpha=1.0, capture_outside=False):
    """Trace-style pass: steering outside, capture inside (unless testing the wrong order)."""
    w, b = probe.readout()
    steered = sorted(steer or {})
    n_cap = max([L, *steered])
    cap_ctx = capturing(model, w, n_cap, positions=slice(P, P + T), steered=steered, keep_raw_at=(L - 1, P - 1))
    st_ctx = steering(model, steer or {}, alpha)
    with torch.no_grad():
        if capture_outside:
            with cap_ctx as rec, st_ctx:
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        else:
            with st_ctx, cap_ctx as rec:
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    d = decompose(stack(rec), b, units, L)
    return d, hs, rec


def gate_risk(hs, probe, units, layer):
    """Independent: the gate's recipe (running mean of the answer's hidden states, then the probe)."""
    H = hs[layer][0, P:P + T].double().numpy()
    return units(probe.logit(np.cumsum(H, 0) / np.arange(1, T + 1)[:, None]))


def test_identity_is_exact_without_steering(tiny):
    model, ids, probe, units, _ = tiny
    d, hs, _ = run(model, ids, probe, units)
    assert d["attn"].shape == d["mlp"].shape == (L, T)
    assert np.abs(identity_residual(d, L)).max() < 1e-9
    assert d["steer"] == {}


def test_risk_matches_gate_recipe_on_hidden_states(tiny):
    model, ids, probe, units, _ = tiny
    d, hs, _ = run(model, ids, probe, units)
    assert np.abs(d["risk"] - gate_risk(hs, probe, units, L)).max() < 1e-9


def test_base_is_independent_embedding_readout(tiny):
    model, ids, probe, units, _ = tiny
    d, hs, _ = run(model, ids, probe, units)
    w, b = probe.readout()
    E = hs[0][0, P:P + T].double().numpy()
    assert np.abs(d["base"] - units(running_mean(E @ w) + b)).max() < 1e-9


def test_dropping_any_layer_term_breaks_the_identity(tiny):
    """Non-vacuous: every layer carries a real share; nothing is silently in the base."""
    model, ids, probe, units, _ = tiny
    d, _, _ = run(model, ids, probe, units)
    for kind in ("attn", "mlp"):
        for i in range(L):
            e = dict(d, **{kind: np.delete(d[kind], i, axis=0)})
            res = identity_residual(e, L)
            assert np.allclose(res, -d[kind][i], atol=1e-9)
            assert np.abs(d[kind][i]).max() > 1e-4


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_steering_lands_in_steer_and_identity_holds(tiny, alpha):
    model, ids, probe, units, v = tiny
    dirs = {3: v[3], 4: v[4], 6: v[6]}  # two at/before the probe layer (hidden index <= 4), one after
    d, hs, _ = run(model, ids, probe, units, steer=dirs, alpha=alpha)
    assert sorted(d["steer"]) == [3, 4, 6]
    assert np.abs(identity_residual(d, L)).max() < 1e-9
    assert np.abs(d["risk"] - gate_risk(hs, probe, units, L)).max() < 1e-9  # post-steer stream
    assert all(np.abs(s).max() > 1e-4 for s in d["steer"].values())
    # The steering edit is exactly what changed the stream: the removed vector's readout.
    base, _, _ = run(model, ids, probe, units)
    assert np.abs(d["risk"] - base["risk"]).max() > 1e-4


def test_alpha_one_leaves_no_leak_signal_at_hooked_layer(tiny):
    model, ids, probe, units, v = tiny
    _, hs, _ = run(model, ids, probe, units, steer={4: v[4]}, alpha=1.0)
    proj = (hs[4][0].float() @ v[4]).abs().max().item()
    unsteered = (run(model, ids, probe, units)[1][4][0].float() @ v[4]).abs().max().item()
    assert proj < 1e-2 * unsteered


def test_control_wrong_hook_order_hides_steering_in_mlp(tiny):
    """Control: with the capture entered outside the steering context, the 'pre' hook runs after
    the steering hook, so steer reads ~0 and the edit is silently folded into mlp_contrib. The
    identity still holds, so this is exactly the bug the steer > 0 check above catches."""
    model, ids, probe, units, v = tiny
    right, _, _ = run(model, ids, probe, units, steer={4: v[4]}, alpha=2.0)
    wrong, _, _ = run(model, ids, probe, units, steer={4: v[4]}, alpha=2.0, capture_outside=True)
    assert np.abs(wrong["steer"][4]).max() < 1e-9 < np.abs(right["steer"][4]).max()
    assert np.allclose(wrong["mlp"][3], right["mlp"][3] - right["steer"][4], atol=1e-9)


def test_hooks_are_removed_and_change_nothing(tiny):
    model, ids, probe, units, v = tiny
    hooks = [(dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in model.modules()]
    with torch.no_grad():
        before = model(input_ids=ids).logits
    run(model, ids, probe, units, steer={4: v[4]}, alpha=2.0)
    with torch.no_grad():
        after = model(input_ids=ids).logits
        _, hs, _ = run(model, ids, probe, units)
        plain = model(input_ids=ids, output_hidden_states=True).hidden_states
    assert torch.equal(before, after)
    assert all(torch.equal(a, b) for a, b in zip(hs, plain))  # capture hooks don't alter the stream
    # Only transformers' own hooks remain (they were there before we started).
    assert [(dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in model.modules()] == hooks


def test_keep_raw_at_gives_last_prompt_vector(tiny):
    model, ids, probe, units, _ = tiny
    _, hs, rec = run(model, ids, probe, units)
    assert np.allclose(rec["raw"], hs[L][0, P - 1].double().numpy(), atol=0)
