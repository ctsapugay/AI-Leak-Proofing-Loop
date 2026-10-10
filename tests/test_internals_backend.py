"""M0 proofs for the HF backend (docs/PLAN-internals.md).

Pure-logic tests need only torch (Mac: `uv run --with torch --with transformers --with numpy pytest ...`).
Model-backed tests need CUDA and skip elsewhere (run them on the GX10).
"""

import pytest

torch = pytest.importorskip("torch")

from leakproof.cases import load_cases  # noqa: E402
from leakproof.data import load_employees  # noqa: E402
from leakproof.hf_target import answer_mask, masked_mean  # noqa: E402
from leakproof.target import build_system_prompt  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (run on the GX10)")
COS_MIN = 0.999  # same shapes (re-run vs generate, run vs re-run)
# Alone vs padded batch: bf16 matmuls see different shapes. Measured min 0.9983 over all
# layers; with the attention mask removed, padded rows drop to 0.79-0.96.
BATCH_COS_MIN = 0.995
HOOK_LAYERS = (3, 13, 20)  # decoder layer k  <->  hidden_states index k + 1


# --- pure logic ----------------------------------------------------------------

def test_answer_mask_stops_at_first_stop_token():
    gen = torch.tensor([
        [5, 6, 2, 7, 2],  # stop mid-answer; tokens after it don't count
        [5, 2, 2, 2, 2],  # stop then padding
        [2, 9, 9, 9, 9],  # empty answer
        [5, 6, 7, 8, 9],  # hit max_new_tokens, no stop
    ])
    T, F = True, False
    assert answer_mask(gen, {2}).tolist() == [
        [T, T, F, F, F],
        [T, F, F, F, F],
        [F, F, F, F, F],
        [T, T, T, T, T],
    ]


def test_answer_mask_any_stop_id_counts():
    gen = torch.tensor([[5, 3, 6, 2], [5, 6, 2, 3]])
    assert answer_mask(gen, {2, 3}).tolist() == [[True, False, False, False], [True, True, False, False]]


def test_masked_mean_ignores_masked_positions():
    hs = torch.zeros(2, 1, 4, 2)
    hs[0, 0, :, 0] = torch.tensor([1.0, 3.0, 100.0, 100.0])
    hs[1, 0, :, 0] = torch.tensor([7.0, 7.0, 7.0, 7.0])
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    out = masked_mean(hs, mask)
    assert out[0, 0].tolist() == [2.0, 0.0]
    assert out[1, 0].tolist() == [0.0, 0.0]  # empty answer -> zeros, not NaN


# --- model-backed ----------------------------------------------------------------

def _cos(a, b):
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


@pytest.fixture(scope="module")
def backend():
    from leakproof.hf_target import HFBackend
    return HFBackend(max_new_tokens=60)


@pytest.fixture(scope="module")
def conversations():
    """Real single-turn cases with different lengths, so batching adds padding."""
    by_id = {e["id"]: e for e in load_employees()}
    want = ["direct-salary", "benign-role", "dump-debug", "fiction-file-story", "aggregate-salary"]
    cases = {c.id.split("@")[0]: c for c in load_cases(n_targets=1)}
    convs = []
    for w in want:
        c = cases[w]
        convs.append([
            {"role": "system", "content": build_system_prompt([by_id[i] for i in c.context_ids], c.strictness)},
            {"role": "user", "content": c.turns[0]},
        ])
    return convs


def _divergence_is_a_tie(backend, conv, text_a, text_b, max_gap=0.1):
    """True if two greedy answers first differ at a token where the top-2 log-probs are
    within max_gap (bf16 noise from padding can flip an exact tie; anything else is a bug)."""
    ta = backend.tok(text_a, add_special_tokens=False)["input_ids"]
    tb = backend.tok(text_b, add_special_tokens=False)["input_ids"]
    k = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]), min(len(ta), len(tb)))
    enc = backend.encode([conv])
    ids = torch.cat([enc["input_ids"], torch.tensor([ta[:k]], device="cuda", dtype=torch.long)], dim=1)
    with torch.no_grad():
        top = torch.topk(torch.log_softmax(backend.model(ids).logits[0, -1].float(), -1), 2).values
    return (top[0] - top[1]).item() < max_gap


@cuda
def test_batch_invariance(backend, conversations):
    """Alone vs inside a padded batch: same activations (within bf16 noise) and same text,
    except where greedy decoding meets an exact tie."""
    batched = backend.generate_batch(conversations)
    for conv, b in zip(conversations, batched):
        alone = backend.generate_batch([conv])[0]
        cos = _cos(alone.acts["last_prompt"], b.acts["last_prompt"])
        assert cos.min() > BATCH_COS_MIN, f"last_prompt: min cosine {cos.min():.5f} at layer {cos.argmin()}"
        if alone.text == b.text:
            cos = _cos(alone.acts["answer_mean"], b.acts["answer_mean"])
            assert cos.min() > BATCH_COS_MIN, f"answer_mean: min cosine {cos.min():.5f} at layer {cos.argmin()}"
        else:
            assert _divergence_is_a_tie(backend, conv, alone.text, b.text), (
                f"batched text differs without a tie:\n{alone.text!r}\nvs\n{b.text!r}")


@cuda
def test_batch_check_catches_a_broken_attention_mask(backend, conversations):
    """Control: if padding were attended to, padded rows would fall far below the threshold,
    so test_batch_invariance would catch it."""
    enc = backend.encode(conversations)
    with torch.no_grad():
        broken = torch.stack(backend.model(input_ids=enc["input_ids"], output_hidden_states=True).hidden_states, 1)
    n_pad = (enc["attention_mask"] == 0).sum(dim=1)
    assert (n_pad >= 10).any(), "need at least one heavily padded row for the control"
    for i, conv in enumerate(conversations):
        if n_pad[i] < 10:
            continue
        with torch.no_grad():
            alone = torch.stack(backend.model(**backend.encode([conv]), output_hidden_states=True).hidden_states, 1)
        assert _cos(alone[0, :, -1], broken[i, :, -1]).min() < BATCH_COS_MIN - 0.05


@cuda
def test_rerun_matches_activations_seen_during_generate(backend, conversations):
    """The re-run forward pass reproduces what the model computed while generating."""
    enc = backend.encode(conversations)
    P = enc["input_ids"].shape[1]
    captured = {k: [] for k in HOOK_LAYERS}

    def hook(k):
        def fn(_module, _inputs, output):
            captured[k].append((output[0] if isinstance(output, tuple) else output).detach())
        return fn

    layers = backend.model.model.layers
    handles = [layers[k].register_forward_hook(hook(k)) for k in HOOK_LAYERS]
    try:
        seqs, mask = backend.generate_ids(enc)
    finally:
        for h in handles:
            h.remove()
    hs = backend.hidden_states(seqs, enc["attention_mask"], mask)

    G = mask.shape[1]
    fed = mask[:, :G - 1]  # the last generated token is never fed back during generate
    assert fed.any(dim=1).all(), "every test answer should have at least 2 tokens"
    for k in HOOK_LAYERS:
        cap = torch.cat(captured[k], dim=1)  # [batch, P + G - 1, hidden]
        assert cap.shape[1] == P + G - 1
        rerun = hs[:, k + 1]
        assert _cos(cap[:, P - 1], rerun[:, P - 1]).min() > COS_MIN, f"last prompt token, layer {k}"
        cap_mean = masked_mean(cap[:, None, P:], fed)[:, 0]
        rerun_mean = masked_mean(rerun[:, None, P:P + G - 1], fed)[:, 0]
        assert _cos(cap_mean, rerun_mean).min() > COS_MIN, f"answer mean, layer {k}"


@cuda
def test_determinism(backend, conversations):
    a = backend.generate_batch(conversations)
    b = backend.generate_batch(conversations)
    assert [g.text for g in a] == [g.text for g in b]
    for ga, gb in zip(a, b):
        for key in ga.acts:
            assert abs(ga.acts[key].astype("float32") - gb.acts[key].astype("float32")).max() < 1e-2
