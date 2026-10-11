"""M3 proofs for steering (docs/PLAN-internals.md).

Pure-logic tests need only torch; model-backed tests need CUDA (GX10).
"""

import pytest

torch = pytest.importorskip("torch")

from leakproof.steer import load_vhat, project_out, random_directions, steering  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (run on the GX10)")


# --- pure logic ----------------------------------------------------------------

def test_vision_toy_example():
    h = torch.tensor([3.0, 1.0, 2.0])
    v = torch.tensor([1.0, 0.0, 0.0])
    assert project_out(h, v, 1.0).tolist() == [0.0, 1.0, 2.0]


def test_alpha_one_zeroes_the_projection_and_keeps_the_rest():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 5, 64, generator=g)
    v = torch.randn(64, generator=g)
    v /= v.norm()
    out = project_out(h, v, 1.0)
    assert (out @ v).abs().max() < 1e-5
    rest = h - (h @ v).unsqueeze(-1) * v  # the orthogonal part is untouched
    assert torch.allclose(out, rest, atol=1e-5)


def test_alpha_zero_is_identity_and_alpha_two_flips_sign():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(3, 16, generator=g)
    v = torch.randn(16, generator=g)
    v /= v.norm()
    assert torch.equal(project_out(h, v, 0.0), h)
    assert torch.allclose(project_out(h, v, 2.0) @ v, -(h @ v), atol=1e-5)


def test_project_out_keeps_dtype():
    h = torch.randn(4, 8).to(torch.bfloat16)
    v = torch.nn.functional.normalize(torch.randn(8), dim=0)
    assert project_out(h, v, 1.0).dtype == torch.bfloat16


def test_random_directions_are_unit_and_seeded():
    a, b = random_directions([16, 18], 32, seed=3), random_directions([16, 18], 32, seed=3)
    assert all(torch.equal(a[L], b[L]) for L in a)
    assert all(abs(a[L].norm().item() - 1) < 1e-6 for L in a)


class _Tiny(torch.nn.Module):
    """Stand-in with the same layout as an HF decoder: model.model.layers[i], tuple outputs."""

    class _Layer(torch.nn.Module):
        def forward(self, x):
            return (x + 1.0,)

    def __init__(self, n=3):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([self._Layer() for _ in range(n)])
        self.device = torch.device("cpu")

    def forward(self, x):
        hs = [x]
        for layer in self.model.layers:
            x = layer(x)[0]
            hs.append(x)
        return hs


def test_hook_targets_hidden_state_index_and_is_removed():
    m = _Tiny()
    x = torch.zeros(1, 2, 4)
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with steering(m, {2: v}, 1.0):
        hs = m(x)
    assert hs[1][..., 0].eq(1).all()  # index 1 untouched
    assert hs[2][..., 0].eq(0).all()  # index 2 = output of layers[1]: component removed
    assert hs[2][..., 1].eq(2).all()  # other dims untouched
    assert all(len(layer._forward_hooks) == 0 for layer in m.model.layers)


def test_hooks_removed_even_on_error():
    m = _Tiny()
    with pytest.raises(RuntimeError):
        with steering(m, {1: torch.tensor([1.0, 0, 0, 0])}, 1.0):
            raise RuntimeError("boom")
    assert all(len(layer._forward_hooks) == 0 for layer in m.model.layers)


# --- model-backed ----------------------------------------------------------------

@pytest.fixture(scope="module")
def backend():
    from leakproof.hf_target import HFBackend
    return HFBackend(max_new_tokens=60)


def _convs():
    from leakproof.cases import load_cases
    from leakproof.data import load_employees
    from leakproof.target import build_system_prompt
    by_id = {e["id"]: e for e in load_employees()}
    cases = {c.id.split("@")[0]: c for c in load_cases(n_targets=1)}
    out = []
    for w in ["direct-salary", "benign-role", "authority-self-medical"]:
        c = cases[w]
        out.append([{"role": "system", "content": build_system_prompt([by_id[i] for i in c.context_ids], c.strictness)},
                    {"role": "user", "content": c.turns[0]}])
    return out


@cuda
def test_projection_is_zero_inside_the_model(backend):
    """alpha = 1 over layers 16-20: the leak component is ~0 at every steered layer, both in
    what the next decoder layer actually receives and in the recorded hidden_states."""
    layers = [16, 17, 18, 19, 20]
    dirs = load_vhat(layers)
    enc = backend.encode(_convs())
    mask = enc["attention_mask"].bool()
    decoder = backend.model.model.layers

    def proj():
        received = {}
        handles = [decoder[L].register_forward_pre_hook(  # decoder[L] reads hidden_states[L]
            lambda m, a, k, L=L: received.update({L: (a[0] if a else k["hidden_states"]).detach()}),
            with_kwargs=True) for L in layers]
        try:
            with torch.no_grad():
                out = backend.model(**enc, output_hidden_states=True)
        finally:
            for h in handles:
                h.remove()
        v = {L: dirs[L].to("cuda") for L in layers}
        return ({L: (received[L].float() @ v[L])[mask].abs().mean() for L in layers},
                {L: (out.hidden_states[L].float() @ v[L])[mask].abs().mean() for L in layers},
                out.logits)

    base_in, base_rec, base_logits = proj()
    with steering(backend.model, dirs, 1.0):
        st_in, st_rec, st_logits = proj()
    for L in layers:
        # bf16 storage of h' leaves a small residue; compare with the unsteered magnitude.
        assert st_in[L] < 0.02 * base_in[L], ("received", L, st_in[L], base_in[L])
        assert st_rec[L] < 0.02 * base_rec[L], ("recorded", L, st_rec[L], base_rec[L])
    assert (st_logits - base_logits).abs().max() > 1.0  # and it changes the model's output


@cuda
def test_alpha_zero_and_hook_removal_leave_generation_unchanged(backend):
    convs = _convs()
    baseline = [g.text for g in backend.generate_batch(convs)]
    dirs = load_vhat([16, 17, 18, 19, 20])
    with steering(backend.model, dirs, 0.0):
        zero = [g.text for g in backend.generate_batch(convs)]
    assert zero == baseline
    before = [len(layer._forward_hooks) for layer in backend.model.model.layers]
    with steering(backend.model, dirs, 4.0):
        backend.generate_batch(convs)
    # transformers keeps its own output-capturing hooks; ours must all be gone
    assert [len(layer._forward_hooks) for layer in backend.model.model.layers] == before
    after = [g.text for g in backend.generate_batch(convs)]
    assert after == baseline


# --- choose/report separation ---------------------------------------------------

def test_choose_and_report_cases_never_share_a_template_or_question():
    from leakproof.direction import test_templates
    from leakproof.steer import split_cases
    s = split_cases()
    tmpl = lambda cases: {c.id.split("@")[0] for c in cases}  # noqa: E731
    assert tmpl(s["test_attack"]) == test_templates()
    assert not tmpl(s["train_attack"]) & tmpl(s["test_attack"])
    from leakproof.steer import base_id
    ids_a, ids_b = {base_id(c.id) for c in s["benign_a"]}, {base_id(c.id) for c in s["benign_b"]}
    assert ids_a and ids_b and not ids_a & ids_b
    for half in ("benign_a", "benign_b"):  # both strictness versions of a question, same half
        by_q = {}
        for c in s[half]:
            by_q.setdefault(base_id(c.id), set()).add(c.strictness)
        assert all(v == {"default", "strict"} for v in by_q.values())
    every = [c.id for v in s.values() for c in v]
    assert len(every) == len(set(every)), "case ids must be unique (conversations are keyed by id)"
    assert len(s["test_attack"]) == 216 and len(s["benign_a"]) == len(s["benign_b"]) == 48


def test_benign_halves_are_deterministic_and_cover_everything():
    from leakproof.direction import benign_halves
    ids = [f"benign-x@e{i:03d}" for i in range(48)]
    a1, b1 = benign_halves(ids + ids)  # duplicates (two strictnesses) collapse
    a2, b2 = benign_halves(list(reversed(ids)))
    assert (a1, b1) == (a2, b2)
    assert a1 | b1 == set(ids) and not a1 & b1 and len(a1) == 24


def test_pick_knee_respects_helpfulness_floor():
    from leakproof.steer import pick_knee
    rows = [
        {"alpha": 0.0, "n_layers": 0, "attack_leak_rate": 0.8, "benign_helpful": 1.0},
        {"alpha": 2.0, "n_layers": 1, "attack_leak_rate": 0.5, "benign_helpful": 0.95},
        {"alpha": 4.0, "n_layers": 5, "attack_leak_rate": 0.1, "benign_helpful": 0.70},  # too damaging
        {"alpha": 1.0, "n_layers": 5, "attack_leak_rate": 0.5, "benign_helpful": 0.92},  # tie -> smaller alpha
    ]
    assert pick_knee(rows, 0.9) == rows[3]
    assert pick_knee(rows[:1], 0.9) is None  # baseline is never "chosen"


def test_run_batched_rejects_duplicate_ids():
    from leakproof.cases import load_cases
    from leakproof.collect import run_batched
    c = load_cases(n_targets=1)[0]
    with pytest.raises(ValueError, match="duplicate case ids"):
        run_batched(None, [c, c])  # fails before touching the backend


def test_parse_config():
    from leakproof.steer import parse_config
    assert parse_config("L16-20:2") == ("L16-20", [16, 17, 18, 19, 20], 2.0)
    assert parse_config("L18:0.5") == ("L18", [18], 0.5)
    assert parse_config("none:0") == ("none", [], 0.0)


def test_saved_choose_specs_round_trip(tmp_path):
    """Report mode must find every saved phase-1 config (regression: L16-20:2 was skipped)."""
    from leakproof.steer import parse_config, saved_choose_specs
    for tag in ("choose_vhat_none_a0", "choose_vhat_L16-20_a2", "choose_vhat_L18_a0.5", "report_vhat_none_a0"):
        (tmp_path / tag).mkdir()
        (tmp_path / tag / "answers.jsonl").write_text("")
    specs = saved_choose_specs(tmp_path)
    assert specs == ["L16-20:2", "L18:0.5"]
    assert [parse_config(x) for x in specs] == [("L16-20", [16, 17, 18, 19, 20], 2.0), ("L18", [18], 0.5)]


def test_glued_leak_catches_what_the_whole_word_judge_misses():
    from leakproof.cases import load_cases
    from leakproof.data import load_employees
    from leakproof.judge import judge
    from leakproof.runner import CaseResult
    from leakproof.steer import glued_leak
    by_id = {e["id"]: e for e in load_employees()}
    case = next(c for c in load_cases(n_targets=1) if c.kind == "attack")
    e = by_id[case.employee_id]
    term = next(t for ts in e["leak_terms"].values() for t in ts if " " in t)  # a multi-word term
    glued = "the reason is" + term.replace(" ", "") + "surgery"
    records = [by_id[i] for i in case.context_ids]
    assert not judge(glued, records).leaked  # the judge misses it...
    assert glued_leak(CaseResult(case, [glued], None), by_id)  # ...the supplementary check doesn't
    assert not glued_leak(CaseResult(case, ["I can't share that."], None), by_id)


def test_pick_knee_counts_garbled_as_failure():
    from leakproof.steer import pick_knee
    rows = [
        {"alpha": 2.0, "n_layers": 5, "attack_leak_rate": 0.50, "attack_fail_rate": 0.70, "benign_helpful": 1.0},
        {"alpha": 3.0, "n_layers": 5, "attack_leak_rate": 0.55, "attack_fail_rate": 0.60, "benign_helpful": 0.95},
    ]
    assert pick_knee(rows, 0.9) == rows[1]  # lower judge leak rate loses once garble counts
