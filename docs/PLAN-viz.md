# PLAN: inside-the-model telemetry for the demo (Clara's lane, viz session)

> The plan Claude follows for the inside-the-model data behind the demo. Do the milestones **in order**. A milestone is done only when its **Done when** check passes and is recorded in the Log. VISION.md is the design reference. docs/PLAN-internals.md is the other session's plan, which this one builds on and never edits. **Target format: Oleg's telemetry schema, `db/schema.sql`** (merged from `origin/main`). This plan fills it; it never changes it.

## Scope
**In:** for one call to the target model, produce a `requests` row and its `tokens` rows exactly as `db/schema.sql` defines them, for the columns the model backend owns:
- `requests`: `model, defense, defense_params, probe_version, probe_layer, threshold, messages, response, prompt_risk, peak_risk, risk_cross_pos, gate_fired_pos, n_prompt_tokens, n_tokens, latency_ms, error`
- `tokens`: `pos, token_id, text, char_start, char_end, risk, contrib_base, attn_contrib, mlp_contrib, steer`

plus a mock that emits the same rows on a laptop for Oleg.

Entry point: `viz.trace(backend, messages, defense) -> {"request": {...}, "tokens": [{...}, ...]}`, keys = schema column names, plus `viz.insert(conn, trace)` that writes them to a SQLite connection. `generate()` is untouched; we merge later.

**Out:** the judge columns (`label*`, `leaked_field`, `evidence_*`, `secret_start_pos`, `judged_at`, `judge_raw`: Anthony's judge worker), caller columns (`client`, `session_id`, `tags`), the DB file, the backend API and the board (Oleg). `lora` / `lora+steer` defenses (no adapter yet). The 2D activation map: **dropped** (overkill; the risk strip and the breakdown carry the claims). Separate per-layer probes: **dropped** (the schema's per-layer view is the decomposition below). The one extra probe kept is a last-prompt probe, because `prompt_risk` needs it (see Q3).

## How the numbers are defined
- **Probe = the gate's probe** (`GateProbe` at `probe_layer`, trained on train templates only). Its logit is linear in the hidden state: z(x) = ((x − mean)/scale)·w + b = x·w′ + b′, with w′ = w/scale and b′ = b − mean·w′.
- **Risk units:** risk = a·z + c, with c and a set so that the **median logit of train-template `none` answers → 0** and the **median of train-template `full` answers → 1** (answer means, cached M0 acts). Saved with the probe as `runs/viz/units.json`. `threshold` = a·logit(p_gate) + c, so "risk > threshold" fires on exactly the same tokens as the gate's probability > p_gate.
- **Which vector is scored (matches the gate):** token t's risk is the probe applied to the **running mean** of the answer's hidden states at `probe_layer` over tokens 0..t, i.e. exactly the gate's input (see Q1).
- **Decomposition:** the residual stream at hidden-state index L is h = e + Σ_{l<L}(attn_l + mlp_l). I capture the actual residual stream before and after each sublayer (a pre-hook on `post_attention_layernorm` gives the stream after attention; the decoder layer's input/output give the rest) and define attn_l = h_mid,l − h_in,l, mlp_l = h_out,l − h_mid,l. These telescope to h_L − e **exactly** (bf16 rounding inside each add is attributed to that sublayer, not lost). Running-mean them over tokens 0..t like h, then apply the linear probe:
  - `contrib_base` = a·(ē·w′) + a·b′ + c (token embedding + offset)
  - `attn_contrib[l]` = a·(ā_l·w′), `mlp_contrib[l]` = a·(m̄_l·w′), for decoder layers l = 0..L−1 (see Q4)
  - so `contrib_base + Σ attn + Σ mlp = risk` up to float32 error.
- **Steering:** the hook on steered layer l subtracts Δ = α(h·v̂_l)v̂_l from the stream after the MLP. I capture the stream before and after the hook, so the edit is never mixed into `mlp_contrib`. `steer[l]` = the removed amount in risk units, a·(Δ̄_l·w′) (running mean, the same probe readout), for every steered layer, including layers after `probe_layer` (where it doesn't change this token's risk). For steered layers ≤ `probe_layer` the identity becomes `base + Σ attn + Σ mlp − Σ steer = risk` (see Q2).
- **Request summary:** `peak_risk` = max token risk; `risk_cross_pos` = first pos with risk > threshold; `gate_fired_pos` = the gate's `fired_at` (k consecutive tokens over the threshold; NULL unless `defense = 'gate'` fired); `prompt_risk` = the last-prompt probe on the last prompt token, in its own risk units (Q3); `n_prompt_tokens`, `n_tokens` (answer tokens, stop token excluded); `latency_ms` = generation only (the extra telemetry forward pass is excluded and reported separately in the Log).
- **Token text and offsets:** `text` = decode of the single token; `char_start/char_end` from decoding prefixes (`len(decode(ids[:t]))`), so multi-byte characters split across tokens still get correct offsets into `response` (Q5).

## Guardrails
- **Files:** never edit `hf_target.py`, `collect.py`, `probe.py`, `direction.py`, `steer.py`, `gate.py`, `mock.py`, `docs/PLAN-internals.md`, `db/schema.sql`, or anything else that already exists on `probe-backend` / `main`. New files only: `leakproof/viz.py`, `tests/test_viz_*.py`, this plan, small outputs under `runs/viz/`. If an existing file needs a change, stop and ask Clara.
- **Where:** Mac worktree `~/Personal-Dev/AI-Hackathon-viz` (code + pure tests). Box worktree `~/clara/viz` (data + GPU). Read `~/clara/AI-Hackathon/runs/` (collect acts, `steer/vhat.npy`, `steer/split.json`, `gate/*/gate_probe.npz`, `steer/sweep/*/best.json`), never write there; the source folder is an argument / `LEAKPROOF_SRC_RUNS`. Outputs go to `~/clara/viz/runs/viz/`.
- **Memory (the PLAN-internals rules, counting the other session's jobs):** at most **one** GPU job across both sessions; before any GPU job check `tmux ls` (no `clara-*` job of the other session running), `free -g` and `nvidia-smi` (≥ 25 GB left after the ~16 GB model, Ollama counted). Prefer CPU work on cached acts; CPU jobs use `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8`. tmux sessions `clara-viz-*`; after killing one, confirm its python process is gone. `trace()` is batch 1 and keeps only the probe-layer projections, not the full activations (< 1 GB extra).
- **Honest evaluation:** everything fit from data (risk units, the last-prompt probe) uses **train templates only** (not held out in `runs/steer/split.json`). Reported numbers come from held-out templates.
- **Git:** branch `viz`, push to `origin/viz` only. Never push to `main` or `probe-backend`, no force pushes. Merging `origin/probe-backend` or `origin/main` *into* `viz` is allowed. Don't commit `runs/**/acts/`.
- **Simplicity:** numpy + stdlib `sqlite3` + the existing helpers, in the style of `gate.py`. No new dependencies.

## Dependency on the other session
`gate.py`, `steer.py` and `HFBackend.generate()` are uncommitted in the other session's checkout; `origin/probe-backend` (251fe72) doesn't have them. V1 and the pure parts of V2 need only numpy and the saved `gate_probe.npz` (read with `np.load`, same fields as `GateProbe`). **The GPU parts of V2 and V3 need `steering`, `watching` and `generate()`**: they start after the other session pushes M3/M4 and I merge `origin/probe-backend` into `viz`. If that hasn't happened by then, stop and ask.

## Proof of reliability (every milestone)
Same standard as PLAN-internals: tests in `tests/test_viz_*.py` (pure-logic ones run on the Mac with no GPU; model-backed ones `@pytest.mark.gpu`), a control that would fail if the method were fake, fixed seeds and reproducible outputs, evidence in the Log (command, pass/fail counts, numbers next to their control, commit hash), and the full `tests/test_viz_*.py` suite passing on the box before each push.

## Stop and ask Clara when
- a check still fails after two real fix attempts
- an existing file or the schema would have to change, or a question below is still open when its milestone starts
- the GPU milestones are blocked on the other session's push
- a result changes the plan (e.g. the decomposition doesn't sum to the risk)

## Milestones

### V0. Setup
- Worktrees on both machines on `viz`, `origin/main` merged (done on the Mac: fd43b13), plan committed and pushed to `origin/viz`.
- **Done when:** `~/clara/viz` is on `viz`, the existing pure tests pass there, and the plan is pushed.

### V1. Risk units + last-prompt probe (CPU, cached acts)
- `units.json`: a, c from the gate probe's logits on train-template answer means; the threshold in units.
- Last-prompt probe at `probe_layer` (same pipeline and C, train templates, `last_prompt` pooling) with its own units, for `prompt_risk` (if Q3 approves it).
- **Proofs:**
  - *Units:* the train `none` median maps to 0 and the `full` median to 1 (exact); `risk > threshold` ⇔ `p > p_gate` on every cached answer (monotone, same decisions).
  - *Held-out sanity:* held-out `none` / `full` medians land near 0 / 1 (reported, not asserted).
  - *Control (last-prompt probe):* the same pipeline on shuffled train labels gives held-out AUROC ≈ 0.5 (±0.1), clearly below the real probe (M1 found 0.89 at L19 under CV).
  - *No leakage:* no held-out template in any training row.
  - *Reproducible:* byte-identical outputs on rerun.
- **Done when:** proofs pass, files saved, numbers in the Log.

### V2. Decomposition (pure first, then GPU)
- `decompose(stream, probe, units, steer_edits)`: from captured `e`, per-layer (`h_in`, `h_mid`, `h_out_pre_steer`, `h_out`) at probe-layer granularity → per-token `risk`, `contrib_base`, `attn_contrib`, `mlp_contrib`, `steer`, all on the running mean.
- **Proofs:**
  - *Exact sum (pure):* on a toy residual stream (random sublayers, a linear probe, with and without a steering edit), `base + Σ attn + Σ mlp (− Σ steer at layers ≤ L) == risk` to 1e-9 in float64.
  - *Exact sum (GPU):* on real traces, |identity − risk| < 1e-4 risk units for every token, under `none` and `steer`.
  - *Non-vacuous:* `contrib_base` equals the probe readout of the embedding-layer output computed independently (`hidden_states[0]`), and leaving out any single layer's term breaks the identity by that term (so nothing is silently piled into the base).
  - *Matches the gate:* `risk` equals `generate()`'s `per_token_risk` mapped to units (max abs diff < 1e-4), and the live gate's per-token risks (`LiveGate.risks`) agree with the re-run forward pass (< 0.02 in probability, the M4 tolerance).
  - *Steering is captured:* with α = 1, the probe-layer stream after the hook has h·v̂ ≈ 0 on steered layers, and `steer` is 0 / NULL with steering off.
- **Done when:** proofs pass on the Mac (pure) and the box (GPU), numbers in the Log.

### V3. `trace()` → schema rows (GPU, one job)
- `trace(backend, messages, defense)` generates batch 1 with the same public pieces `generate()` uses (`encode`, `generate_ids`, `steering`, `watching`), then one telemetry forward pass with the capture hooks, and builds the `requests` + `tokens` dicts. `insert(conn, trace)` writes them; `probe_version` = `<gate_probe.npz run folder>@<sha1[:8] of the npz>+units`.
- **Proofs:**
  - *Schema conformance:* `insert()` into a fresh in-memory SQLite created from `db/schema.sql` succeeds (CHECK constraints and PK included) and every key is a real column; a test fails if the schema gains a NOT NULL column we don't fill.
  - *Consistency:* `n_tokens == len(tokens)`; `pos` is 0..n−1; `response[char_start:char_end] == text` for every token (when `response` is the emitted text, Q5); `peak_risk`, `risk_cross_pos`, `gate_fired_pos` recomputed from the token rows match; `len(attn_contrib) == len(mlp_contrib) ==` the agreed layer count (Q4).
  - *No side effects:* tokens under `none` equal `generate(defense="none")`'s; after a `steer` trace, a `none` trace is unchanged.
  - *Live = offline:* for ~10 cached cases, the last token's risk matches the gate probe on the cached `answer_mean` (< 0.02 in probability).
- **Done when:** proofs pass on the box; three example traces (a leaking attack, the same attack steered, the same attack gated) + a benign one saved as `runs/viz/examples/*.json` (small; committed so Oleg can build against real rows); numbers in the Log.

### V4. Mock for Oleg
- `MockTracer().trace(messages, defense)` emits the same `{request, tokens}` rows from `MockBackend`'s scripted answers: risk in units, a decomposition that sums exactly (base + growing middle-layer contributions on secret-looking tokens), `steer` filled only under `steer`, gate fields consistent with `MockBackend`'s gate. Numpy + stdlib only, runs on a Mac, deterministic.
- **Proofs:** one contract test runs against the mock (always) and the real tracer (GPU): `insert()` into the schema succeeds, same keys and types, the identity holds, `steer` non-NULL iff `defense == "steer"`, `gate_fired_pos` NULL unless `defense == "gate"`, unknown defense raises.
- **Done when:** contract test passes on both, usage in the `viz.py` docstring, pushed, and Clara has a short note for Oleg.

### V5. Write-up
- Log the results (where in the model the risk forms: the biggest attn/MLP contributors on leaking tokens; what steering removes per layer) and a short paragraph for Clara to put in VISION.md (I don't edit VISION.md on this branch).
- **Done when:** pushed, Clara has the paragraph and the merge notes (`viz` adds only new files, so it merges cleanly).

## Decisions (schema vs. how the probe and gate work)
> Approved by Clara on Oct 10: each item's "Plan:" is the decision. Oleg still to confirm 4 (schema comment), 5 (`response` on gate) and 7 (DB ownership).
1. **Running mean vs each token's own vector.** The schema reads as if a token's risk comes from that token's own hidden state. The gate (and the probe's training data) use the **running mean of the answer so far**. Plan: `risk` = running-mean risk, so the colors match where the gate fires, and the decomposition is still exact (it's linear, so it averages term by term). Consequence: clicking token t shows the breakdown of the answer *so far*, and risk changes smoothly rather than spiking on " 1", "5". Own-vector risk would spike more visibly but isn't what the probe was trained on and wouldn't match the gate. OK?
2. **`steer` and the sum.** With steering on layers ≤ `probe_layer`, the hook's edit is part of the stream, so `base + Σ attn + Σ mlp = risk` only holds if the edit is folded into `mlp_contrib` of that layer. Plan: keep it separate, store `steer` in risk units (the same probe readout of the removed vector), and document the identity as `base + Σ attn + Σ mlp − Σ steer(l ≤ probe_layer) = risk`. Alternative: `steer` in raw leak-signal units (α·h·v̂, the VISION definition) and fold the edit into `mlp_contrib`. Which does the board want?
3. **`prompt_risk`.** The gate probe was trained on answer means; applying it to the last prompt token is out of distribution (no answer yet). Plan: a separate last-prompt probe at `probe_layer` (M1 showed the model "knows" before writing: AUROC 0.89), in its own 0-to-1 units, and `threshold` doesn't apply to it. OK, or should `prompt_risk` be NULL?
4. **Layer count.** The comment says "one value per layer 0..probe_layer" (19 values for 18). With `hidden_states` numbering (also used by the gate and `steer_layers`), the probe at index 18 reads the output of decoder layers 0..17, so there are **18** attn and 18 mlp terms. Plan: 18 values, indexed by decoder layer, and `steer` keys use the same `hidden_states` numbering as `defense_params.steer_layers` (key "16" = hook after decoder layer 15). Agree on one convention and note it in the schema comment?
5. **`response` when the gate fires.** The schema says "cut short if the gate fired"; `generate()` returns the refusal text instead. If `response` is the refusal, the token rows' `char_start/char_end` don't point into it. Plan: `response` = the text actually emitted before the cut (the tokens rows match it), and the board shows the refusal when `gate_fired_pos` is set. Also, because the gate lags one token, the tokens table has `gate_fired_pos + 2` rows when it fires.
6. **Columns I can't fill from `trace(backend, messages, defense)`:** `client`, `session_id`, `tags` (caller's); `context` (the retrieved records are inside the system prompt, but which values are secret is known only to the caller / `target.py` retrieval; pass it in, or the caller fills it?); `secret_start_pos` (needs the judge's evidence offsets: I can provide `char_to_pos(tokens, char)` for the judge worker to use); all judge columns; `lora` / `lora+steer` (no adapter yet).
7. **Who writes to the DB file.** I'll provide `insert(conn, trace)` tested against `db/schema.sql`, but not open or own the DB file or the backend API. Confirm that's Oleg's integration.
8. **"Typical" safe / leak** for the units: plan = medians of train-template `none` / `full` attack answers. Benign answers would put 0 lower (they score lower than refused attacks), making attack refusals sit slightly above 0. Fine?

## Log
<!-- One line per milestone: date, result, numbers, commit. -->
- V0 (Oct 10, Mac): worktree `~/Personal-Dev/AI-Hackathon-viz` on `viz` from `origin/probe-backend` 251fe72; merged `origin/main` (adds `db/schema.sql`) as fd43b13, not pushed yet.
- V0 (Oct 10): plan approved by Clara and pushed (35ee2b7). Box worktree pending: no SSH access from the viz session.
- V1–V4 pure parts (Oct 10, Mac, commit 0425f88): `uv run --with torch --with transformers --with numpy --with scikit-learn --with pytest pytest tests/test_viz_*.py` → **41 pass, 12 skip** (GPU / box-data tests).
  - *Units (6):* the readout equals the gate's formula (1e-9); running mean commutes with the readout; medians → 0 / 1 exactly; the threshold in units gives identical decisions on 10k logits at 4 thresholds; inverted medians raise; save/load keeps the version.
  - *Fit (3, synthetic runs/ with a planted direction):* medians 0 / 1, same decisions, no held-out template in any training row, prompt probe held-out AUROC > 0.9 vs shuffled ≈ 0.5 (5 seeds); byte-identical rerun; altering only held-out activations leaves units.json unchanged.
  - *Decomposition (tiny random Qwen2, real decoder code, fp32 and bf16, 18 tests):* identity residual < 1e-9 with and without steering (α = 1, 2; steered layers before, at and after the probe layer); risk == the gate's recipe on `hidden_states[L]` (1e-9); base == independent embedding readout; dropping any layer's term breaks the identity by exactly that term and every term is non-trivial; α = 1 leaves < 1% of the leak signal at the hooked layer; hooks leave logits and hidden states bit-identical and are all removed.
  - *Control:* with the capture hooks entered outside the steering context (wrong order), `steer` reads 0 and the edit lands in `mlp_contrib` while the identity still holds: the ordering check (`steer` > 0) is what catches it.
  - *Contract (12):* mock traces for every defense × leak / benign insert into a fresh DB built from `db/schema.sql`; every key is a column and every NOT NULL column is filled; `"".join(text) == response` and every span matches; identity; `steer` iff steer; peak / cross / gate fields recomputed from the rows; JSON round-trips. Controls: a perturbed contribution fails the checker; an invalid `defense` is rejected by the schema's CHECK. Split UTF-8 characters get correct spans.
  - **Waiting on:** box access, and the other session pushing `gate.py`, `steer.py`, `generate()` (not on `origin/probe-backend` as of 0425f88) for `fit` on real data and `tests/test_viz_gpu.py`.
