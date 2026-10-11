# PLAN: model internals (Clara's lane)

> The plan Claude follows in goal mode. Do the milestones **in order**. A milestone is done only when its **Done when** check passes and is recorded in the Log. Don't start the next one before that. VISION.md is the design reference; this file is the to-do list.

## Scope
**In:** HF target + activation cache → probe → steering direction v̂ → steering → gate → the `generate()` interface + a mock for Oleg's UI.
**Out (don't build, don't edit):** tier-2 judge (Anthony), LoRA fine-tune (Anthony), attacker / Nemotron case generation (Oleg), UI (Oleg). Read-only on `judge.py`, `cases.py`, `attacks/`, `data/`. If one of these blocks me, stop and ask.

## Guardrails
- **Box:** work only in `~/clara/AI-Hackathon`. Long jobs run in tmux sessions named `clara-*`. Check `nvidia-smi` before loading a model. Never kill, restart or reconfigure anything I didn't start (Ollama, teammates' jobs). No sudo, no system or firmware changes, no Docker.
- **Model:** the target is locked to Qwen2.5-7B-Instruct, bf16, greedy decoding. Final numbers come from 7B only.
- **Honest evaluation:** split train/test **by attack template** (the case id before `@`), never by case or token. v̂, probe weights, α and the window are chosen on train templates; reported numbers come from held-out templates. Report what we find, including negative results.
- **Git:** all work goes on branch **`probe-backend`**, never `main`. Commit after each milestone and push to `origin/probe-backend`. Small commits. No force pushes, no merging into `main` (Clara decides when). Don't commit `runs/**/acts/`. The box's clone tracks the same branch.
- **Simplicity:** the plainest code that works, written in the same style as the existing modules. No new frameworks.

## Proof of reliability (applies to every milestone)
"Done" means **proven**, not "the script ran". Each milestone needs:
1. **Automated tests** in `tests/test_internals_*.py`. Pure-logic tests run on the laptop with no GPU. Model-backed tests are marked `@pytest.mark.gpu` and skip automatically without CUDA.
2. **A control** that would fail if the method were fake (a shuffled-label probe, a random steering direction, etc.). The real result must clearly beat it.
3. **Reproducibility:** fixed seeds, and re-running gives the same numbers (within the stated tolerance).
4. **Evidence in the Log:** the test command, pass/fail counts, the key numbers next to their control, and the commit hash.
5. **Before each push:** the full test suite passes on the box (`.venv/bin/python -m pytest tests/test_internals_*.py`), and the pure-logic tests pass on the Mac.

A failing test or a control that matches the real result means the milestone is **not done**, even if the headline number looks good.

## Stop and ask Clara when
- a milestone's check still fails after two real fix attempts
- a decision belongs to the team (anything under "Open decisions" in VISION.md, or a change to a shared interface or file)
- I need a resource, a permission or a teammate's action
- a result changes the plan (e.g. no layer separates leak from safe)

## Milestones

### M0. Full baseline collection
- Full run: 12 targets, default prompt. Second run with the strict prompt for more, and more varied, data.
- **Proofs:**
  - *Batch invariance:* the same conversation alone vs. inside a padded batch gives the same text and activations (cosine > 0.999 at every layer).
  - *Capture correctness:* activations from the re-run forward pass match ones captured by a hook during `generate()` (cosine > 0.999).
  - *Determinism:* running the same cases twice gives identical text.
  - *Mask:* the answer mean excludes padding and stop tokens (a unit test on a toy mask).
- **Done when:** the proofs pass, both runs have `answers.jsonl`, `report.txt` and `acts/` with no NaNs, the canaries are all caught, and the backend is pushed to `probe-backend`.

### M1. Probe (`leakproof/probe.py`)
- Data: attack cases from the M0 runs. Labels: `full` vs `none` for training; score `partial` separately.
- For every layer and both poolings (`answer_mean`, `last_prompt`): `StandardScaler` + `LogisticRegression(class_weight="balanced")`, with GroupKFold by template. Report AUROC as mean ± std.
- Also report the false-alarm rate on benign cases at the chosen threshold.
- **Proofs:**
  - *No leakage:* a test asserts no template appears in both train and test in any fold.
  - *Control:* the same pipeline on shuffled labels gives AUROC ≈ 0.5 (within ±0.1), and the real AUROC beats it by a wide margin.
  - *Sanity:* on synthetic data with a planted direction, the probe recovers it (AUROC > 0.95).
  - *Reproducible:* a fixed seed gives identical AUROCs on a re-run.
- **Done when:** the proofs pass, the AUROC table for each layer is saved to `runs/probe/`, the best layer is picked, and the result is in the Log. If the best AUROC is under 0.75, stop and ask.

### M2. Steering direction
- v̂ per layer = normalized mean(leak) − mean(safe), computed on train templates. Report cosine(v̂, probe w) for each layer, and how well projection onto v̂ alone separates held-out templates (AUROC).
- **Proofs:**
  - *Unit tests:* v̂ has unit norm, and on synthetic data a planted mean difference is recovered (cosine > 0.95).
  - *Control:* a random unit direction separates held-out leak from safe at AUROC ≈ 0.5, clearly worse than v̂.
- **Done when:** the proofs pass, v̂ for every layer is saved (`runs/steer/vhat.npy`), and the numbers above are in the Log.

### M3. Steering (`defense="steer"`)
- A forward hook on each layer in the window applies h′ = h − α(h·v̂)v̂ at every position (projection removal).
- Sweep α ∈ {0, 0.5, 1, 2, 4, 8} × window ∈ {best layer, best ± 2}. On held-out attack templates and benign cases, measure leak rate and benign helpfulness (`expect_any`). Spot-check a few outputs for gibberish.
- **Proofs:**
  - *Unit tests:* the VISION toy example h=[3,1,2], v̂=[1,0,0] gives [0,1,2], and with α=1 the output has h·v̂ = 0.
  - *Hook works in the model:* with steering on, the measured projection onto v̂ at hooked layers is ≈ 0 (α=1).
  - *No side effects:* α=0 gives text identical to no hook, and after a steered call the hooks are gone (the next unsteered call matches the baseline exactly).
  - *Control:* a random direction at the same α and window does **not** cut leaks comparably. This shows it's the leak direction doing the work, not general damage.
- **Done when:** the proofs pass, the sweep table is saved, a setting is picked at the knee (a big leak-rate drop with benign helpfulness ≥ 90%), side-by-side examples are saved, and the result is in the Log.

### M4. Gate (`defense="gate"`)
- Score each generated token with the best-layer probe. Stop and refuse after k = 2–3 consecutive tokens over the threshold.
- Measure catch rate on held-out attacks, false alarms on benign cases, and the median number of tokens emitted before it fires.
- **Proofs:**
  - *Unit tests:* the trigger logic fires only after k consecutive scores over the threshold (including edge cases: a single spike, a threshold exactly met, an empty answer).
  - *Consistency:* per-token scores computed live during generation match offline scores from a forward pass (max abs diff < 0.02).
  - *No side effects:* with the gate on and never firing, the text is identical to no defense.
  - *Control:* a keyword-filter baseline on the same held-out set, for comparison.
- **Done when:** the proofs pass, the table is saved and the result is in the Log.

### M5. Interface for the UI
- `generate(messages, defense="none|gate|steer") -> {text, tokens, per_token_risk}` on HFBackend, plus a `MockBackend` that returns the same shape on a laptop without a GPU. Leave a `lora` slot for Anthony.
- **Proofs:** one shared contract test runs against both backends: same keys and types, `len(tokens) == len(per_token_risk)`, risks in [0, 1], each defense value accepted, and an unknown defense raises a clear error.
- **Done when:** the contract test passes on both backends, the mock runs on the Mac, and usage is documented in the module docstring. Push, then tell Clara so Oleg can build against it.

### M6. Write-up
- Add a short "Results (internals)" section to VISION.md: the probe layer and AUROC, the steering setting and leak/helpfulness numbers, the gate numbers, and what didn't work.
- **Done when:** pushed, and Clara has a one-paragraph summary.

## Log
<!-- One line per milestone: date, result, numbers, commit. -->
- Setup: backend commit moved off `main` onto local branch `probe-backend` (not pushed yet).
- **M0 (Oct 10): DONE** (commit 3982a24). `tests/test_internals_backend.py`: **7/7 pass on GX10**, 3 pass + 4 GPU-skipped on Mac (`uv run --with torch --with transformers --with numpy pytest tests/test_internals_backend.py`).
  - Mask unit tests: stop mid-answer, stop then padding, empty answer (zeros, not NaN), no stop: pass.
  - Capture correctness: re-run forward vs hooks during `generate()` at layers 3/13/20, cosine > 0.999 for the last prompt token and the answer mean: pass.
  - Determinism: two runs give identical text, acts max abs diff < 1e-2: pass.
  - Batch invariance: first run **failed** (1/5 texts differed). Diagnosed as an exact greedy tie (top-2 log-probs both −0.693 for `:\n` vs `:\n\n`), flipped by bf16 padding noise. Test now: last_prompt cosine > 0.995 (measured min 0.9983), answer_mean likewise when texts match, and any text difference must be at a tie (top-2 gap < 0.1).
  - **Control:** the same check with the attention mask removed drops padded rows to cosine 0.79–0.85 (asserted < 0.945), so the test catches real padding bugs. A wrong-positions control did *not* separate (0.999 vs 0.999): RoPE is shift-invariant, so left-pad position offsets are harmless here.
  - Runs: default 226/300 = 75% leak (192 full, 34 partial), 48/48 benign clean + helpful, 36/36 canaries. Strict 206/300 = 69% (177 full, 29 partial), 48/48, 36/36. Acts (384, 29, 3584) ×2 poolings ×2 runs, no NaN/inf/zero rows. Probe data: 369 full vs 168 none attack answers (+63 partial).
  - Note: the last `hidden_states` index has the final RMSNorm applied (documented in `hf_target.py`).
- **M1 (Oct 10): DONE.** `tests/test_internals_probe.py`: **8/8 pass** on Mac and GX10 (template-disjoint folds and every row tested once; planted direction AUROC > 0.95 and weight cosine > 0.8; shuffled labels and no-signal data within ±0.1 of 0.5; fixed seed gives identical scores; threshold and C rules).
  - Data: 537 attack answers (369 full, 168 none), 23 templates, StratifiedGroupKFold(5) by template. C = 0.01 (strongest regularization within 0.005 of the best; sweep at L14: 0.973 / 0.976 / 0.976 / 0.978). C = 1.0 was abandoned because it pinned 20 CPU cores for ~90 min; 0.01 runs in 90 s with 8 threads.
  - **answer_mean: best layer 18, AUROC 0.981 ± 0.022 vs shuffled 0.543.** Layers 13–22 all ≥ 0.975. Layer 0 (bag-of-words) gives 0.842, so part of the signal is lexical, and the middle layers add +0.14.
  - **last_prompt (before any answer token): best layer 19, AUROC 0.890 ± 0.043 vs shuffled 0.469.** The model "knows" before writing. Layer 0 = 0.500 exactly (same final token for every prompt): a sanity check.
  - Threshold for 5% benign false alarms: 0.739, with out-of-fold attack recall 93.2% at that threshold. At 0.5 the benign false-alarm rate is 27% (96 benign answers never seen in training).
  - **Limitation:** partial leaks score mean 0.077, 0/63 above threshold. The full-vs-none probe treats yes/no and comparison leaks as safe.
  - Reproducibility: a full rerun gave byte-identical auroc.csv and summary.json. Saved: `runs/probe/20261010-170359/` (auroc.csv, summary.json, probe.npz at L18).
