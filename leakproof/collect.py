"""Run cases through the HF target in batches, judge them and cache activations.

    .venv/bin/python -m leakproof.collect [--targets 12] [--strictness default] [--batch 32]

Writes runs/collect/<time>__<model>__<strictness>/:
  answers.jsonl, report.txt      same format as runs/baseline (rejudge works on it)
  acts/last_prompt.npy           [n_cases, n_layers + 1, hidden] fp16
  acts/answer_mean.npy           same shape
  acts/index.jsonl               row i of the .npy files = case i: id, kind, strategy, field, label

Activations are for each case's final reply (the last turn of multi-turn cases).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from leakproof.cases import Case, load_cases
from leakproof.data import load_employees
from leakproof.hf_target import HFBackend
from leakproof.runner import RUNS_DIR, CaseResult, judge_case, report, write_jsonl
from leakproof.target import build_system_prompt


def run_batched(backend: HFBackend, cases: list[Case]) -> tuple[dict[str, CaseResult], dict[str, dict]]:
    """Turn t of every case that has one goes in the same batch; replies are threaded back in."""
    by_id = {e["id"]: e for e in load_employees()}
    convs = {c.id: [{"role": "system", "content": build_system_prompt(
        [by_id[i] for i in c.context_ids], c.strictness)}] for c in cases}
    replies: dict[str, list[str]] = {c.id: [] for c in cases}
    acts: dict[str, dict] = {}

    start = time.perf_counter()
    for t in range(max(len(c.turns) for c in cases)):
        active = [c for c in cases if len(c.turns) > t]
        for c in active:
            convs[c.id].append({"role": "user", "content": c.turns[t]})
        gens = backend.generate_batch([convs[c.id] for c in active])
        for c, g in zip(active, gens):
            convs[c.id].append({"role": "assistant", "content": g.text})
            replies[c.id].append(g.text)
            acts[c.id] = g.acts  # overwritten each turn, so the final turn's remain
        print(f"  turn {t + 1}: {len(active)} cases, {time.perf_counter() - start:.0f}s elapsed", flush=True)

    per_case = (time.perf_counter() - start) / len(cases)
    results = {c.id: CaseResult(c, replies[c.id], judge_case(c, replies[c.id], by_id), None, per_case)
               for c in cases}
    return results, acts


def write_acts(results: dict[str, CaseResult], acts: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ids = list(results)
    for key in ("last_prompt", "answer_mean"):
        np.save(out / f"{key}.npy", np.stack([acts[i][key] for i in ids]))
    with (out / "index.jsonl").open("w") as f:
        for row, i in enumerate(ids):
            c, v = results[i].case, results[i].verdict
            f.write(json.dumps({"row": row, "id": i, "kind": c.kind, "strategy": c.strategy,
                                "field": c.field, "label": v.label}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=12, help="employees to render cases for")
    ap.add_argument("--strictness", default="default", choices=["lax", "default", "strict"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    cases = load_cases(n_targets=args.targets, strictness=args.strictness)
    print(f"{len(cases)} cases, loading {args.model} ...", flush=True)
    backend = HFBackend(args.model, batch_size=args.batch)

    results, acts = run_batched(backend, cases)
    run = RUNS_DIR / "collect" / f"{time.strftime('%Y%m%d-%H%M%S')}__{backend.name}__{args.strictness}"
    write_jsonl(results, run / "answers.jsonl", backend.name)
    (run / "report.txt").write_text(report(results, backend.name) + "\n")
    write_acts(results, acts, run / "acts")
    print(report(results, backend.name))
    print(f"\nsaved to {run}")


if __name__ == "__main__":
    main()
