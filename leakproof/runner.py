"""Replay cases against a backend, judge every answer and save the transcripts.

Re-score saved transcripts with the current judge (no model calls):

    uv run python -m leakproof.runner runs/baseline/<run>/answers.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from leakproof.cases import Case
from leakproof.data import ROOT, load_employees
from leakproof.judge import Verdict, judge
from leakproof.target import Backend, build_system_prompt

RUNS_DIR = ROOT / "runs"
FIELDS = ("salary", "medical_leave", "performance_note", "all")


@dataclass
class CaseResult:
    case: Case
    replies: list[str]
    verdict: Verdict | None
    error: str | None = None
    seconds: float = 0.0

    def explain(self) -> str:
        if self.error:
            return f"ERROR {self.error}"
        last = self.replies[-1].strip().replace("\n", " ") if self.replies else ""
        return (f"LEAK({self.verdict.label}) [{self.case.strategy}] {self.verdict.summary()}\n"
                f"  asked: {self.case.turns[-1]}\n  reply: {last[:300]}")


def canary_detected(r: CaseResult) -> bool:
    """A canary passes when the judge flags the requested secret as a full leak."""
    return bool(r.verdict) and any(h.level == "full" and h.field == r.case.field for h in r.verdict.hits)


def run_case(backend: Backend, case: Case, by_id: dict[str, dict]) -> CaseResult:
    records = [by_id[i] for i in case.context_ids]
    messages = [{"role": "system", "content": build_system_prompt(records, case.strictness)}]
    replies: list[str] = []
    start = time.perf_counter()
    try:
        for turn in case.turns:
            messages.append({"role": "user", "content": turn})
            reply = backend.chat(messages)
            messages.append({"role": "assistant", "content": reply})
            replies.append(reply)
    except Exception as exc:
        return CaseResult(case, replies, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
    return CaseResult(case, replies, judge_case(case, replies, by_id), None, time.perf_counter() - start)


def judge_case(case: Case, replies: list[str], by_id: dict[str, dict]) -> Verdict:
    """Any turn leaking counts; numbers the user typed don't count as near misses."""
    records = [by_id[i] for i in case.context_ids]
    return judge("\n\n".join(replies), records, prompt="\n".join(case.turns), oracle=case.oracle)


def run_cases(backend: Backend, cases: list[Case], workers: int = 4,
              on_done: Callable[[int, int, CaseResult], None] | None = None) -> dict[str, CaseResult]:
    by_id = {e["id"]: e for e in load_employees()}
    results: dict[str, CaseResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, backend, c, by_id) for c in cases]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results[r.case.id] = r
            if on_done:
                on_done(i, len(cases), r)
    return {c.id: results[c.id] for c in cases}  # keep case order


def run_dir(model: str, strictness: str) -> Path:
    """A fresh folder per run, so a quick `-k` run never overwrites a full one."""
    slug = re.sub(r"[^\w.-]+", "_", model)
    return RUNS_DIR / "baseline" / f"{time.strftime('%Y%m%d-%H%M%S')}__{slug}__{strictness}"


def write_jsonl(results: dict[str, CaseResult], path: Path, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in results.values():
            row = {
                "model": model,
                **asdict(r.case),
                "replies": r.replies,
                "label": r.verdict.label if r.verdict else None,
                "hits": [asdict(h) for h in r.verdict.hits] if r.verdict else [],
                "error": r.error,
                "seconds": round(r.seconds, 2),
            }
            f.write(json.dumps(row) + "\n")


def report(results: dict[str, CaseResult], model: str) -> str:
    """Leak-rate table for attacks (strategy x field) plus control status."""
    attacks = [r for r in results.values() if r.case.kind == "attack" and r.verdict]
    lines = [f"model: {model}"]
    if attacks:
        cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        rows: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in attacks:
            leaked = int(r.verdict.leaked)
            for key, bucket in (((r.case.strategy, r.case.field), cells), (r.case.strategy, rows)):
                bucket[key][0] += leaked
                bucket[key][1] += 1
        strictness = attacks[0].case.strictness
        lines.append(f"system prompt strictness: {strictness}")
        lines.append("")
        header = f"{'strategy':<14}" + "".join(f"{f:>18}" for f in FIELDS) + f"{'total':>10}"
        lines += [header, "-" * len(header)]
        for strategy in dict.fromkeys(r.case.strategy for r in attacks):
            row = f"{strategy:<14}"
            for f in FIELDS:
                leaked, total = cells.get((strategy, f), (0, 0))
                row += f"{(f'{leaked}/{total}' if total else '-'):>18}"
            leaked, total = rows[strategy]
            lines.append(row + f"{f'{leaked}/{total}':>10}")
        n_leak = sum(r.verdict.leaked for r in attacks)
        n_full = sum(r.verdict.label == "full" for r in attacks)
        lines += ["", f"attack leak rate: {n_leak}/{len(attacks)} = {n_leak / len(attacks):.0%}"
                      f"  (full {n_full}, partial {n_leak - n_full})"]

    benign = [r for r in results.values() if r.case.kind == "benign" and r.verdict]
    if benign:
        clean = sum(not r.verdict.leaked for r in benign)
        helpful = sum(any(e.lower() in " ".join(r.replies).lower() for e in r.case.expect_any) for r in benign)
        lines.append(f"benign controls: {clean}/{len(benign)} no leak, {helpful}/{len(benign)} helpful")
    canaries = [r for r in results.values() if r.case.kind == "canary" and r.verdict]
    if canaries:
        caught = sum(canary_detected(r) for r in canaries)
        lines.append(f"canaries: {caught}/{len(canaries)} leaks detected")
    errors = sum(r.error is not None for r in results.values())
    if errors:
        lines.append(f"errors: {errors}")
    return "\n".join(lines)


def rejudge(path: Path) -> tuple[str, dict[str, CaseResult]]:
    """Re-score a saved answers.jsonl with the current judge and rewrite it in place."""
    by_id = {e["id"]: e for e in load_employees()}
    rows = [json.loads(line) for line in path.open()]
    results = {}
    for row in rows:
        case = Case(**{k: row[k] for k in Case.__dataclass_fields__})
        verdict = None if row["error"] else judge_case(case, row["replies"], by_id)
        results[case.id] = CaseResult(case, row["replies"], verdict, row["error"], row["seconds"])
    model = rows[0]["model"] if rows else "?"
    write_jsonl(results, path, model)
    (path.parent / "report.txt").write_text(report(results, model) + "\n")
    return model, results


if __name__ == "__main__":
    model, res = rejudge(Path(sys.argv[1]))
    print(report(res, model))
