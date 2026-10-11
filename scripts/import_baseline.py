"""Load saved baseline transcripts into the telemetry database, one row per model call.

    uv run python scripts/import_baseline.py                 # every runs/baseline/*/answers.jsonl
    uv run python scripts/import_baseline.py runs/baseline/<run>/answers.jsonl

A multi-turn case becomes one row per turn, sharing session_id; each row's messages hold
the history up to that call (system prompt rebuilt with build_system_prompt). These runs
came from LM Studio (4-bit), so the model is tagged "@lmstudio-q4" to keep them apart from
the bf16 GX10 numbers. No tokens or risk: only the transcripts and, after judging, labels.
Re-running skips rows already imported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leakproof.data import load_employees  # noqa: E402
from leakproof.db import connect, context_snapshot  # noqa: E402
from leakproof.target import build_system_prompt  # noqa: E402


def import_file(conn, path: Path, by_id: dict[str, dict]) -> int:
    run_id = path.parent.name
    added = 0
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("error") or not r.get("replies"):
            continue
        records = [by_id[i] for i in r["context_ids"]]
        messages = [{"role": "system", "content": build_system_prompt(records, r["strictness"])}]
        session_id = f"{run_id}/{r['id']}"
        for turn, (user, reply) in enumerate(zip(r["turns"], r["replies"])):
            messages.append({"role": "user", "content": user})
            source_key = f"{session_id}#{turn}"
            exists = conn.execute("SELECT 1 FROM requests WHERE json_extract(tags, '$.source_key') = ?",
                                  (source_key,)).fetchone()
            if not exists:
                tags = {
                    "source_key": source_key, "run_id": run_id, "case_id": r["id"], "kind": r["kind"],
                    "strategy": r["strategy"], "field": r["field"], "employee_id": r["employee_id"],
                    "strictness": r["strictness"], "oracle": r["oracle"], "source": r.get("source", "seed"),
                    "turn": turn, "n_turns": len(r["turns"]),
                }
                conn.execute(
                    "INSERT INTO requests (client, session_id, model, defense, messages, context, response, tags,"
                    " latency_ms) VALUES ('import', ?, ?, 'none', ?, ?, ?, ?, ?)",
                    (session_id, f"{r['model']}@lmstudio-q4", json.dumps(messages),
                     json.dumps(context_snapshot(records)), reply, json.dumps(tags),
                     round(r["seconds"] * 1000) if len(r["turns"]) == 1 else None),
                )
                added += 1
            messages.append({"role": "assistant", "content": reply})
    conn.commit()
    return added


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "runs" / "baseline").glob("*/answers.jsonl"))
    by_id = {e["id"]: e for e in load_employees()}
    conn = connect()
    for p in paths:
        print(f"{p.parent.name}: {import_file(conn, p, by_id)} rows added")
    print(f"requests total: {conn.execute('SELECT count(*) FROM requests').fetchone()[0]}")
