"""Export the requests table to JSONL, so judged data can be shared through git.

    uv run python scripts/export_db.py                       # -> runs/judged/requests.jsonl
    uv run python scripts/export_db.py --out path.jsonl --judged-only

One line per row, every column; JSON columns (messages, context, tags, judge_raw, ...)
are written as objects, not strings. The SQLite file itself stays out of git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leakproof.db import connect  # noqa: E402

JSON_COLUMNS = {"defense_params", "messages", "context", "tags", "judge_raw"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "judged" / "requests.jsonl")
    ap.add_argument("--judged-only", action="store_true")
    args = ap.parse_args()

    conn = connect()
    where = " WHERE judged_at IS NOT NULL" if args.judged_only else ""
    rows = conn.execute(f"SELECT * FROM requests{where} ORDER BY id").fetchall()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            d = {k: (json.loads(r[k]) if k in JSON_COLUMNS and r[k] else r[k]) for k in r.keys()}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {args.out.relative_to(ROOT)}")
