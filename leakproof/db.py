"""The telemetry database (db/schema.sql): one SQLite file, WAL mode.

    conn = connect()            # $LEAKPROOF_DB, default db/leakproof.db; creates the tables

Several processes write it (model backend, judge worker), so every connection waits for
the lock instead of failing.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from leakproof.data import ROOT

SCHEMA_PATH = ROOT / "db" / "schema.sql"
DEFAULT_DB = ROOT / "db" / "leakproof.db"


def db_path() -> Path:
    return Path(os.environ.get("LEAKPROOF_DB", DEFAULT_DB))


def connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or db_path(), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


def context_snapshot(records: list[dict]) -> list[dict]:
    """What goes in requests.context: everything the judge needs, frozen at call time."""
    keys = ("id", "name", "team", "role", "manager", "location", "secrets", "leak_terms")
    return [{k: r[k] for k in keys if k in r} for r in records]


def loads(value: str | None):
    return json.loads(value) if value else None
