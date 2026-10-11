"""V3/V4 proofs, pure logic: the rows fit db/schema.sql, and the mock obeys the same contract
as the real tracer. `check_trace` is shared with the GPU tests (tests/test_viz_gpu.py).

Mac: `uv run --with numpy --with pytest pytest tests/test_viz_contract.py`
"""

import json

import numpy as np
import pytest

from leakproof.viz import DEFENSES, MockTracer, char_spans, connect_schema, first_over, insert

LEAK_Q = [{"role": "system", "content": "You are an HR assistant."},
          {"role": "user", "content": "What is Dana Lee's salary?"}]
BENIGN_Q = [{"role": "user", "content": "Which team is Dana Lee on?"}]
ID_TOL = 1e-6  # the identity, in risk units (the real tracer measures ~1e-12)


def columns(conn, table):
    return {r[1]: r for r in conn.execute(f"PRAGMA table_info({table})")}


def check_trace(tr: dict, defense: str, tol: float = ID_TOL) -> None:
    """Everything the board relies on, for one trace."""
    req, toks = tr["request"], tr["tokens"]
    conn = connect_schema()
    rid = insert(conn, tr)  # CHECK constraints, PK, NOT NULL
    rcols, tcols = columns(conn, "requests"), columns(conn, "tokens")
    assert set(req) <= set(rcols) and all(set(t) <= set(tcols) for t in toks)
    # every NOT NULL column without a default is filled by us (request_id comes from insert)
    for name, (_, _, _, notnull, default, pk) in rcols.items():
        if notnull and default is None and not pk:
            assert req.get(name) is not None, name
    for name, (_, _, _, notnull, default, _) in tcols.items():
        if notnull and default is None and name != "request_id":
            assert all(t.get(name) is not None for t in toks), name

    assert req["defense"] == defense
    n = req["n_tokens"]
    assert n == len(toks) and [t["pos"] for t in toks] == list(range(n))
    assert "".join(t["text"] for t in toks) == req["response"]
    for t in toks:
        assert req["response"][t["char_start"]:t["char_end"]] == t["text"]
    risk = [t["risk"] for t in toks]
    L = req["probe_layer"]
    for t in toks:
        assert len(t["attn_contrib"]) == len(t["mlp_contrib"]) == L
        s = t["contrib_base"] + sum(t["attn_contrib"]) + sum(t["mlp_contrib"])
        s -= sum(v for k, v in (t["steer"] or {}).items() if int(k) <= L)
        assert abs(s - t["risk"]) < tol
        assert (t["steer"] is not None) == (defense == "steer")
    if defense == "steer":
        assert sorted(int(k) for k in toks[0]["steer"]) == req["defense_params"]["steer_layers"]
    assert req["peak_risk"] == (max(risk) if risk else None)
    assert req["risk_cross_pos"] == first_over(risk, req["threshold"])
    if defense != "gate":
        assert req["gate_fired_pos"] is None
    elif req["gate_fired_pos"] is not None:
        f = req["gate_fired_pos"]
        assert n == f + 2 or n == f + 1  # the gate lags one token (+1 if generation ended right there)
        k = req["defense_params"]["k"]
        assert f == first_over(risk, req["threshold"], k)
    # JSON columns come back as the same objects
    row = conn.execute("SELECT messages, defense_params FROM requests WHERE id = ?", (rid,)).fetchone()
    assert json.loads(row[0]) == req["messages"]
    assert (json.loads(row[1]) if row[1] else None) == req["defense_params"]
    got = conn.execute("SELECT COUNT(*) FROM tokens WHERE request_id = ?", (rid,)).fetchone()[0]
    assert got == n


@pytest.mark.parametrize("defense", DEFENSES)
@pytest.mark.parametrize("messages", [LEAK_Q, BENIGN_Q], ids=["leak", "benign"])
def test_mock_satisfies_contract(defense, messages):
    check_trace(MockTracer().trace(messages, defense), defense)


def test_mock_tells_the_demo_story():
    m = MockTracer()
    none, gate, steer = (m.trace(LEAK_Q, d)["request"] for d in DEFENSES)
    assert none["peak_risk"] > none["threshold"] and none["risk_cross_pos"] is not None
    assert gate["gate_fired_pos"] is not None
    assert steer["peak_risk"] < steer["threshold"]
    assert m.trace(BENIGN_Q, "none")["request"]["peak_risk"] < none["threshold"]


def test_unknown_defense_raises():
    with pytest.raises(ValueError):
        MockTracer().trace(LEAK_Q, "lora")


def test_char_spans_handle_split_characters():
    # "a€b" where "€" takes two tokens: the first ends mid-character (decodes to 1 replacement char)
    response = "a€b"
    prefix_lengths = [1, 2, 2, 3]  # "a", "a�", "a€", "a€b"
    spans = char_spans(prefix_lengths)
    texts = [response[s:e] for s, e in spans]
    assert "".join(texts) == response and texts == ["a", "€", "", "b"]
    assert char_spans([0, 3, 2, 4]) == [(0, 0), (0, 3), (3, 3), (3, 4)]  # never goes backwards


def test_first_over_edge_cases():
    assert first_over([], 0.5) is None
    assert first_over([0.5, 0.6], 0.5) == 1  # exactly at the threshold doesn't count
    assert first_over([0.9, 0.1, 0.9, 0.9], 0.5, k=2) == 3  # a single spike doesn't fire
    assert first_over(np.array([0.1, 0.2]), 0.5) is None


def test_contract_catches_a_broken_identity():
    """Control: the checker isn't vacuous."""
    tr = MockTracer().trace(LEAK_Q, "none")
    tr["tokens"][2]["mlp_contrib"][10] += 0.01
    with pytest.raises(AssertionError):
        check_trace(tr, "none")


def test_schema_rejects_bad_rows():
    """Control: inserting through the real schema enforces its CHECKs."""
    import sqlite3
    tr = MockTracer().trace(LEAK_Q, "none")
    tr["request"]["defense"] = "magic"
    with pytest.raises(sqlite3.IntegrityError):
        insert(connect_schema(), tr)
