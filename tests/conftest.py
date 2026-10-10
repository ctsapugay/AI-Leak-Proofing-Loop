"""Shared fixtures for the model-backed tests.

All selected cases run once, concurrently, against the backend (LM Studio by
default); each test then checks its own result. At the end a leak-rate table is
printed and every transcript is saved under runs/baseline/.

Environment:
    LEAKPROOF_MODEL       model id served by LM Studio (default qwen2.5-7b-instruct)
    LEAKPROOF_BASE_URL    default http://localhost:1234/v1
    LEAKPROOF_STRICTNESS  lax | default | strict system prompt (default "default")
    LEAKPROOF_TARGETS     how many employees to render cases for (default 3)
    LEAKPROOF_WORKERS     concurrent requests (default 4)
"""

import os

import pytest

from leakproof.runner import report, run_cases, run_dir, write_jsonl
from leakproof.target import LMStudioBackend

CASES_KEY = pytest.StashKey[list]()
RUN_KEY = pytest.StashKey[tuple]()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Remember which cases were selected (respects -k), so only those get run."""
    config.stash[CASES_KEY] = [
        item.callspec.params["case"]
        for item in items
        if hasattr(item, "callspec") and "case" in item.callspec.params
    ]


@pytest.fixture(scope="session")
def results(pytestconfig):
    backend = LMStudioBackend.from_env()
    problem = backend.check()
    if problem:
        pytest.skip(problem)

    cases = pytestconfig.stash[CASES_KEY]
    capture = pytestconfig.pluginmanager.getplugin("capturemanager")

    def progress(i, n, r):
        status = "ERROR" if r.error else r.verdict.label
        with capture.global_and_fixture_disabled():
            print(f"  [{i:>3}/{n}] {r.case.id:<44} {status:<8} {r.seconds:5.1f}s", flush=True)

    with capture.global_and_fixture_disabled():
        print(f"\nrunning {len(cases)} cases against {backend.name} ...", flush=True)
    res = run_cases(backend, cases, workers=int(os.environ.get("LEAKPROOF_WORKERS", "4")), on_done=progress)

    strictness = os.environ.get("LEAKPROOF_STRICTNESS", "default")
    out = run_dir(backend.name, strictness) / "answers.jsonl"
    write_jsonl(res, out, backend.name)
    (out.parent / "report.txt").write_text(report(res, backend.name) + "\n")
    pytestconfig.stash[RUN_KEY] = (backend.name, res, out)
    return res


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if RUN_KEY not in config.stash:
        return
    model, res, out = config.stash[RUN_KEY]
    terminalreporter.write_sep("=", "leak report")
    terminalreporter.write_line(report(res, model))
    terminalreporter.write_line(f"transcripts: {out}")
