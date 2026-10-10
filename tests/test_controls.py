"""Controls that must stay green, or the red attack results can't be trusted.

- benign: ordinary questions get a helpful answer and nothing is flagged
  (catches false alarms now, and defenses that break helpfulness later)
- canary: the prompt allows sharing, so the judge must detect the secret
  (if this fails, the harness or judge is broken, not the model)
"""

import pytest

from leakproof.cases import load_cases
from leakproof.runner import canary_detected

BENIGN = load_cases("benign")
CANARIES = load_cases("canary")


@pytest.mark.parametrize("case", BENIGN, ids=lambda c: c.id)
def test_benign_is_answered_without_leaking(case, results):
    r = results[case.id]
    if r.error:
        pytest.fail(r.explain())
    assert not r.verdict.leaked, r.explain()
    answer = " ".join(r.replies).lower()
    assert any(e.lower() in answer for e in case.expect_any), (
        f"unhelpful: expected one of {case.expect_any} in: {r.replies[-1][:300]}"
    )


@pytest.mark.parametrize("case", CANARIES, ids=lambda c: c.id)
def test_canary_leak_is_detected(case, results):
    r = results[case.id]
    if r.error:
        pytest.fail(r.explain())
    assert canary_detected(r), f"judge missed an allowed disclosure: {r.replies[-1][:300]}"
