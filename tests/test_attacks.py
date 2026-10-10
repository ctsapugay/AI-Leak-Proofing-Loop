"""Attack cases. Each asserts the target did NOT leak, so a leak is a red (failed) test.

At baseline (no defenses) red is the expected result: it proves the attacks work.
The hardening loop's job is to turn these green.
"""

import pytest

from leakproof.cases import load_cases

ATTACKS = load_cases("attack")


@pytest.mark.parametrize("case", ATTACKS, ids=lambda c: c.id)
def test_attack_does_not_leak(case, results):
    r = results[case.id]
    if r.error:
        pytest.fail(r.explain())
    assert not r.verdict.leaked, r.explain()
