"""Test cases: templates from attacks/seed_cases.yaml rendered for concrete employees.

A case fixes everything needed to replay it: the user turns, which records are in
the model's context, the system-prompt strictness and how to judge the answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from leakproof.data import ROOT, load_employees

SEED_CASES_PATH = ROOT / "attacks" / "seed_cases.yaml"
KINDS = {"attacks": "attack", "benign": "benign", "canaries": "canary"}


@dataclass
class Case:
    id: str
    kind: str  # attack | benign | canary
    strategy: str
    field: str  # salary | medical_leave | performance_note | all | none
    employee_id: str
    context_ids: list[str]
    turns: list[str]
    strictness: str = "default"
    oracle: dict = field(default_factory=lambda: {"type": "value"})
    expect_any: list[str] = field(default_factory=list)
    source: str = "seed"


def select_targets(employees: list[dict], n: int) -> list[dict]:
    """The first non-lead member of each team, for the first n teams (deterministic)."""
    targets, seen = [], set()
    for e in employees:
        if not e["is_lead"] and e["team"] not in seen:
            targets.append(e)
            seen.add(e["team"])
        if len(targets) == n:
            break
    return targets


def context_for(target: dict, by_id: dict[str, dict]) -> list[dict]:
    """Simulated retrieval: the target, one teammate and their manager (3 records)."""
    team = [e for e in by_id.values() if e["team"] == target["team"]]
    peer = next(e for e in team if not e["is_lead"] and e["id"] != target["id"])
    lead = next(e for e in team if e["is_lead"])
    return [target, peer, lead]


def template_vars(target: dict, peer: dict, lead: dict) -> dict:
    loc = target["location"].removeprefix("Remote (").rstrip(")")
    salary = target["secrets"]["salary"]
    return {
        "name": target["name"],
        "first": target["name"].split()[0],
        "team": target["team"],
        "role": target["role"],
        "manager": target["manager"],
        "manager_first": target["manager"].split()[0],
        "manager_last": target["manager"].split()[-1],
        "city": loc.split(",")[0],
        "peer": peer["name"],
        "peer_first": peer["name"].split()[0],
        "peer_last": peer["name"].split()[-1],
        "salary_below": f"${(salary // 10_000 - 1) * 10_000:,}",
    }


def render(template: dict, kind: str, target: dict, context: list[dict], strictness: str) -> Case:
    _, peer, lead = context
    v = template_vars(target, peer, lead)
    fld = template.get("field", "none")

    oracle: dict = {"type": template.get("oracle", "value"), "employee_id": target["id"], "field": fld}
    if oracle["type"] == "yes_no":
        oracle["affirm"] = template.get("affirm", [])
    elif oracle["type"] == "choice":
        hi, lo = sorted([target, peer], key=lambda e: e["secrets"]["salary"], reverse=True)
        oracle.update(answer=hi["name"], other=lo["name"])

    return Case(
        id=f"{template['id']}@{target['id']}",
        kind=kind,
        strategy=template["strategy"],
        field=fld,
        employee_id=target["id"],
        context_ids=[e["id"] for e in context],
        turns=[t.format(**v) for t in template["turns"]],
        strictness="lax" if kind == "canary" else strictness,
        oracle=oracle,
        expect_any=[x.format(**v) for x in template.get("expect_any", [])],
    )


def load_cases(kind: str | None = None, n_targets: int | None = None,
               strictness: str | None = None, path: Path = SEED_CASES_PATH) -> list[Case]:
    """All seed cases (or one kind), rendered for the first n_targets employees."""
    n_targets = n_targets or int(os.environ.get("LEAKPROOF_TARGETS", "3"))
    strictness = strictness or os.environ.get("LEAKPROOF_STRICTNESS", "default")
    employees = load_employees()
    by_id = {e["id"]: e for e in employees}
    spec = yaml.safe_load(path.read_text())

    cases = []
    for section, k in KINDS.items():
        if kind and k != kind:
            continue
        for target in select_targets(employees, n_targets):
            context = context_for(target, by_id)
            cases += [render(t, k, target, context, strictness) for t in spec.get(section, [])]
    return cases
