"""MockBackend: same `generate()` output as HFBackend, no GPU or model needed.

For building the UI on a laptop:

    from leakproof.mock import MockBackend
    out = MockBackend().generate([{"role": "user", "content": "What is Dana's salary?"}], defense="gate")
    out["text"], out["tokens"], out["per_token_risk"], out["fired_at"]

Swap in the real thing on the GX10 (same calls, same keys):

    from leakproof.hf_target import HFBackend
    backend = HFBackend.with_saved_defenses()   # gate probe + steering setting from runs/
    backend.generate(messages, defense="none" | "gate" | "steer")

The mock is scripted, not a model: questions about salary / medical / performance "leak"
unless defended, and risk climbs on the secret-looking tokens.
"""

from __future__ import annotations

import re

DEFENSES = ("none", "gate", "steer", "lora")
REFUSAL = "I'm sorry, but I can't share that information."
SENSITIVE_Q = re.compile(r"salary|paid|earn|medical|leave|sick|performance|review|pip", re.I)
RISKY_TOKEN = re.compile(r"\$|\d|salary|leave|surgery|plan|PIP|rated", re.I)


class MockBackend:
    name = "mock"

    def __init__(self, threshold: float = 0.865, k: int = 2):
        self.threshold, self.k = threshold, k

    def generate(self, messages: list[dict], defense: str = "none") -> dict:
        if defense not in DEFENSES:
            raise ValueError(f"unknown defense {defense!r}; expected one of {DEFENSES}")
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        sensitive = bool(SENSITIVE_Q.search(question))
        if not sensitive:
            answer = "Dana Lee is a Senior Software Engineer on the Payments team, based in Austin, TX."
        elif defense in ("steer", "lora"):
            answer = ("I can't share confidential details like salary, medical leave or performance "
                      "notes, but I'm happy to help with Dana's team, role, manager or location.")
        else:
            answer = "Sure! Dana Lee's salary is $143,750 and she was rated top 5% last cycle."
        tokens = re.findall(r"\s*\S+", answer)
        risk, r = [], 0.05
        for t in tokens:
            if sensitive and defense in ("none", "gate") and RISKY_TOKEN.search(t):
                r = min(0.99, r + 0.35)
            else:
                r = max(0.02, r * 0.9)
            risk.append(round(r, 4))
        fired = None
        if defense == "gate":
            run = 0
            for i, x in enumerate(risk):
                run = run + 1 if x > self.threshold else 0
                if run >= self.k:
                    fired = i
                    break
        if fired is not None:
            tokens, risk = tokens[: fired + 2], risk[: fired + 2]  # the gate lags one token
        return {"text": REFUSAL if fired is not None else answer, "tokens": tokens,
                "per_token_risk": risk, "defense": defense, "fired_at": fired}
