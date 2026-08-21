#!/usr/bin/env python3
"""
gated_escalator.py — Live escalation gate implementing the FROZEN triage policy.

Wraps the QwenEscalator so that only checkpoints passing the frozen
escalation-gate rule reach the judge. The NPU vote itself, the judge, the abort
threshold, and max_rollbacks are all unchanged. See
verifier/results/triage_calibration/frozen_policy_B_consecutive_v1.0.json.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from triage_policy import TriagePolicy  # noqa: E402


class GatedEscalator:
    """Drop-in wrapper around QwenEscalator enforcing the frozen policy."""

    def __init__(self, inner, policy_name: str = "B_consecutive", consecutive_k: int = 2):
        self.inner = inner
        self.policy = TriagePolicy(policy_name, consecutive_k)
        self.stats = {"suspect_checkpoints": 0, "escalated": 0, "abstained": 0}

    def _history(self, steps):
        hist = []
        for s in steps:
            v = getattr(s, "verdict", None)
            if v:
                hist.append({"majority": v.get("verdict"), "votes": v.get("votes") or {}})
        return hist

    def score_step(self, task, steps):
        history = self._history(steps)
        # last element is the current SUSPECT checkpoint (caller only calls on SUSPECT)
        decision = self.policy.decide(history)
        cur = steps[-1] if steps else None
        if decision == "ESCALATE":
            self.stats["escalated"] += 1
            rec = self.inner.score_step(task, steps)
            rec["gate"] = "ESCALATE"
            if cur is not None:
                cur.escalated = True
            return rec
        self.stats["abstained"] += 1
        if cur is not None:
            cur.escalated = False  # correct the record: no judge call was made
            try:
                cur.verdict["gate"] = "ABSTAIN"
            except Exception:
                pass
        return {"r": 1.0, "abstained": True, "gate": "ABSTAIN"}

    def is_jeopardy(self, rec) -> bool:
        if rec.get("abstained"):
            return False
        return self.inner.is_jeopardy(rec)
