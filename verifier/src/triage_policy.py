#!/usr/bin/env python3
"""
triage_policy.py — Escalation-gate policies for the Tier-2 NPU verifier.

Shared by the offline replay harness and the live paired runner so that the
frozen policy is byte-identical in both paths. See
verifier/results/triage_calibration_protocol_2026-08-21.md for the
pre-registered definitions.
"""

MAJORITY_SUSPECT = "SUSPECT"


class TriagePolicy:
    """Decides ESCALATE / ABSTAIN for a SUSPECT checkpoint given vote history."""

    def __init__(self, name: str = "A_reference", consecutive_k: int = 2):
        self.name = name
        self.consecutive_k = consecutive_k

    def decide(self, checkpoint_history: list[dict]) -> str:
        """
        checkpoint_history: list of {"majority": "CONTINUE"|"SUSPECT"|"ABORT",
                                     "votes": {...}} oldest-first, including the
        current checkpoint as the last element.
        Returns "ESCALATE" or "ABSTAIN".
        """
        cur = checkpoint_history[-1]
        if cur["majority"] != MAJORITY_SUSPECT:
            return "ABSTAIN"  # not SUSPECT: no escalation needed (CONTINUE/ABORT handled upstream)
        if self.name == "A_reference":
            return "ESCALATE"
        if self.name == "B_consecutive":
            k = self.consecutive_k
            if len(checkpoint_history) >= k and all(
                    h["majority"] == MAJORITY_SUSPECT for h in checkpoint_history[-k:]):
                return "ESCALATE"
            return "ABSTAIN"
        if self.name == "C_unanimous":
            votes = cur.get("votes") or {}
            if votes and set(votes) == {MAJORITY_SUSPECT}:
                return "ESCALATE"
            return "ABSTAIN"
        raise ValueError(f"unknown policy {self.name}")


POLICY_NAMES = ["A_reference", "B_consecutive", "C_unanimous"]
