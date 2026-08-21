#!/usr/bin/env python3
"""
Hard-rule deterministic baseline for pairwise evaluation.
Rejects explicit failed tests, patch conflicts, tracebacks, and syntax errors.
"""

import json
import re

FAILURE_KEYWORDS = [
    r"fail",
    r"error",
    r"exception",
    r"traceback",
    r"conflict",
    r"zerodivision",
    r"keyerror",
    r"typeerror",
    r"oom",
    r"assertionerror",
    r"merge conflict",
    r"could not find",
    r"infinite loop",
    r"leak",
    r"vulnerability",
    r"injection",
    r"unhandled",
    r"todo"
]

SUCCESS_KEYWORDS = [
    r"passed",
    r"clean",
    r"success",
    r"100%",
    r"verified",
    r"all tests",
    r"ok",
    r"200 ok"
]

def score_text_hardrule(text):
    text_lower = text.lower()
    fail_score = sum(1 for kw in FAILURE_KEYWORDS if re.search(r"\b" + kw, text_lower))
    success_score = sum(1 for kw in SUCCESS_KEYWORDS if re.search(r"\b" + kw, text_lower))
    return success_score - 2 * fail_score

def evaluate_hardrules(data_path="standalone-eval/data/pairs.jsonl"):
    with open(data_path) as f:
        pairs = [json.loads(line) for line in f]

    correct = 0
    ties = 0
    total = len(pairs)

    for p in pairs:
        sa = score_text_hardrule(p["candidate_a"])
        sb = score_text_hardrule(p["candidate_b"])
        
        if sa > sb:
            winner = "A"
        elif sb > sa:
            winner = "B"
        else:
            winner = "TIE"

        if winner == p["preferred"]:
            correct += 1
        elif winner == "TIE":
            ties += 1

    accuracy = correct / total
    tie_rate = ties / total
    print("=== Hard-Rule Baseline Results ===")
    print(f"Total pairs: {total}")
    print(f"Correct: {correct}/{total} ({accuracy * 100:.1f}%)")
    print(f"Ties: {ties}/{total} ({tie_rate * 100:.1f}%)")
    return {"accuracy": accuracy, "tie_rate": tie_rate, "correct": correct, "total": total}

if __name__ == "__main__":
    evaluate_hardrules()
