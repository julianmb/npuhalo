#!/usr/bin/env python3
"""
run_shadow.py — Phase 3: SHADOW MODE for Set D.

Runs all 30 tasks with the NPU verifier logging verdicts but NEVER
intervening. For every FAILED trajectory an autopsy record is written:
  - each step, its tool output, and the validator's verdict at that boundary
  - the first "evidence moment": the earliest step whose tool output contains
    objective failure evidence (non-zero exit, traceback, test failure,
    missing file)
  - the verifier's vote distribution at that evidence step

Raw runs:   verifier/results/raw/setd/     (per-task JSON)
Autopsy:    verifier/results/shadow/autopsy_<task>_<run>.json
Summary:    verifier/results/shadow/shadow_summary.json
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))
sys.path.insert(0, HERE)

from verifier_client import NPUVerifierClient  # noqa: E402

GEN = "http://127.0.0.1:8012/v1"
NPU = "http://127.0.0.1:8001/v1"
DATA = os.path.join(REPO, "verifier", "data", "set_d_agentic.jsonl")
PROMPT_FILE = os.path.join(REPO, "verifier", "prompts", "final_verdict_prompt.txt")
RAW_DIR = os.path.join(REPO, "verifier", "results", "raw", "setd")
OUT_DIR = os.path.join(REPO, "verifier", "results", "shadow")

EVIDENCE_PATTERNS = [
    (r"Traceback \(most recent call last\)", "python_traceback"),
    (r"(AssertionError|SyntaxError|NameError|TypeError|ValueError|ImportError|KeyError|IndexError|AttributeError|ModuleNotFoundError|FileNotFoundError|NotADirectoryError|PermissionError)", "python_exception"),
    (r"FAILED|AssertionError", "test_failure_marker"),
    (r"No such file or directory", "missing_file"),
    (r"command not found", "command_not_found"),
    (r"ModuleNotFoundError", "missing_module"),
    (r"Connection (refused|reset)|TimeoutError", "connection_failure"),
]


def evidence_kind(tool_out: str, exit_code):
    kinds = []
    for pat, kind in EVIDENCE_PATTERNS:
        if re.search(pat, tool_out, re.IGNORECASE):
            kinds.append(kind)
    if exit_code not in (0, None) and not kinds:
        kinds.append("nonzero_exit")
    return kinds


def load_tasks():
    with open(DATA) as f:
        return [json.loads(line) for line in f]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    verifier = NPUVerifierClient(NPU)
    # pre-warm NPU (documented 74s cold start)
    print("[*] warming NPU verifier ...", flush=True)
    verifier.evaluate_checkpoint("warm", "warm", "{task} {trajectory}", k_samples=3, temperature=0.7)

    from agent_loop import AgentLoop

    tasks = load_tasks()
    if limit:
        tasks = tasks[:limit]

    prompt = open(PROMPT_FILE).read()
    summary = {"runs": [], "totals": {"tasks": 0, "passed": 0,
                                      "failed": 0, "evidence_steps": 0,
                                      "detected_at_evidence": 0}}
    loop = AgentLoop(GEN, verifier, prompt, mode="shadow",
                     max_steps=15, base_temperature=0.3)

    for idx, task in enumerate(tasks):
        print(f"[shadow] {task['id']} ...", flush=True)
        rec = loop.run(task, f"shadow{idx}")
        rec["config"] = "shadow"
        with open(os.path.join(RAW_DIR, f"{task['id']}_shadow{idx}.json"), "w") as f:
            json.dump(rec, f, indent=2)

        summary["totals"]["tasks"] += 1
        if rec["passed"]:
            summary["totals"]["passed"] += 1
            summary["runs"].append({"task": task["id"], "passed": True})
            continue
        summary["totals"]["failed"] += 1

        autopsy = {
            "task": task["id"],
            "category": task["category"],
            "instruction": task["instruction"],
            "passed": False,
            "num_steps": rec["num_steps"],
            "wall_time": rec["wall_time"],
            "steps": [],
        }
        # detect the earliest evidence-backed moment from tool outputs
        first_evidence = None
        for s in rec["steps"]:
            tr = s.get("tool_result") or {}
            combined = (tr.get("stdout") or "") + "\n" + (tr.get("stderr") or "")
            kinds = evidence_kind(combined, tr.get("exit_code"))
            entry = {
                "step": s["step"],
                "tool": (s.get("tool_call") or {}).get("name"),
                "exit_code": tr.get("exit_code"),
                "evidence_kinds": kinds,
                "verifier_votes": (s.get("verdict") or {}).get("votes"),
                "verifier_verdict": (s.get("verdict") or {}).get("verdict"),
                "output_head": combined[:400],
            }
            autopsy["steps"].append(entry)
            if kinds and first_evidence is None:
                first_evidence = entry
        autopsy["first_evidence_step"] = first_evidence
        if first_evidence is not None:
            summary["totals"]["evidence_steps"] += 1
            v = (first_evidence.get("verifier_verdict") or "")
            if v in ("SUSPECT", "ABORT"):
                summary["totals"]["detected_at_evidence"] += 1
            autopsy["caught_at_evidence"] = v in ("SUSPECT", "ABORT")
        else:
            autopsy["caught_at_evidence"] = None

        with open(os.path.join(OUT_DIR, f"autopsy_{task['id']}.json"), "w") as f:
            json.dump(autopsy, f, indent=2)
        summary["runs"].append({"task": task["id"], "passed": False,
                                "evidence_step": (first_evidence or {}).get("step"),
                                "caught_at_evidence": autopsy["caught_at_evidence"]})

    t = summary["totals"]
    if t["failed"]:
        t["failures_with_evidence"] = t["evidence_steps"]
        t["real_recall_on_catchable"] = (t["detected_at_evidence"] / t["evidence_steps"]
                                         if t["evidence_steps"] else None)
    with open(os.path.join(OUT_DIR, "shadow_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(t, indent=2))


if __name__ == "__main__":
    main()
