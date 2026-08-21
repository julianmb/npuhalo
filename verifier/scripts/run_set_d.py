#!/usr/bin/env python3
"""
run_set_d.py — Phase 2–4 runner for Set D agentic tasks.

Modes:
  --mode baseline   no verifier at all (calibration + metric baseline)
  --mode shadow     NPU verifier logs verdicts, never intervenes
  --mode active_lfm LFM-only active mode (ABORT/SUSPECT escalate)
  --mode active_esc LFM + Qwen3.5-2B logprob escalation

Saves raw JSON per-run into results/raw/setd/ and aggregates.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from agent_loop import (  # noqa: E402
    AgentLoop,
)
from verifier_client import NPUVerifierClient  # noqa: E402

GEN = "http://127.0.0.1:8012/v1"
NPU = "http://127.0.0.1:8001/v1"
ESCALATOR = "http://127.0.0.1:8013/v1"
DATA = os.path.join(REPO, "verifier", "data", "set_d_agentic.jsonl")
PROMPT = open(os.path.join(REPO, "verifier", "prompts",
                           "final_verdict_prompt.txt")).read()
OUT_DIR = os.path.join(REPO, "verifier", "results", "raw", "setd")


def load_tasks():
    with open(DATA) as f:
        return [json.loads(line) for line in f]


def run_single(task, mode, run_id, verifier=None, escalator=None,
               max_steps=15, no_verifier=False):
    """One task, one seed. Returns record dict."""
    os.makedirs(OUT_DIR, exist_ok=True)
    temperature = 0.3
    if mode == "baseline":
        loop = AgentLoopNoVerifier(GEN, max_steps=max_steps, temperature=temperature)
        return loop.run(task, run_id)
    loop = AgentLoop(GEN, verifier, PROMPT, mode=("shadow" if mode == "shadow" else "active"),
                     escalator=escalator, max_steps=max_steps,
                     base_temperature=temperature)
    return loop.run(task, run_id)


class AgentLoopNoVerifier(AgentLoop):
    """Baseline: identical prompting and tools, but never verifies/intervenes."""
    def __init__(self, gen_endpoint, max_steps=15, temperature=0.3):
        super().__init__(gen_endpoint, verifier=None, verifier_prompt="{task} {trajectory}",
                         mode="baseline", max_steps=max_steps, base_temperature=temperature)

    def _verify_step(self, step, steps, task):
        return {"verdict": "CONTINUE", "evidence": "no verifier (baseline)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["baseline", "shadow", "active_lfm", "active_esc"])
    ap.add_argument("--runs-per-task", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    verifier = None
    escalator = None
    if args.mode == "shadow":
        verifier = NPUVerifierClient(NPU)
    elif args.mode in ("active_lfm", "active_esc"):
        verifier = NPUVerifierClient(NPU)

    if args.mode == "active_esc":
        from escalator import QwenEscalator
        escalator = QwenEscalator(ESCALATOR)

    aggregate = {}
    per_run = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        for r in range(args.runs_per_task):
            run_id = f"{args.mode.split('_')[0]}{i}" if args.runs_per_task == 1 else f"{args.mode.split('_')[0]}{i}_s{r}"
            print(f"[{args.mode}] {task['id']} run {r+1}/{args.runs_per_task} ...", flush=True)
            rec = run_single(task, args.mode, run_id, verifier=verifier,
                             escalator=escalator)
            rec["run_id"] = run_id
            rec["config"] = args.mode
            per_run.append(rec)
            aggregate[task["id"]] = aggregate.get(task["id"], {"passed": 0, "total": 0,
                                                               "wall": 0.0, "steps": 0,
                                                               "verdicts": {"CONTINUE": 0, "SUSPECT": 0, "ABORT": 0},
                                                               "rollbacks": 0})
            aggregate[task["id"]]["total"] += 1
            aggregate[task["id"]]["passed"] += int(rec["passed"])
            aggregate[task["id"]]["wall"] += rec["wall_time"]
            aggregate[task["id"]]["steps"] += rec["num_steps"]
            for k, v in rec["verdicts"].items():
                aggregate[task["id"]]["verdicts"][k] += v
            aggregate[task["id"]]["rollbacks"] += rec["rollbacks"]
            # persist raw immediately
            with open(os.path.join(OUT_DIR, f"{task['id']}_{run_id}_{args.tag}.json"), "w") as f:
                json.dump(rec, f, indent=2)

    n = len(per_run)
    summary = {
        "mode": args.mode,
        "tasks": len(tasks),
        "runs": n,
        "wall_total_s": time.time() - t0,
        "passed": sum(a["passed"] for a in aggregate.values()),
        "aggregate": aggregate,
    }
    with open(os.path.join(OUT_DIR, f"summary_{args.mode}_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: {"passed": v["passed"], "total": v["total"]} for k, v in aggregate.items()}, indent=1))
    print(f"\nOverall: {summary['passed']}/{n} pass rate "
          f"{100*summary['passed']/n:.1f}% | wall {summary['wall_total_s']:.0f}s")


if __name__ == "__main__":
    main()
