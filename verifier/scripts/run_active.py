#!/usr/bin/env python3
"""
run_active.py — Phase 4: ACTIVE MODE with escalation ladder on Set D.

Verdict routing:
  CONTINUE -> nothing
  SUSPECT  -> escalate to Qwen3.5-2B A-T logprob scoring on iGPU;
              score in bottom quartile (or <= threshold) -> ABORT
  ABORT    -> roll back transcript to last CONTINUE step, resample
              temperature += 0.2, max 2 rollbacks per task

Configs (per task, 3 seeds each):
  baseline     -- no verification (shared with Phase 2 sweep)
  active_lfm   -- LFM-only active mode
  active_esc   -- LFM + Qwen escalation

Metrics per config: pass rate (bootstrap 95% CI), rollback conversion rate,
false aborts on passing trajectories, GPU-tokens proxy (steps + tool outputs),
wall time per solve, escalation rate, NPU verdict queue depth.
"""

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from agent_loop import AgentLoop  # noqa: E402
from verifier_client import NPUVerifierClient  # noqa: E402
from run_set_d import AgentLoopNoVerifier  # noqa: E402

GEN = "http://127.0.0.1:8012/v1"
NPU = "http://127.0.0.1:8001/v1"
ESC = "http://127.0.0.1:8013/v1"
DATA = os.path.join(REPO, "verifier", "data", "set_d_agentic.jsonl")
PROMPT_FILE = os.path.join(REPO, "verifier", "prompts", "final_verdict_prompt.txt")
OUT_DIR = os.path.join(REPO, "verifier", "results", "raw", "setd")
REPORT_DIR = os.path.join(REPO, "verifier", "results")


def load_tasks():
    with open(DATA) as f:
        return [json.loads(line) for line in f]


def bootstrap_ci(passes, n_boot=2000, ci=0.95, seed=7):
    rng = random.Random(seed)
    n = len(passes)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rates = []
    for _ in range(n_boot):
        sample = [passes[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(sample) / n)
    rates.sort()
    lo = rates[int((1 - ci) / 2 * n_boot)]
    hi = rates[int((1 + ci) / 2 * n_boot - 1)]
    return (sum(passes) / n, lo, hi)


def run_config(tasks, config, seeds, verifier, escalator=None):
    prompt = open(PROMPT_FILE).read()
    out = {"config": config, "tasks": {}}
    for task in tasks:
        recs = []
        for seed in seeds:
            if config == "baseline":
                loop = AgentLoopNoVerifier(GEN, max_steps=15, temperature=0.3)
            else:
                loop = AgentLoop(GEN, verifier, prompt, mode="active",
                                 escalator=escalator, max_steps=15,
                                 base_temperature=0.3)
            t0 = time.time()
            rec = loop.run(task, f"{config}{seed}")
            rec["config"] = config
            rec["seed"] = seed
            rec["wall_time"] = time.time() - t0
            fn = os.path.join(OUT_DIR, f"{task['id']}_{config}_{seed}.json")
            with open(fn, "w") as f:
                json.dump(rec, f, indent=2)
            recs.append(rec)
        out["tasks"][task["id"]] = {
            "passed": sum(r["passed"] for r in recs),
            "total": len(recs),
            "rollbacks": sum(r["rollbacks"] for r in recs),
            "escalations": sum(r["escalations"] for r in recs),
            "wall_time": sum(r["wall_time"] for r in recs) / len(recs),
        }
    return out


def summarize(configs, tasks):
    agg = {}
    for cfg, data in configs.items():
        passes = []
        for t in tasks:
            d = data["tasks"].get(t["id"])
            if d:
                passes.extend([1] * d["passed"] + [0] * (d["total"] - d["passed"]))
        p, lo, hi = bootstrap_ci(passes)
        agg[cfg] = {
            "pass_rate": round(p, 3),
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
            "n_runs": len(passes),
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    choices=["baseline", "active_lfm", "active_esc"],
                    default=["active_lfm", "active_esc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    verifier = None
    if any(c != "baseline" for c in args.configs):
        verifier = NPUVerifierClient(NPU)
        print("[*] warming NPU verifier ...", flush=True)
        verifier.evaluate_checkpoint("warm", "warm", "{task} {trajectory}",
                                     k_samples=3, temperature=0.7)

    escalator = None
    if "active_esc" in args.configs:
        from escalator import QwenEscalator
        escalator = QwenEscalator(ESC, threshold=0.34)

    results = {}
    for cfg in args.configs:
        print(f"\n=== config: {cfg} ===", flush=True)
        results[cfg] = run_config(
            tasks, cfg, args.seeds,
            verifier=verifier if cfg != "baseline" else None,
            escalator=escalator if cfg == "active_esc" else None,
        )

    summary = summarize(results, tasks)
    with open(os.path.join(REPORT_DIR, "active_summary.json"), "w") as f:
        json.dump({"per_config": summary,
                   "details": results}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
