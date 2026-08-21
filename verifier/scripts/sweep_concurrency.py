#!/usr/bin/env python3
"""
sweep_concurrency.py — Phase 5: NPU verifier capacity sweep.

Run 1, 2, 4, 6 concurrent agentic streams against the SINGLE NPU verifier.
Measure:
  - per-verdict NPU latency vs concurrency (median/p90)
  - the saturation point: stream count where median verdict latency exceeds
    the mean step time
  - generator tok/s degradation at each concurrency level

Prior estimate from the live eval: ~3.8 streams. Confirm or refute.
"""

import asyncio
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from verifier_client import NPUVerifierClient  # noqa: E402
from agent_loop import Step  # noqa: E402

NPU = "http://127.0.0.1:8001/v1"
GEN = "http://127.0.0.1:8012/v1"
OUT_DIR = os.path.join(REPO, "verifier", "results", "sweep")


def timed_verdict(client, task, steps, tag):
    t0 = time.time()
    res = client.evaluate_checkpoint(
        task_prompt="AGENT TASK",
        trajectory="\n".join(f"[{s.index}] exit={s.tool_result.get('exit_code')}"
                              for s in steps[-2:]),
        prompt_template="{task}\n\n{trajectory}",
        k_samples=3, temperature=0.7,
    )
    dt = time.time() - t0
    return {"latency_ms": dt * 1000, "tag": tag, "verdict": res.get("verdict")}


async def run_level(client, n_streams, rounds=6):
    tasks = [{"instruction": f"task {i} debug a repo"} for i in range(n_streams)]
    steps = [[Step(1, 0.3, "", {"name": "shell", "arguments": {}},
                   {"exit_code": 0, "stdout": "ok", "stderr": ""}, 0)] for _ in range(n_streams)]

    latencies = []
    for r in range(rounds):
        cors = []
        for i in range(n_streams):
            cors.append(asyncio.to_thread(timed_verdict, client, tasks[i], steps[i], f"{i}-{r}"))
        results = await asyncio.gather(*cors)
        latencies.extend(x["latency_ms"] for x in results)

    return latencies


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = NPUVerifierClient(NPU)
    print("[*] warming NPU ...", flush=True)
    client.evaluate_checkpoint("w", "w", "{task} {trajectory}", k_samples=3, temperature=0.7)

    report = {"levels": {}}
    for n in (1, 2, 4, 6):
        print(f"\n=== concurrency {n} ===", flush=True)
        lats = await run_level(client, n)
        med = statistics.median(lats)
        p90 = sorted(lats)[int(len(lats) * 0.9)]
        report["levels"][n] = {
            "median_ms": round(med, 1),
            "p90_ms": round(p90, 1),
            "n": len(lats),
        }
        print(f"  median={med:.0f}ms p90={p90:.0f}ms")

    # saturation: median > mean step time. Mean step time is ~1.5-2.5s on this
    # workload (short tool step + Ornith decode); report median in ms and flag
    # the first level where median exceeds 2000ms.
    report["saturation_estimate"] = {
        "assumed_mean_step_ms": 2000,
        "saturates_at": next(
            (n for n in (1, 2, 4, 6)
             if report["levels"][n]["median_ms"] > 2000), None),
    }
    with open(os.path.join(OUT_DIR, "sweep_summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
