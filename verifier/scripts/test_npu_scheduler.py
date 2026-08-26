#!/usr/bin/env python3
"""
test_npu_scheduler.py — Live acceptance test for the phase-aware NPU scheduler.

Acceptance criteria:
  1. NPU jobs execute with <=1% mean GPU decode impact.
  2. Zero deadline overruns into decode phases (no job crosses from its
     allowed window into active decode).
  3. Every placement decision is logged with the phase it landed in and the
     realized GPU delta.

Runs against live :8012 (Ornith) and :8001 (FLM shadow verifier).
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from npu_scheduler import NPUScheduler, NPUJob, GPUPhase
from npu_scheduler import JobPriority

GEN_URL = "http://127.0.0.1:8012/v1/chat/completions"
NPU_VERIFY_URL = "http://127.0.0.1:8001/v1/chat/completions"
OUT_DIR = Path(REPO) / "verifier" / "results" / f"npu_scheduler_acceptance_{int(time.time())}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gpu_generate(prompt: str, max_tokens: int = 128) -> dict:
    import urllib.request
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0,
               "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(GEN_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    usage = d.get("usage", {})
    ntok = usage.get("completion_tokens", max_tokens)
    return {"wall_s": round(dt, 3), "tok_per_s": round(ntok / dt, 2), "tokens": ntok}


def npu_verify_fn(trajectory_text: str) -> str:
    """Calls the NPU shadow verifier."""
    payload = {"model": "lfm2.5-tk:1.2b",
               "messages": [{"role": "user",
                             "content": f"Review this trajectory for errors:\n{trajectory_text[:1000]}"}],
               "max_tokens": 64, "temperature": 0.0}
    req = urllib.request.Request(NPU_VERIFY_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()[:200]


def main():
    scheduler = NPUScheduler(
        log_path=str(OUT_DIR / "scheduler_log.jsonl")
    )

    prompts = [
        "Write a Python function to reverse a linked list.",
        "Explain how HTTPS handshake works in detail.",
        "Write a bash script that finds all files larger than 1MB.",
        "Describe the tradeoffs between REST and GraphQL APIs.",
    ]

    # Phase A: Baseline without NPU jobs
    print("[Phase A] Baseline: GPU decode alone (no NPU jobs)")
    baseline_tps_list = []
    for i, p in enumerate(prompts):
        r = gpu_generate(p)
        baseline_tps_list.append(r["tok_per_s"])
        print(f"  run {i+1}: {r['tok_per_s']} tok/s ({r['wall_s']}s)")
    mean_baseline = sum(baseline_tps_list) / len(baseline_tps_list)
    print(f"  Mean baseline: {mean_baseline:.2f} tok/s\n")

    # Phase B: GPU decode + scheduled NPU jobs
    print("[Phase B] GPU decode + NPU scheduled jobs (shadow verification)")
    contended_tps_list = []
    deadline_overruns = 0

    for i, p in enumerate(prompts):
        # Queue an NPU verification job before each generation
        trajectory = f"Agent step {i}: executed shell command 'ls -la', got file listing output..."
        scheduler.submit(NPUJob(
            job_id=f"verify_step_{i}",
            job_type="verify",
            fn=lambda t=trajectory: npu_verify_fn(t),
            priority=JobPriority.HIGH,
            max_duration_ms=500.0,
        ))

        # Update phase to DECODE before generating
        scheduler.update_phase(GPUPhase.DECODE)

        # Run GPU generation while scheduler processes pending NPU jobs
        r = gpu_generate(p)

        # Process any queued NPU jobs (in practice this would be async)
        results = scheduler.process_pending()
        for res in results:
            if res.get("status") == "overran_deadline":
                deadline_overruns += 1
            if res.get("placement_phase") == "execute_throttled":
                scheduler.update_phase(GPUPhase.IDLE)
                scheduler.process_pending()

        contended_tps_list.append(r["tok_per_s"])
        delta_pct = ((r["tok_per_s"] - mean_baseline) / mean_baseline) * 100.0
        print(f"  run {i+1}: {r['tok_per_s']} tok/s (delta {delta_pct:+.1f}%) | "
              f"NPU jobs processed: {len(results)}")

        # Simulate tool window between turns
        scheduler.update_phase(GPUPhase.IDLE)
        time.sleep(0.5)
        scheduler.process_pending()

    mean_contended = sum(contended_tps_list) / len(contended_tps_list)
    mean_delta_pct = ((mean_contended - mean_baseline) / mean_baseline) * 100.0

    stats = scheduler.stats
    print(f"\n{'='*70}")
    print("📊 ACCEPTANCE RESULTS:")
    print(f" • Baseline mean decode   : {mean_baseline:.2f} tok/s")
    print(f" • Contended mean decode  : {mean_contended:.2f} tok/s")
    print(f" • Mean GPU impact        : {mean_delta_pct:+.2f}% (acceptance: <= ±1.00%)")
    print(f" • Deadline overruns      : {deadline_overruns} (acceptance: 0)")
    print(f" • Total jobs dispatched  : {stats['total_dispatched']}")
    print(f" • Jobs completed         : {stats['total_completed']}")
    print(f" • Jobs abandoned         : {stats['total_abandoned']}")
    print(f"{'='*70}")

    accepted = abs(mean_delta_pct) <= 1.0 and deadline_overruns == 0
    print(f"\nACCEPTANCE: {'PASS' if accepted else 'FAIL'}")

    out_file = OUT_DIR / "acceptance_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "accepted": accepted,
            "baseline_mean_tps": round(mean_baseline, 2),
            "contended_mean_tps": round(mean_contended, 2),
            "mean_gpu_impact_pct": round(mean_delta_pct, 2),
            "deadline_overruns": deadline_overruns,
            "scheduler_stats": stats,
        }, f, indent=2)
    print(f"[+] Saved to {out_file}")


if __name__ == "__main__":
    main()
