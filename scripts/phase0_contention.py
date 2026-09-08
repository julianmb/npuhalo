#!/usr/bin/env python3
"""
phase0_contention.py — Phase 0 contention-proxy experiment for the NPU pre-drafting idea.

Question: does concurrent DRAM read traffic (what an NPU draft head would add while
overlapped drafting) degrade embedded-MTP decode by more than the 5% gate?

Each level: start bw_stress (subprocess group) -> settle (thermal steady state)
-> N benchmark runs -> kill stress group -> cooldown.

Usage:
  python3 scripts/phase0_contention.py baseline
  python3 scripts/phase0_contention.py 8      # ~8 GB/s added traffic
  python3 scripts/phase0_contention.py 16
  python3 scripts/phase0_contention.py 24
"""

import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

LLAMA = os.environ.get("LLAMA_CLI_BIN", "build-strix-rocmfp4/bin/llama-cli")
MODEL = os.environ.get("QWEN38_GGUF", "models/Qwen3.8-27B-ROCmFP4-FAST.gguf")
PROMPT = "Write a python function to compute the fibonacci sequence, then a second one for primes."
STRESS = ["python3", os.path.join(os.path.dirname(__file__), "bw_stress.py")]
RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/npuhalo-{os.getuid()}")) / "npuhalo"
SETTLE = 45   # s of sustained load before measuring (thermal steady state)
RUNS = 3
COOLDOWN = 20 # s after killing stress

def bench():
    cmd = [LLAMA, "-m", MODEL, "-p", PROMPT, "-n", "96", "-ngl", "99", "-fa", "1",
           "-b", "2048", "-ub", "2048", "--device", "Vulkan0", "--no-mmap",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--spec-draft-p-min", "0.0",
           "-st", "--simple-io", "-no-cnv", "--reasoning", "off"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", r.stdout + r.stderr)
    return float(m.group(1)) if m else 0.0

def run_level(label, target, workers):
    stress = None
    if target is not None:
        RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(RUNTIME_DIR, 0o700)
        stress_log = (RUNTIME_DIR / "phase0_stress.log").open("ab", buffering=0)
        stress = subprocess.Popen(
            STRESS + ["--target", str(target), "--workers", str(workers),
                      "--duration", str(SETTLE + 400)],
            stdout=stress_log, stderr=stress_log,
            stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"[{label}] stress pid={stress.pid} (target {target} GB/s), settling {SETTLE}s ...", flush=True)
        time.sleep(SETTLE)
    results = []
    warm = bench()  # warm-up (clock ramp / cold caches), not recorded
    print(f"[{label}] warmup: {warm:.1f} tok/s", flush=True)
    for i in range(RUNS):
        tps = bench()
        results.append(tps)
        print(f"[{label}] run{i+1}: {tps:.1f} tok/s", flush=True)
    if stress is not None:
        try:
            os.killpg(os.getpgid(stress.pid), signal.SIGKILL)
        except Exception:
            pass
        time.sleep(COOLDOWN)
    med = statistics.median(results)
    print(f"[{label}] MEDIAN: {med:.1f} tok/s  (runs: {['%.1f' % r for r in results]})", flush=True)
    return med

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    levels = {
        "baseline": ("baseline", None, 1),
        "3":  ("3GBps",   3, 1),
        "4":  ("4GBps",   4, 1),
        "8":  ("8GBps",  8, 1),
        "16": ("16GBps", 16, 1),
        "24": ("24GBps", 12, 2),   # 2 workers x 12 GB/s (single worker peaks ~23)
    }
    if which not in levels:
        sys.exit(f"unknown level {which!r}; use: baseline | 8 | 16 | 24")
    label, target, workers = levels[which]
    run_level(label, target, workers)

if __name__ == "__main__":
    main()
