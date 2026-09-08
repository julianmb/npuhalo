#!/usr/bin/env python3
"""
sweep_mtp.py — Empirical Parameter Sweep for Qwen 3.8 27B MTP on Strix Halo iGPU
Tests combinations of:
 - K draft depth: 2, 3, 4, 5, 6
 - strict-qwen: ON vs OFF
 - draft p_min: 0.0, 0.60
"""

import os
import sys
import re
import time
import json
import subprocess
from pathlib import Path

MODEL_PATH = os.environ.get("QWEN38_GGUF", "models/Qwen3.8-27B-ROCmFP4-FAST.gguf")
LLAMA_CLI_BIN = os.environ.get("LLAMA_CLI_BIN", "build-strix-rocmfp4/bin/llama-cli")
PROMPT = "Write a clear, concise guide explaining how speculative decoding accelerates transformer inference."

def run_single_benchmark(k: int, strict_qwen: bool, p_min: float, device: str = "Vulkan0") -> dict:
    cmd = [
        LLAMA_CLI_BIN,
        "-m", MODEL_PATH,
        "-p", PROMPT,
        "-n", "48",
        "-ngl", "99",
        "-fa", "1",
        "-b", "2048",
        "-ub", "2048",
        "--device", device,
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(k),
        "--spec-draft-p-min", str(p_min),
        "--no-mmap",
        "-st", "--simple-io", "-no-cnv",
        "--reasoning", "off",
    ]

    if strict_qwen:
        cmd.append("--spec-mtp-strict-qwen")

    env = os.environ.copy()
    env["ROCBLAS_USE_HIPBLASLT"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    elapsed = time.perf_counter() - t0

    out = proc.stdout + proc.stderr
    prompt_ts = 0.0
    gen_ts = 0.0

    match = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", out)
    if match:
        prompt_ts = float(match.group(1))
        gen_ts = float(match.group(2))

    return {
        "k": k,
        "strict_qwen": strict_qwen,
        "p_min": p_min,
        "device": device,
        "prompt_ts": prompt_ts,
        "gen_ts": gen_ts,
        "total_time_s": elapsed,
        "success": (gen_ts > 0.0)
    }

def main():
    print("=" * 80)
    print(" 🚀 RUNNING MTP PARAMETER SWEEP ON QWEN 3.8 27B")
    print("=" * 80)
    sys.stdout.flush()

    k_values = [2, 3, 4, 6]
    strict_options = [False, True]
    p_min_options = [0.0, 0.60]

    results = []

    print(f"\n{'Device':<8} | {'K':<3} | {'Strict':<7} | {'p_min':<5} | {'Prompt t/s':<11} | {'Decode t/s':<11} | {'Time (s)':<8}")
    print("-" * 80)
    sys.stdout.flush()

    for strict in strict_options:
        for k in k_values:
            for p_min in p_min_options:
                res = run_single_benchmark(k=k, strict_qwen=strict, p_min=p_min, device="Vulkan0")
                results.append(res)
                strict_str = "ON" if strict else "OFF"
                print(f"{res['device']:<8} | {k:<3} | {strict_str:<7} | {p_min:<5.2f} | {res['prompt_ts']:<11.1f} | {res['gen_ts']:<11.1f} | {res['total_time_s']:<8.2f}")
                sys.stdout.flush()

    valid_results = [r for r in results if r["success"]]
    if valid_results:
        best = max(valid_results, key=lambda x: x["gen_ts"])
        print("=" * 80)
        print(" 🏆 BEST MTP CONFIGURATION (Vulkan0):")
        print(f"    • K depth:       {best['k']}")
        print(f"    • Strict Qwen:   {'ON' if best['strict_qwen'] else 'OFF'}")
        print(f"    • p_min:         {best['p_min']}")
        print(f"    • Prompt Speed:  {best['prompt_ts']:.1f} tok/s")
        print(f"    • Decode Speed:  {best['gen_ts']:.1f} tok/s")
        print("=" * 80)

    out_json = Path("docs/mtp_sweep_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results written to {out_json}")

if __name__ == "__main__":
    main()
