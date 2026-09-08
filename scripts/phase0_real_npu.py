#!/usr/bin/env python3
"""Measure iGPU MTP decode while the real XDNA 2 FLM model is generating."""

import json
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.request


LLAMA = os.environ.get("LLAMA_CLI_BIN", "build-strix-rocmfp4/bin/llama-cli")
MODEL = os.environ.get("QWEN38_GGUF", "models/Qwen3.8-27B-ROCmFP4-FAST.gguf")
NPU_URL = "http://127.0.0.1:13305/v1/chat/completions"
PROMPT = "Write a python function to compute the fibonacci sequence, then a second one for primes."


def gpu_bench() -> float:
    cmd = [
        LLAMA, "-m", MODEL, "-p", PROMPT, "-n", "96", "-ngl", "99",
        "-fa", "1", "-b", "2048", "-ub", "2048", "--device", "Vulkan0",
        "--no-mmap", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
        "--spec-draft-p-min", "0.0", "-st", "--simple-io", "-no-cnv",
        "--reasoning", "off",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    match = re.search(r"Generation:\s*([\d.]+)\s*t/s", result.stdout + result.stderr)
    if not match:
        raise RuntimeError(f"GPU benchmark failed with exit code {result.returncode}")
    return float(match.group(1))


class NPULoad:
    def __init__(self):
        self.stop = threading.Event()
        self.active = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.speeds = []
        self.tokens = 0
        self.errors = []

    def start(self):
        self.thread.start()
        if not self.active.wait(timeout=30):
            raise RuntimeError("NPU load did not start")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=30)

    def _run(self):
        payload = json.dumps({
            "model": "qwen3.5-0.8b-FLM",
            "messages": [{
                "role": "user",
                "content": (
                    "Write a long numbered list of concise facts about mathematics. "
                    "Continue until you have produced at least 250 tokens."
                ),
            }],
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False,
        }).encode()
        while not self.stop.is_set():
            request = urllib.request.Request(
                NPU_URL, data=payload, headers={"Content-Type": "application/json"}
            )
            self.active.set()
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = json.load(response)
                usage = data.get("usage", {})
                self.tokens += int(usage.get("completion_tokens", 0))
                speed = usage.get("decoding_speed_tps")
                if speed is not None:
                    self.speeds.append(float(speed))
            except Exception as exc:
                self.errors.append(str(exc))


def run_block(label: str, runs: int = 3):
    values = []
    for index in range(runs):
        value = gpu_bench()
        values.append(value)
        print(f"[{label}] run {index + 1}: {value:.1f} tok/s", flush=True)
    median = statistics.median(values)
    print(f"[{label}] median: {median:.1f} tok/s", flush=True)
    return values


def main():
    warmup = gpu_bench()
    print(f"[warmup] {warmup:.1f} tok/s", flush=True)
    before = run_block("baseline-before")

    load = NPULoad()
    load.start()
    time.sleep(10)
    during = run_block("real-npu-active")
    load.close()

    time.sleep(10)
    after = run_block("baseline-after")

    baseline = statistics.median(before + after)
    contended = statistics.median(during)
    degradation = 100 * (baseline - contended) / baseline
    print("\nRESULT", flush=True)
    print(f"baseline combined median: {baseline:.2f} tok/s", flush=True)
    print(f"real NPU active median:   {contended:.2f} tok/s", flush=True)
    print(f"GPU degradation:          {degradation:.2f}%", flush=True)
    print(f"NPU completion tokens:    {load.tokens}", flush=True)
    if load.speeds:
        print(f"NPU decode median:         {statistics.median(load.speeds):.2f} tok/s", flush=True)
    print(f"NPU request errors:        {len(load.errors)}", flush=True)


if __name__ == "__main__":
    main()
