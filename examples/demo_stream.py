#!/usr/bin/env python3
"""
examples/demo_stream.py — Visual interactive demo of NPU live verification.

Streams tokens live from the iGPU generator (:8012) while the XDNA 2 NPU
(:8001) audits intermediate checkpoints in the background.

Prints color-coded NPU verification badges directly in the terminal:
  [✓ CONTINUE]  -- NPU confirms trajectory is sound (<1.0s, 0 GPU tokens)
  [? SUSPECT]   -- NPU flags potential flaw / ambiguity
  [✗ ABORT]     -- NPU halts execution and triggers rollback

Usage:
  python3 examples/demo_stream.py
  python3 examples/demo_stream.py --prompt "Write a quicksort algorithm in Python with tests."
"""

import argparse
import asyncio
import os
import sys
import time

# Add verifier src to path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from verifier_client import NPUVerifierClient, LFMVerifierClient  # noqa: E402
from stream_tap import StreamTapProxy  # noqa: E402

# ANSI formatting
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
CYAN = "\033[1;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DEFAULT_PROMPT = "Write a complete Python implementation of Quicksort with an explanation and unit tests."
PROMPT_TEMPLATE_FILE = os.path.join(REPO, "verifier", "prompts", "final_verdict_prompt.txt")


async def run_demo(prompt: str, gen_url: str, npu_url: str, checkpoint_tokens: int, max_tokens: int):
    print(f"\n{BOLD}{CYAN}⚡ NPUHalo: Live Streaming Verification on AMD Strix Halo{RESET}")
    print(f"{DIM}Generator: {gen_url} (Radeon 8060S iGPU, ROCmFP4 MoE)")
    print(f"Verifier : {npu_url} (XDNA 2 NPU, FastFlowLM LFM2.5-tk){RESET}\n")

    # Load prompt template
    if os.path.exists(PROMPT_TEMPLATE_FILE):
        with open(PROMPT_TEMPLATE_FILE) as f:
            template = f.read()
    else:
        template = "Task:\n{task}\n\nTrajectory:\n{trajectory}\n\nVerdict:"

    # Initialize NPU verifier client
    try:
        verifier = NPUVerifierClient(npu_url)
        print(f"[*] Connected to NPU verifier ({verifier._pick_model_id()})")
    except Exception as e:
        print(f"[*] NPU endpoint unreachable ({e}), falling back to local PyTorch...")
        verifier = LFMVerifierClient()

    proxy = StreamTapProxy(
        verifier=verifier,
        generator_url=gen_url if gen_url.endswith("/chat/completions") else f"{gen_url.rstrip('/')}/chat/completions",
        token_limit=checkpoint_tokens,
    )

    verdict_history = []

    def on_verdict(res):
        verdict_history.append(res)
        v = res.get("verdict", "CONTINUE")
        lat = res.get("latency_ms", 0.0)
        idx = res.get("checkpoint_index", len(verdict_history))
        trig = res.get("trigger", "checkpoint")
        ev = res.get("evidence", "")

        if v == "CONTINUE":
            badge = f"\n{GREEN}[✓ NPU Checkpoint #{idx}: CONTINUE | {lat:.0f}ms | {trig}]{RESET}\n"
        elif v == "SUSPECT":
            badge = f"\n{YELLOW}[? NPU Checkpoint #{idx}: SUSPECT | {lat:.0f}ms | {trig}]{RESET}\n"
        else:  # ABORT
            badge = f"\n{RED}[✗ NPU Checkpoint #{idx}: ABORT | {lat:.0f}ms | {ev or trig}]{RESET}\n"

        sys.stdout.write(badge)
        sys.stdout.flush()

    print(f"\n{BOLD}Prompt:{RESET} {prompt}\n")
    print(f"{DIM}{'─' * 70}{RESET}")

    t_start = time.time()
    total_tokens = 0
    aborted = False

    async for event in proxy.stream_and_verify(
        task_prompt=prompt,
        prompt_template=template,
        temperature=0.3,
        max_tokens=max_tokens,
        on_verdict_cb=on_verdict,
    ):
        if event["type"] == "token":
            total_tokens += 1
            sys.stdout.write(event["token"])
            sys.stdout.flush()
        elif event["type"] == "abort":
            aborted = True
            sys.stdout.write(f"\n{RED}{BOLD}⚡ STREAM TERMINATED BY NPU AUDITOR (early fault caught){RESET}\n")
            sys.stdout.flush()
            break

    wall_time = time.time() - t_start
    tps = total_tokens / wall_time if wall_time > 0 else 0.0

    print(f"\n{DIM}{'─' * 70}{RESET}")
    print(f"\n{BOLD}📊 Run Summary:{RESET}")
    print(f" • Total Tokens Generated : {BOLD}{total_tokens}{RESET}")
    print(f" • Generation Wall Time   : {wall_time:.2f} s")
    print(f" • Effective Throughput   : {BOLD}{tps:.1f} tok/s{RESET}")
    print(f" • NPU Audits Completed   : {len(verdict_history)}")
    if verdict_history:
        avg_lat = sum(r.get("latency_ms", 0) for r in verdict_history) / len(verdict_history)
        print(f" • Mean NPU Checkpoint Lat: {BOLD}{avg_lat:.0f} ms{RESET} (off-GPU, ~2 W power)")
    print(f" • Final Stream Status    : {RED + 'ABORTED' if aborted else GREEN + 'PASSED'}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="NPUHalo Live Streaming Verification Demo")
    parser.add_argument("--prompt", "-p", default=DEFAULT_PROMPT, help="Prompt to send to the generator")
    parser.add_argument("--generator-url", default="http://127.0.0.1:8012/v1", help="Generator OpenAI /v1 endpoint")
    parser.add_argument("--npu-url", default="http://127.0.0.1:8001/v1", help="NPU verifier OpenAI /v1 endpoint")
    parser.add_argument("--checkpoint-tokens", "-c", type=int, default=150, help="Tokens between fallback checkpoints")
    parser.add_argument("--max-tokens", "-n", type=int, default=350, help="Max tokens to generate")
    args = parser.parse_args()

    asyncio.run(run_demo(
        prompt=args.prompt,
        gen_url=args.generator_url,
        npu_url=args.npu_url,
        checkpoint_tokens=args.checkpoint_tokens,
        max_tokens=args.max_tokens,
    ))


if __name__ == "__main__":
    main()
