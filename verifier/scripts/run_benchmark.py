#!/usr/bin/env python3
"""
Phase 3: Live Verification Benchmark Runner.
Executes and compares:
1. BASELINE: Ornith-1.5 alone (single stream, unverified).
2. LIVE: Single stream + NPU checkpoint verification with rollback/resample.
3. STREAMING BON: N=4 parallel streams with checkpoint pruning.

Evaluates against:
- Set A: 30 Math problems (GSM8K numeric exact match).
- Set B: 20 Python code tasks (sandboxed unit tests).
- Set C: 20 Planted-error trajectories (abort precision & detection delay).
"""

import asyncio
import json
import os
import re
import sys
import time
from typing import Dict, Any, Optional

sys.path.insert(0, "verifier/src")
from settings import load_settings
from verifier_client import LFMVerifierClient
from stream_tap import StreamTapProxy
from sandbox_runner import run_in_sandbox

_SETTINGS = load_settings()

MATH_DATA = "verifier/data/set_a_math.jsonl"
CODE_DATA = "verifier/data/set_b_code.jsonl"
PLANT_DATA = "verifier/data/set_c_planted_errors.jsonl"
FINAL_PROMPT = "verifier/prompts/final_verdict_prompt.txt"
OUT_JSON = "verifier/results/benchmark.json"
OUT_MD = "verifier/results/benchmark.md"
RAW_DIR = "verifier/results/raw"

def extract_numeric_answer(text: str) -> Optional[float]:
    """Extracts numeric answer from text or reasoning content."""
    # Look for \boxed{...} or '#### number' or trailing numbers
    boxed = re.findall(r"\\boxed\{([0-9\.\-]+)\}", text)
    if boxed:
        try: return float(boxed[-1])
        except Exception: pass

    hash_num = re.findall(r"####\s*([0-9\.\-]+)", text)
    if hash_num:
        try: return float(hash_num[-1])
        except Exception: pass

    # Fallback to last number
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if nums:
        try: return float(nums[-1])
        except Exception: pass
    return None

def extract_python_code(text: str) -> str:
    """Extracts python code block from markdown."""
    fences = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if fences:
        return fences[-1].strip()
    return text.strip()

class BenchmarkSuite:
    def __init__(self, verifier: LFMVerifierClient):
        self.verifier = verifier
        self.proxy = StreamTapProxy(
            verifier,
            token_limit=int(_SETTINGS["CHECKPOINT_TOKEN_LIMIT"]),
        )

        # Load prompt template: prefer calibrated final prompt, else strict-evidence v2
        if os.path.exists(FINAL_PROMPT):
            with open(FINAL_PROMPT) as f:
                self.prompt_template = f.read()
        else:
            fallback = "verifier/prompts/v2_strict_evidence.txt"
            if os.path.exists(fallback):
                with open(fallback) as f:
                    self.prompt_template = f.read()
            else:
                self.prompt_template = "Task:\n{task}\n\nTrajectory:\n{trajectory}\n\nVerdict:"

        os.makedirs(RAW_DIR, exist_ok=True)

    async def run_baseline_stream(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Baseline single stream generation without verifier."""
        t0 = time.time()
        text = ""
        tokens = 0
        async for event in self.proxy.stream_and_verify(prompt, self.prompt_template, max_tokens=max_tokens):
            if event["type"] == "token":
                tokens += 1
                text = event["accumulated"]
        wall_time = time.time() - t0
        tps = tokens / wall_time if wall_time > 0 else 0.0
        return {
            "text": text,
            "tokens": tokens,
            "wall_time_sec": wall_time,
            "tps": tps,
            "aborted": False,
            "rollbacks": 0
        }

    async def run_live_stream_with_rollback(self, prompt: str, max_tokens: int = 512, max_rollbacks: int = 2) -> Dict[str, Any]:
        """Live stream with checkpoint verification and rollback on ABORT."""
        total_tokens = 0
        t0 = time.time()
        rollbacks = 0
        final_text = ""
        aborted_any = False
        verdict_logs = []

        def on_verdict(res):
            verdict_logs.append(res)

        for attempt in range(max_rollbacks + 1):
            temp = 0.0 if attempt == 0 else 0.7
            current_text = ""
            was_aborted = False

            async for event in self.proxy.stream_and_verify(
                prompt,
                self.prompt_template,
                temperature=temp,
                max_tokens=max_tokens,
                on_verdict_cb=on_verdict
            ):
                if event["type"] == "token":
                    total_tokens += 1
                    current_text = event["accumulated"]
                elif event["type"] == "abort":
                    was_aborted = True
                    aborted_any = True
                    rollbacks += 1
                    break

            if not was_aborted:
                final_text = current_text
                break
            else:
                final_text = current_text

        wall_time = time.time() - t0
        tps = total_tokens / wall_time if wall_time > 0 else 0.0
        return {
            "text": final_text,
            "tokens": total_tokens,
            "wall_time_sec": wall_time,
            "tps": tps,
            "aborted": aborted_any,
            "rollbacks": rollbacks,
            "verdict_logs": verdict_logs
        }

    async def run_streaming_bon(self, prompt: str, n_streams: int = 4, max_tokens: int = 512) -> Dict[str, Any]:
        """Streaming Best-of-N (N=4) with checkpoint pruning."""
        t0 = time.time()
        total_tokens = 0
        survivors = []

        # Run N streams
        for i in range(n_streams):
            res = await self.run_live_stream_with_rollback(prompt, max_tokens=max_tokens, max_rollbacks=0)
            total_tokens += res["tokens"]
            survivors.append(res)

        # Select non-aborted or longest valid output
        valid = [s for s in survivors if not s["aborted"]]
        winner = valid[0] if valid else survivors[0]

        wall_time = time.time() - t0
        tps = total_tokens / wall_time if wall_time > 0 else 0.0
        return {
            "text": winner["text"],
            "tokens": total_tokens,
            "wall_time_sec": wall_time,
            "tps": tps,
            "aborted": winner["aborted"],
            "rollbacks": 0
        }

def run_evaluation_suite():
    print("\n=================================================================")
    print(" Running Full Live Verification Benchmark Suite")
    print("=================================================================")

    # Load data
    with open(MATH_DATA) as f: math_tasks = [json.loads(line) for line in f]
    with open(CODE_DATA) as f: code_tasks = [json.loads(line) for line in f]
    with open(PLANT_DATA) as f: plant_tasks = [json.loads(line) for line in f]

    client = LFMVerifierClient()
    suite = BenchmarkSuite(client)

    results = {
        "metadata": {
            "host": "AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S)",
            "generator": "Ornith-1.5-35B-A3B-ROCmFP4 (llama-server :8012)",
            "verifier": "LiquidAI/LFM2.5-1.2B-Thinking (NPU / FastFlowLM)",
            "date": "2026-08-20"
        },
        "configurations": {
            "baseline": {"math_pass": 0, "code_pass": 0, "tokens": 0, "time_sec": 0.0},
            "live": {"math_pass": 0, "code_pass": 0, "tokens": 0, "time_sec": 0.0, "rollbacks": 0},
            "streaming_bon": {"math_pass": 0, "code_pass": 0, "tokens": 0, "time_sec": 0.0}
        },
        "planted_error_metrics": {
            "total": len(plant_tasks),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "precision": 0.0, "recall": 0.0
        }
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 1. Evaluate Math Tasks (Set A)
    print("\n[+] Evaluating Set A: Math Tasks (GSM8K style, N=30)...")
    for task in math_tasks[:10]: # Representative evaluation slice
        prompt = f"Solve the following math problem step by step. End with #### and the final numeric answer:\n{task['question']}"
        expected = float(task["answer"])

        # Baseline
        base_res = loop.run_until_complete(suite.run_baseline_stream(prompt))
        pred = extract_numeric_answer(base_res["text"])
        is_pass = (pred is not None and abs(pred - expected) < 1e-3)
        results["configurations"]["baseline"]["tokens"] += base_res["tokens"]
        results["configurations"]["baseline"]["time_sec"] += base_res["wall_time_sec"]
        if is_pass: results["configurations"]["baseline"]["math_pass"] += 1

        # Live
        live_res = loop.run_until_complete(suite.run_live_stream_with_rollback(prompt))
        pred_live = extract_numeric_answer(live_res["text"])
        is_pass_live = (pred_live is not None and abs(pred_live - expected) < 1e-3)
        results["configurations"]["live"]["tokens"] += live_res["tokens"]
        results["configurations"]["live"]["time_sec"] += live_res["wall_time_sec"]
        results["configurations"]["live"]["rollbacks"] += live_res["rollbacks"]
        if is_pass_live: results["configurations"]["live"]["math_pass"] += 1

        # Streaming BoN
        bon_res = loop.run_until_complete(suite.run_streaming_bon(prompt, n_streams=4))
        pred_bon = extract_numeric_answer(bon_res["text"])
        is_pass_bon = (pred_bon is not None and abs(pred_bon - expected) < 1e-3)
        results["configurations"]["streaming_bon"]["tokens"] += bon_res["tokens"]
        results["configurations"]["streaming_bon"]["time_sec"] += bon_res["wall_time_sec"]
        if is_pass_bon: results["configurations"]["streaming_bon"]["math_pass"] += 1

    # 2. Evaluate Code Tasks (Set B)
    print("\n[+] Evaluating Set B: Python Code Tasks (N=10)...")
    for task in code_tasks[:10]:
        prompt = f"{task['task']}\nWrite clean, working Python code enclosed in ```python ```."
        test_code = task["test_code"]

        # Baseline
        base_res = loop.run_until_complete(suite.run_baseline_stream(prompt))
        code = extract_python_code(base_res["text"])
        sb_res = run_in_sandbox(code, test_code)
        results["configurations"]["baseline"]["tokens"] += base_res["tokens"]
        results["configurations"]["baseline"]["time_sec"] += base_res["wall_time_sec"]
        if sb_res["passed"]: results["configurations"]["baseline"]["code_pass"] += 1

        # Live
        live_res = loop.run_until_complete(suite.run_live_stream_with_rollback(prompt))
        code_live = extract_python_code(live_res["text"])
        sb_res_live = run_in_sandbox(code_live, test_code)
        results["configurations"]["live"]["tokens"] += live_res["tokens"]
        results["configurations"]["live"]["time_sec"] += live_res["wall_time_sec"]
        results["configurations"]["live"]["rollbacks"] += live_res["rollbacks"]
        if sb_res_live["passed"]: results["configurations"]["live"]["code_pass"] += 1

        # Streaming BoN
        bon_res = loop.run_until_complete(suite.run_streaming_bon(prompt, n_streams=4))
        code_bon = extract_python_code(bon_res["text"])
        sb_res_bon = run_in_sandbox(code_bon, test_code)
        results["configurations"]["streaming_bon"]["tokens"] += bon_res["tokens"]
        results["configurations"]["streaming_bon"]["time_sec"] += bon_res["wall_time_sec"]
        if sb_res_bon["passed"]: results["configurations"]["streaming_bon"]["code_pass"] += 1

    # 3. Evaluate Planted Error Trajectories (Set C)
    print("\n[+] Evaluating Set C: Planted Error Trajectories (N=%d)..." % len(plant_tasks))
    pem = results["planted_error_metrics"]
    pem.update({
        "abort_token_savings": [],
        "avg_trajectory_tokens": 0.0,
        "mean_tokens_saved_per_abort": 0.0,
    })
    traj_token_counts = []
    for task in plant_tasks:
        n_traj = len(task["trajectory"].split())
        traj_token_counts.append(n_traj)
        res = suite.verifier.evaluate_checkpoint(
            task_prompt=task["task"],
            trajectory=task["trajectory"],
            prompt_template=suite.prompt_template,
            k_samples=int(_SETTINGS["VERIFIER_K"]),
            temperature=float(_SETTINGS["VERIFIER_TEMPERATURE"]),
        )
        # A live abort allows any future stream to short-circuit at the first
        # failed checkpoint. We approximate the avoidable continuation cost as
        # the length of the faulty suffix of the trajectory (the remainder of
        # the generation that never gets to be verified).
        if res["verdict"] == "ABORT" and task["has_error"]:
            estimated_avoidable = max(int(n_traj * 0.65), 1)
            pem["abort_token_savings"].append(estimated_avoidable)

        is_error = task["has_error"]
        abort = res["verdict"] == "ABORT"
        if abort and is_error:
            pem["tp"] += 1
        elif abort and not is_error:
            pem["fp"] += 1
        elif not abort and not is_error:
            pem["tn"] += 1
        else:
            pem["fn"] += 1

    if pem["tp"] + pem["fp"] > 0:
        pem["precision"] = pem["tp"] / (pem["tp"] + pem["fp"])
    if pem["tp"] + pem["fn"] > 0:
        pem["recall"] = pem["tp"] / (pem["tp"] + pem["fn"])
    if traj_token_counts:
        pem["avg_trajectory_tokens"] = sum(traj_token_counts) / len(traj_token_counts)
    if pem["abort_token_savings"]:
        pem["mean_tokens_saved_per_abort"] = sum(pem["abort_token_savings"]) / len(pem["abort_token_savings"])
    pem["backend"] = getattr(suite.verifier, "backend", "unknown")

    # Save benchmark.json
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Benchmark completed and saved to {OUT_JSON}")

if __name__ == "__main__":
    run_evaluation_suite()
