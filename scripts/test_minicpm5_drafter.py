#!/usr/bin/env python3
"""
test_minicpm5_drafter.py — Comprehensive Speculative Drafter Benchmark
AMD XDNA 2 NPU (MiniCPM5-2B @ 63 tok/s) as Drafter
AMD Radeon 8060S iGPU (Ornith-1.5-35B-A3B ROCmFP4 @ 50-70 tok/s) as Target/Verifier
"""

import os
import sys
import time
import json
import asyncio
import aiohttp
from typing import List, Dict, Any

NPU_URL = "http://127.0.0.1:8001"
NPU_MODEL = "minicpm5:2b"
GPU_URL = "http://127.0.0.1:8012"
GPU_MODEL = "Ornith-1.5-35B-A3B-ROCmFP4.gguf"

BENCHMARK_PROMPTS = [
    {
        "category": "Factual",
        "prompt": "What is the capital of France? Answer in one short sentence.",
    },
    {
        "category": "Factual",
        "prompt": "Name the three primary colors in one short sentence.",
    },
    {
        "category": "Factual",
        "prompt": "Who wrote Romeo and Juliet? Answer in one short sentence.",
    },
    {
        "category": "Arithmetic",
        "prompt": "What is 15 * 23? Give only the number.",
    },
    {
        "category": "Arithmetic",
        "prompt": "Calculate 48 / 6. Give only the result.",
    },
    {
        "category": "Reasoning/Code",
        "prompt": "Write a one-line Python lambda function to check if a string is a palindrome.",
    },
    {
        "category": "Scientific",
        "prompt": "Explain photosynthesis in one concise sentence.",
    },
    {
        "category": "Creative",
        "prompt": "Write a 5-7-5 syllable haiku about the ocean waves.",
    },
]

async def call_npu(session: aiohttp.ClientSession, prompt: str, max_tokens: int = 32, retries: int = 2) -> Dict[str, Any]:
    payload = {
        "model": NPU_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Connection": "close"}
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            async with session.post(f"{NPU_URL}/v1/chat/completions", json=payload, headers=headers) as resp:
                data = await resp.json()
            t_elapsed = (time.perf_counter() - t0) * 1000
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {
                "content": content,
                "tokens": usage.get("completion_tokens", len(content.split())),
                "elapsed_ms": t_elapsed,
                "tps": usage.get("decoding_speed_tps", 0.0),
                "ttft_ms": usage.get("prefill_duration_ttft", 0.0) * 1000,
            }
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectorError) as e:
            if attempt < retries:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            return {"content": "", "tokens": 0, "elapsed_ms": 0.0, "tps": 0.0, "ttft_ms": 0.0}

async def call_gpu(session: aiohttp.ClientSession, prompt: str, max_tokens: int = 32) -> Dict[str, Any]:
    t0 = time.perf_counter()
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Connection": "close"}
    async with session.post(f"{GPU_URL}/v1/chat/completions", json=payload, headers=headers) as resp:
        data = await resp.json()
    t_elapsed = (time.perf_counter() - t0) * 1000
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    return {
        "content": content,
        "tokens": usage.get("completion_tokens", len(content.split())),
        "elapsed_ms": t_elapsed,
        "predicted_per_second": timings.get("predicted_per_second", 0.0),
        "prompt_ms": timings.get("prompt_ms", 0.0),
    }

async def call_speculative(session: aiohttp.ClientSession, prompt: str, draft_tokens: int = 15, max_tokens: int = 30) -> Dict[str, Any]:
    """
    Speculative Drafting Protocol:
    1. Draft tokens generated on XDNA 2 NPU via MiniCPM5-2B
    2. Verification on Radeon 8060S iGPU via Ornith-1.5-35B
    3. Calculate acceptance rate and end-to-end latency
    """
    t_start = time.perf_counter()
    headers = {"Connection": "close"}
    
    # 1. Draft on NPU
    t_draft_0 = time.perf_counter()
    draft_res = await call_npu(session, prompt, max_tokens=draft_tokens)
    t_draft_ms = (time.perf_counter() - t_draft_0) * 1000
    draft_text = draft_res["content"].strip()
    
    # 2. Tokenize draft text with verifier tokenizer
    t_verify_0 = time.perf_counter()
    async with session.post(f"{GPU_URL}/tokenize", json={"content": draft_text}, headers=headers) as resp:
        tok_data = await resp.json()
    verifier_tokens = tok_data.get("tokens", [])
    total_draft = len(verifier_tokens)
    
    # 3. Verify on GPU
    chatml_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{draft_text}"
    v_payload = {
        "prompt": chatml_prompt,
        "n_predict": max(1, max_tokens - total_draft),
        "temperature": 0.0,
        "cache_prompt": True,
        "echo": True,
        "logprobs": 5,
    }
    async with session.post(f"{GPU_URL}/v1/completions", json=v_payload, headers=headers) as resp:
        v_data = await resp.json()
    t_verify_ms = (time.perf_counter() - t_verify_0) * 1000
    t_total_ms = (time.perf_counter() - t_start) * 1000
    
    v_choice = v_data.get("choices", [{}])[0]
    logprobs_data = v_choice.get("logprobs") or {}
    v_tokens = logprobs_data.get("tokens", [])
    v_top_logprobs = logprobs_data.get("top_logprobs", [])
    
    accepted_count = 0
    assistant_pos = -1
    for idx, t in enumerate(v_tokens):
        if "assistant" in t:
            assistant_pos = idx + 1
            break
            
    if assistant_pos >= 0 and v_top_logprobs:
        for i in range(total_draft):
            curr_idx = assistant_pos + i
            if curr_idx < len(v_tokens) and curr_idx < len(v_top_logprobs):
                cand_token = v_tokens[curr_idx]
                top_dict = v_top_logprobs[curr_idx] or {}
                if top_dict:
                    best_tok = max(top_dict, key=top_dict.get)
                    if best_tok == cand_token:
                        accepted_count += 1
                    else:
                        break
                else:
                    accepted_count += 1
            else:
                break
    else:
        # Fallback word-level prefix match
        gpu_direct = await call_gpu(session, prompt, max_tokens=max_tokens)
        gpu_words = gpu_direct["content"].split()
        draft_words = draft_text.split()
        for w1, w2 in zip(draft_words, gpu_words):
            if w1.lower() == w2.lower():
                accepted_count += 1
            else:
                break
                
    acc_rate = (accepted_count / total_draft) if total_draft > 0 else 0.0
    
    return {
        "draft_text": draft_text,
        "total_draft": total_draft,
        "accepted": accepted_count,
        "acceptance_rate": acc_rate,
        "draft_ms": t_draft_ms,
        "verify_ms": t_verify_ms,
        "total_ms": t_total_ms,
    }

async def run_benchmark():
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        print("=" * 80)
        print("AMD STRIX HALO SPECULATIVE DECODING BENCHMARK")
        print(f"  NPU Drafter : AMD XDNA 2 ({NPU_MODEL}) @ {NPU_URL}")
        print(f"  iGPU Target : Radeon 8060S ({GPU_MODEL}) @ {GPU_URL}")
        print("=" * 80)
        
        results = []
        for i, item in enumerate(BENCHMARK_PROMPTS):
            category = item["category"]
            prompt = item["prompt"]
            print(f"\n[{i+1}/{len(BENCHMARK_PROMPTS)}] ({category}) '{prompt}'")
            
            # 1. Pure GPU
            gpu_res = await call_gpu(session, prompt, max_tokens=25)
            await asyncio.sleep(0.1)
            # 2. Pure NPU
            npu_res = await call_npu(session, prompt, max_tokens=25)
            await asyncio.sleep(0.1)
            # 3. Speculative (Draft + Verify)
            spec_res = await call_speculative(session, prompt, draft_tokens=12, max_tokens=25)
            await asyncio.sleep(0.1)
            
            print(f"  • GPU-Only   : {gpu_res['elapsed_ms']:6.1f} ms | {gpu_res['tokens']:2d} tok | '{gpu_res['content'].strip()[:60]}'")
            print(f"  • NPU-Only   : {npu_res['elapsed_ms']:6.1f} ms | {npu_res['tokens']:2d} tok | '{npu_res['content'].strip()[:60]}'")
            print(f"  • Speculative: {spec_res['total_ms']:6.1f} ms | Draft {spec_res['draft_ms']:.0f}ms, Verify {spec_res['verify_ms']:.0f}ms | Accept: {spec_res['accepted']}/{spec_res['total_draft']} ({spec_res['acceptance_rate']:.0%})")
            
            results.append({
                "category": category,
                "prompt": prompt,
                "gpu": gpu_res,
                "npu": npu_res,
                "speculative": spec_res,
            })
            
        print("\n" + "=" * 80)
        print("AGGREGATE BENCHMARK SUMMARY")
        print("=" * 80)
        avg_gpu_ms = sum(r["gpu"]["elapsed_ms"] for r in results) / len(results)
        avg_npu_ms = sum(r["npu"]["elapsed_ms"] for r in results) / len(results)
        avg_spec_ms = sum(r["speculative"]["total_ms"] for r in results) / len(results)
        avg_acc = sum(r["speculative"]["acceptance_rate"] for r in results) / len(results)
        avg_draft_ms = sum(r["speculative"]["draft_ms"] for r in results) / len(results)
        avg_verify_ms = sum(r["speculative"]["verify_ms"] for r in results) / len(results)
        
        print(f"{'Execution Mode':<25} | {'Avg Latency (ms)':<18} | {'Details'}")
        print("-" * 80)
        print(f"{'1. GPU-Only (Ornith 1.5)':<25} | {avg_gpu_ms:10.1f} ms     | Baseline RDNA 3.5 iGPU (MTP enabled)")
        print(f"{'2. NPU-Only (MiniCPM5-2B)':<25} | {avg_npu_ms:10.1f} ms     | AMD XDNA 2 NPU (63.6 tok/s decode)")
        print(f"{'3. Speculative (NPU+GPU)':<25} | {avg_spec_ms:10.1f} ms     | Draft={avg_draft_ms:.0f}ms, Verify={avg_verify_ms:.0f}ms, Acc={avg_acc:.1%}")
        print("=" * 80)
        
        delta_pct = ((avg_spec_ms - avg_gpu_ms) / avg_gpu_ms) * 100
        print(f"\nSpeculative vs GPU-Only Delta: {delta_pct:+.1f}% latency ({'Slower' if delta_pct > 0 else 'Faster'})")
        
        # Save results
        out_path = "/home/user/source/npuhalo/docs/minicpm5_drafter_benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump({"summary": {
                "avg_gpu_ms": avg_gpu_ms,
                "avg_npu_ms": avg_npu_ms,
                "avg_spec_ms": avg_spec_ms,
                "avg_acceptance_rate": avg_acc,
                "delta_pct": delta_pct,
            }, "runs": results}, f, indent=2)
        print(f"[+] Saved complete benchmark results to {out_path}\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
