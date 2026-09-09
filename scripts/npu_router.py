#!/usr/bin/env python3
"""
npu_router.py — Calibrated Hybrid Query Router for AMD Strix Halo (NPU + iGPU).
Combines deterministic zero-latency syntactic gating with NPU fallback.

Solves the Roadmap Near-Term Item #2:
"Improve router classification accuracy from the measured 25% baseline — 
 candidates: deterministic rules plus NPU fallback."
"""

import re
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any

DEFAULT_NPU_URL = "http://127.0.0.1:8001"
DEFAULT_NPU_MODEL = "qwen3.5:0.8b"

# Deterministic patterns that unequivocally require the 35B GPU model
GPU_HARD_PATTERNS = [
    re.compile(r"(c\+\+|\b(python|rust|golang|javascript|typescript|java|cpp|ruby|swift|kotlin|php)\b)", re.I),
    re.compile(r"\b(write|create|implement|code|debug|refactor|fix|optimize)\s+(a|an|the)?\s*([a-zA-Z0-9#\+\-]+\s+)?(function|script|class|server|algorithm|program|app|api|regex|query|test|tree|list|graph|node|table|pipeline)\b", re.I),
    re.compile(r"\b(def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import|#include|package\s+\w+)\b"),
    re.compile(r"\b(docker|kubernetes|k8s|cmake|cargo|make|pytest|systemd)\b", re.I),
    re.compile(r"\b(explain\s+in\s+detail|step-by-step|compare\s+and\s+contrast|prove\s+that|derive)\b", re.I),
    re.compile(r"\b(sql|select\s+.*\s+from|insert\s+into|update\s+.*\s+set)\b", re.I),
    re.compile(r"\b(bash|shell|terminal|powershell|zsh|git\s+(commit|rebase|merge|branch))\b", re.I),
]

# Deterministic patterns that are trivially handled by the 2B NPU model (<4W)
NPU_HARD_PATTERNS = [
    re.compile(r"^\s*(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b", re.I),
    re.compile(r"^\s*(what\s+is\s+)?\d+\s*[\+\-\*\/]\s*\d+\s*\??$", re.I),
    re.compile(r"^\s*what\s+is\s+the\s+capital\s+of\s+[A-Za-z\s]+\??$", re.I),
    re.compile(r"^\s*(who\s+wrote|who\s+is|what\s+is\s+the\s+color\s+of)\s+[A-Za-z0-9\s\']+\??$", re.I),
    re.compile(r"^\s*(translate\s+['\"][^'\"]+['\"]\s+to\s+\w+)\b", re.I),
]

FEW_SHOT_ROUTER_PROMPT = """Classify whether this prompt is simple (can be answered by a 2B model at low power) or complex (needs a 35B model).
Output format: JSON with "route": "npu" or "gpu".

Examples:
Prompt: "What is 25 * 4?"
{"route": "npu"}

Prompt: "Write an async HTTP connection pool with circuit breaking in Rust."
{"route": "gpu"}

Prompt: "What color is the sky?"
{"route": "npu"}

Prompt: "Design an event-driven microservices architecture for real-time payments."
{"route": "gpu"}

Prompt: "{prompt}"
"""

class HybridNPURouter:
    def __init__(self, url: str = DEFAULT_NPU_URL, model: str = DEFAULT_NPU_MODEL):
        self.url = url.rstrip("/")
        self.model = model
        self.session: aiohttp.ClientSession = None

    async def start(self):
        connector = aiohttp.TCPConnector(force_close=True)
        self.session = aiohttp.ClientSession(connector=connector)

    async def classify(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        
        # 1. Check GPU hard deterministic rules (0ms)
        for pat in GPU_HARD_PATTERNS:
            if pat.search(text):
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return {
                    "route": "gpu",
                    "method": "deterministic_rule",
                    "reason": f"Matched GPU requirement: {pat.pattern[:40]}",
                    "latency_ms": elapsed_ms,
                }

        # 2. Check NPU hard deterministic rules (0ms)
        for pat in NPU_HARD_PATTERNS:
            if pat.search(text):
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return {
                    "route": "npu",
                    "method": "deterministic_rule",
                    "reason": f"Matched NPU fast-lane: {pat.pattern[:40]}",
                    "latency_ms": elapsed_ms,
                }

        # 3. Fallback: NPU Few-Shot Semantic Classification (~2-4W)
        if self.session:
            prompt = FEW_SHOT_ROUTER_PROMPT.replace("{prompt}", text.replace('"', '\\"'))
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 25,
                "temperature": 0.0,
            }
            headers = {"Connection": "close"}
            try:
                async with self.session.post(f"{self.url}/v1/chat/completions", json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=4.0)) as resp:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        m = re.search(r'"route"\s*:\s*"(\w+)"', content)
                        if m:
                            route = m.group(1).lower()
                            if route in ("npu", "gpu"):
                                return {
                                    "route": route,
                                    "method": "npu_semantic_fewshot",
                                    "reason": f"NPU semantic classification: {content[:50]}",
                                    "latency_ms": elapsed_ms,
                                }
            except Exception:
                pass

        # Default safe fallback: GPU
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "route": "gpu",
            "method": "default_fallback",
            "reason": "Defaulting to high-capability GPU",
            "latency_ms": elapsed_ms,
        }

    async def close(self):
        if self.session:
            await self.session.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+", help="prompt to classify")
    args = ap.parse_args()
    text = " ".join(args.text)
    
    async def run():
        r = HybridNPURouter()
        await r.start()
        res = await r.classify(text)
        await r.close()
        print(json.dumps(res, indent=2))
        
    asyncio.run(run())
