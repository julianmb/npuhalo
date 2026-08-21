#!/usr/bin/env python3
"""
npu_router.py — Always-On Prompt Router / Intent Classifier on the XDNA 2 NPU.

Runs the 0.8B model on /dev/accel/accel0 at ~2W to classify each incoming prompt
before the 27B iGPU model is woken. Outputs a routing decision that lets the
pipeline pick the cheapest/fastest path:

  - INTENT:    chat | code | math | classification | translation | toolcall
  - PRIORITY:  short (NPU handles it) vs long (route to 27B iGPU)
  - HYBRID:    whether to enable the NPU burst (long-prompt TTFT optimization)

Usage:
  python3 scripts/npu_router.py "write a python function to reverse a string"
"""

import asyncio
import argparse
import json
import re
import aiohttp

NPU_URL = "http://127.0.0.1:13305"
NPU_MODEL = "qwen3.5-0.8b-FLM"

ROUTER_PROMPT = (
    "Classify the following user request. Reply with a single JSON object with keys:\n"
    '  "intent": one of chat|code|math|classification|translation|toolcall\n'
    '  "length": short|medium|long\n'
    '  "route": npu|gpu\n'
    "Use route=npu for simple facts, greetings, or single-word answers; "
    "route=gpu for anything requiring reasoning, code, math, or long output.\n"
    "Request:\n"
)

class NPURouter:
    def __init__(self, url=NPU_URL, model=NPU_MODEL):
        self.url = url
        self.model = model
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

    async def classify(self, text: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": ROUTER_PROMPT + text}],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        async with self.session.post(f"{self.url}/v1/chat/completions", json=payload) as r:
            if r.status != 200:
                return {"error": f"npu status {r.status}", "route": "gpu"}
            data = await r.json()
            content = data["choices"][0]["message"]["content"]
            return self._parse(content)

    @staticmethod
    def _parse(content: str) -> dict:
        # try full JSON
        m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                d.setdefault("route", "gpu")
                return d
            except json.JSONDecodeError:
                pass
        # repair truncated JSON: extract key:value pairs by regex
        intent = re.search(r'"intent"\s*:\s*"?(\w+)', content)
        length = re.search(r'"length"\s*:\s*"?(\w+)', content)
        route  = re.search(r'"route"\s*:\s*"?(\w+)', content)
        if intent or route:
            return {
                "intent": intent.group(1) if intent else "chat",
                "length": length.group(1) if length else "medium",
                "route":  route.group(1) if route else "gpu",
            }
        return {"raw": content.strip(), "route": "gpu"}

    async def close(self):
        if self.session:
            await self.session.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+", help="prompt to classify")
    args = ap.parse_args()
    text = " ".join(args.text)

    router = NPURouter()
    await router.start()
    decision = await router.classify(text)
    await router.close()
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
