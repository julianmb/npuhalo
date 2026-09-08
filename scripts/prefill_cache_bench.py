#!/usr/bin/env python3
"""
prefill_cache_bench.py — measure prompt-cache / KV-shift reuse TTFT benefit
on repeated identical agent-style requests.

Sends the same long prompt N times (non-streaming, n_predict=1) and reports
wall-clock ms per request. With --cache-prompt --cache-reuse enabled, request 2+
should hit the KV cache and skip most of prefill. With --no-cache-prompt,
every request prefill-pattern fully.

Usage:
  python3 scripts/prefill_cache_bench.py http://127.0.0.1:8099 --runs 4
"""
import json
import statistics
import sys
import time
import urllib.request

def build_prompt(target_tokens: int = 2000) -> str:
    """A realistic long agent system prompt."""
    base = ("You are a senior systems engineer assistant. You have access to a "
            "repository checkout and you must reason "
            "carefully about build commands, kernel flags, memory bandwidth, and "
            "speculative decoding configurations. Always prefer concrete measurements "
            "over heuristics. When asked about Apple M-series chips, answer honestly "
            "with the latest known benchmarks. Maintain a strict JSON output schema "
            "for every tool call. ")
    prompt = base
    while len(prompt.split()) < target_tokens:
        prompt += base
    return prompt

def post(url: str, payload: dict, timeout: int = 300) -> tuple[float, dict]:
    req = urllib.request.Request(url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return (time.perf_counter() - t0) * 1000, body

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cache_prompt = "--cache-on" in sys.argv  # advisory, matched server-side

    prompt = build_prompt()
    n_prompt_tokens_est = len(prompt.split())
    print(f"prompt ~{n_prompt_tokens_est} words, runs={runs}, cache_server={'on' if cache_prompt else 'off'}")

    url = base_url.rstrip("/") + "/completion"
    payload = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": cache_prompt,
    }
    results = []
    for i in range(runs):
        ms, body = post(url, payload)
        tok = body.get("prompt_tokens") or body.get("usage", {}).get("prompt_tokens", 0)
        tps = tok / (ms / 1000) if ms > 0 and tok else 0
        results.append(ms)
        print(f"  run{i+1}: {ms:8.0f} ms   (~{tps:7.1f} tok/s prefill, {tok} prompt tok)")

    cold = results[0]
    warm = results[1:]
    print(f"\ncold: {cold:.0f} ms")
    if warm:
        print(f"warm median: {statistics.median(warm):.0f} ms")
        if cold > 0:
            print(f"speedup on repeat: {cold / statistics.median(warm):.2f}x")

if __name__ == "__main__":
    main()
