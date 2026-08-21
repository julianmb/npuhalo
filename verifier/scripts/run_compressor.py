#!/usr/bin/env python3
"""
run_compressor.py — Phase 6 (conditional): NPU as context compressor.

If active verification fails to beat the 15% quality-per-token threshold, the
alternative NPU role is: when any tool output exceeds 500 tokens, route it
through LFM2.5 on the NPU for structured extraction (command, exit code,
error lines, key paths) BEFORE appending to the generator context.

Measures: context tokens saved per task, pass-rate delta vs full-output
baseline, added latency.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from typing import Dict
from verifier_client import NPUVerifierClient  # noqa: E402

NPU = "http://127.0.0.1:8001/v1"
OUT_DIR = os.path.join(REPO, "verifier", "results", "compressor")

EXTRACT_PROMPT = """Extract a compact structured summary from this tool output.
Return ONLY these fields, one per line, no prose:
COMMAND: <the command or tool name>
EXIT_CODE: <numeric exit code or N/A>
ERROR_LINES: <first error/traceback line or NONE>
KEY_PATHS: <comma-separated file paths mentioned or NONE>

Tool output:
{output}"""


def compress(client, output: str) -> Dict:
    t0 = time.time()
    res = client.evaluate_checkpoint(
        task_prompt="COMPRESS",
        trajectory=EXTRACT_PROMPT.format(output=output[:6000]),
        prompt_template="{task}\n\n{trajectory}",
        k_samples=1, temperature=0.0,
    )
    dt = time.time() - t0
    raw = (res.get("raw_responses") or [""])[0] if isinstance(res.get("raw_responses"), list) else ""
    return {
        "compressed": raw,
        "orig_chars": len(output),
        "compressed_chars": len(raw),
        "latency_ms": dt * 1000,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = NPUVerifierClient(NPU)
    client.evaluate_checkpoint("w", "w", "{task} {trajectory}", k_samples=1, temperature=0.0)

    samples = []
    for f in sorted(os.listdir(os.path.join(REPO, "verifier", "results", "raw", "setd"))):
        if not f.endswith(".json") or f.startswith("summary"):
            continue
        try:
            d = json.load(open(os.path.join(REPO, "verifier", "results", "raw", "setd", f)))
        except Exception:
            continue
        for s in d.get("steps", []):
            tr = s.get("tool_result") or {}
            out = (tr.get("stdout") or "") + "\n" + (tr.get("stderr") or "")
            if len(out) > 500:
                r = compress(client, out)
                r["task"] = d.get("task_id")
                r["step"] = s.get("step")
                samples.append(r)
                print(f"[{d.get('task_id')}:{s.get('step')}] {len(out)} -> {len(r['compressed'])} chars "
                      f"({100*len(r['compressed'])/max(1,len(out)):.0f}%) latency={r['latency_ms']:.0f}ms", flush=True)

    if samples:
        summary = {
            "n": len(samples),
            "mean_saved_ratio": sum(s["compressed_chars"] / max(1, s["orig_chars"]) for s in samples) / len(samples),
            "mean_latency_ms": sum(s["latency_ms"] for s in samples) / len(samples),
        }
    else:
        summary = {"n": 0}
    with open(os.path.join(OUT_DIR, "compressor_summary.json"), "w") as f:
        json.dump({"summary": summary, "samples": samples}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
