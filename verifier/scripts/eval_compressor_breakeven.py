#!/usr/bin/env python3
"""
eval_compressor_breakeven.py — Compressor end-to-end breakeven measurement.

Closes README caveat #9 ("Compressor size reduction is measured; end-to-end
time savings are not").

For realistic tool outputs of increasing size:
  1. Measure GPU prefill wall time with the RAW output in context (max_tokens=1
     isolates prefill).
  2. Compress via the NPU (structured extraction, same prompt as
     run_compressor.py), measuring compression latency.
  3. Measure GPU prefill with the COMPRESSED output.

net_saving(size) = (prefill_raw - prefill_compressed) - compression_latency
The breakeven size is where net saving crosses zero.

Usage: python3 verifier/scripts/eval_compressor_breakeven.py [--out ...]
"""

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

GEN_URL = os.environ.get("CB_GEN", "http://127.0.0.1:8012/v1/chat/completions")
NPU_URL = os.environ.get("CB_NPU", "http://127.0.0.1:8001/v1/chat/completions")
NPU_MODEL = os.environ.get("CB_NPU_MODEL", "lfm2.5-tk:1.2b")

EXTRACT_PROMPT = """Extract a compact structured summary from this tool output.
Return ONLY these fields, one per line, no prose:
COMMAND: <the command or tool name>
EXIT_CODE: <numeric exit code or N/A>
ERROR_LINES: <first error/traceback line or NONE>
KEY_PATHS: <comma-separated file paths mentioned or NONE>

Tool output:
{output}"""

SIZES_CHARS = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
TRIALS = 3


def make_tool_output(size_chars: int, seed: int) -> str:
    """Realistic pytest-style failure output, grown to the requested size."""
    block = (
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0\n"
        "rootdir: /workspace/project\n"
        "collected 128 items\n\n"
        "tests/test_api.py ....F.. [ 12%]\n"
        "tests/test_auth.py ..F....... [ 25%]\n"
        "tests/test_db.py F.F........ [ 37%]\n"
        f"________________________________ test_upload_chunk_{seed} ________________________________\n\n"
        "    def test_upload_chunk():\n"
        ">       assert uploader.send(chunk) == 201\n"
        "E       assert 500 == 201\n"
        "E        +  where 500 = <Uploader object at 0x7f3a2b8c>.send(b'\\x00\\x01...')\n\n"
        "uploader.py:42: AssertionError\n"
        "----------------------------- Captured stderr call -----------------------------\n"
        "Traceback (most recent call last):\n"
        "  File 'uploader.py', line 42, in send\n"
        "    raise StorageError('chunk write failed: no space left on device')\n"
        "StorageError: chunk write failed: no space left on device\n"
        "  File 'storage.py', line 118, in write_chunk\n"
        "    os.write(self.fd, data)\n"
        "  File 'storage.py', line 77, in _flush\n"
        "    self._rotate_log()\n"
    )
    filler = ("DEBUG storage.pool: acquired lease id=%d ttl=30s\n" % seed * 400)
    out = (block + filler) * (size_chars // (len(block) + len(filler)) + 1)
    return out[:size_chars]


def chat(url: str, payload: dict, timeout: int = 120, retries: int = 3) -> tuple[str, float]:
    body = json.dumps(payload).encode()
    last_exc = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            dt = (time.time() - t0) * 1000
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            return (msg.get("content") or "").strip(), dt
        except Exception as exc:  # transient server drops under rapid prefill load
            last_exc = exc
            print(f"  [retry {attempt + 1}/{retries}] {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise last_exc


def prefill_ms(text: str) -> float:
    """Wall time for generator to ingest `text` with max_tokens=1 (prefill only)."""
    _, dt = chat(GEN_URL, {
        "messages": [
            {"role": "system", "content": "You are a coding agent. Tool output follows."},
            {"role": "user", "content": text},
        ],
        "max_tokens": 1, "temperature": 0,
    })
    return dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(REPO, "verifier", "results", "compressor_breakeven.json"))
    args = parser.parse_args()

    # Warm both endpoints (NPU cold start ~74s on first call after load)
    print("[warm] warming NPU and GPU endpoints...", flush=True)
    chat(NPU_URL, {"model": NPU_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    prefill_ms("warmup request")

    rows = []
    for size in SIZES_CHARS:
        raw_prefills, comp_prefills, comp_lats, ratios = [], [], [], []
        for trial in range(TRIALS):
            output = make_tool_output(size, seed=trial)
            p_raw = prefill_ms(output)

            comp, t_comp = chat(NPU_URL, {
                "model": NPU_MODEL,
                "messages": [{"role": "user", "content": EXTRACT_PROMPT.format(output=output[:6000])}],
                "max_tokens": 160, "temperature": 0.0,
            })
            p_comp = prefill_ms(comp)
            raw_prefills.append(p_raw)
            comp_prefills.append(p_comp)
            comp_lats.append(t_comp)
            ratios.append(len(comp) / max(1, len(output)))
            print(f"[{size:>6} ch t{trial}] prefill raw={p_raw:7.0f}ms comp={p_comp:7.0f}ms "
                  f"compress={t_comp:6.0f}ms ratio={ratios[-1]*100:4.1f}%", flush=True)

        p_raw = sorted(raw_prefills)[len(raw_prefills) // 2]
        p_comp = sorted(comp_prefills)[len(comp_prefills) // 2]
        t_comp = sorted(comp_lats)[len(comp_lats) // 2]
        net = (p_raw - p_comp) - t_comp
        rows.append({
            "size_chars": size,
            "prefill_raw_ms": round(p_raw),
            "prefill_compressed_ms": round(p_comp),
            "compression_latency_ms": round(t_comp),
            "mean_size_ratio": round(sum(ratios) / len(ratios), 3),
            "net_saving_ms": round(net),
        })
        print(f"[{size:>6} ch] NET SAVING: {net:+.0f} ms", flush=True)

    breakeven = next((r["size_chars"] for r in rows if r["net_saving_ms"] > 0), None)
    result = {
        "breakeven_chars": breakeven,
        "verdict": (
            f"Compression pays off end-to-end above ~{breakeven} chars" if breakeven
            else "Compression never paid off within tested sizes (up to %d chars)" % SIZES_CHARS[-1]
        ),
        "rows": rows,
        "caveats": [
            "Prefill isolated via max_tokens=1; decode-phase effects not modeled.",
            "Median of 3 trials per size; single-seed synthetic outputs.",
            "Generator KV/prompt-cache state may favor later trials; sizes interleaved identically per trial.",
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
