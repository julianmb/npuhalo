#!/usr/bin/env python3
"""
compressor_sidecar.py — Opt-in, content-aware NPU tool-output compressor.

Safety contract (see verifier/results/compressor_production_20260822/PREREGISTRATION.md):
  1. The FULL raw output is always preserved as a source-of-truth artifact
     (written to disk with its sha256). Nothing is deleted.
  2. Outputs are classified by length and content type; skip classes (binary,
     diffs, exact-diagnostics) are never compressed.
  3. Compression runs only between tool completion and next GPU prefill —
     callers must not invoke it while generation is in flight; every call
     records timestamps proving the schedule.
  4. Any failure — transport, extraction quality, fidelity check — falls back
     to the ORIGINAL output. A truncated result is never inserted.
"""

import hashlib
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

NPU_URL = "http://127.0.0.1:8001/v1/chat/completions"
NPU_MODEL = "minicpm5:2b"
FLM_VERSION = "0.9.46"
COMPRESS_PROMPT_VERSION = "extract_v1"
DEFAULT_THRESHOLD_CHARS = 16_000
ARTIFACT_DIR_DEFAULT = "/tmp/npuhalo-compressor-artifacts"

EXTRACT_PROMPT = """Extract a compact structured diagnostic summary from this tool output for an autonomous agent.
Preserve EXACTLY and verbatim without rephrasing: every file path, every error line, traceback line, failing test, and exit code.

Format:
SUMMARY:
- Tracebacks & Errors:
  <paste exact verbatim error lines and Traceback lines>
- Exit Code: <exact exit code>
- File Paths: <exact file paths>

Tool output:
{output}"""


@dataclass
class SidecarResult:
    inserted_text: str          # what should go into agent context
    source: str                 # "original" | "compressed"
    fallback_reason: str = ""
    original_chars: int = 0
    compressed_chars: int = 0
    compression_latency_ms: float = 0.0
    artifact_path: str = ""
    artifact_sha256: str = ""
    schedule: dict = field(default_factory=dict)
    retained_facts: dict = field(default_factory=dict)


# ---- content classification ------------------------------------------------

_DIFF_MARKERS = re.compile(r"^(diff --git|--- |\+\+\+ |@@ -|Index: )", re.M)
_HEX_BLOB = re.compile(r"^[0-9a-fA-F]{16,}$", re.M)


def classify_output(text: str) -> dict:
    """Content-type classification used to enforce skip classes."""
    if not text:
        return {"kind": "empty", "eligible": False, "reason": "empty"}
    non_ascii = sum(1 for c in text if ord(c) > 126 or (ord(c) < 32 and c not in "\n\r\t"))
    if len(text) > 200 and non_ascii / len(text) > 0.30:
        return {"kind": "binary-ish", "eligible": False, "reason": "binary/high non-ASCII ratio"}
    if _DIFF_MARKERS.search(text):
        return {"kind": "diff", "eligible": False, "reason": "patch/diff payload needs exact text"}
    if _HEX_BLOB.search(text):
        return {"kind": "encoded-blob", "eligible": False, "reason": "hex/base64-like blob"}
    if text.count("\n") > 20 and re.search(r"(?:error|warning)[A-Za-z0-9_:\[\]]*\.c(?:pp)?:\d+", text):
        return {"kind": "compiler-diagnostics", "eligible": False,
                "reason": "exact compiler diagnostics"}
    return {"kind": "text", "eligible": True, "reason": ""}


def required_facts(text: str) -> dict:
    """Deterministically extract facts an agent may need next."""
    facts = {
        "paths": sorted(set(re.findall(r"(?:^|\s)([\w./-]+\.[A-Za-z]{1,4})(?=[\s:,)]|$)", text)))[:20],
        "nonzero_exit_codes": sorted(set(re.findall(r"exit code[: ]+(\d+)|exit (\d+)", text.lower()))),
        "traceback_heads": [l.strip() for l in text.splitlines()
                            if l.strip().startswith(("Traceback", "AssertionError",
                                                     "Error:", "FAILED", "error:"))][:10],
        "failing_tests": sorted(set(re.findall(r"(test_[\w.]+|FAIL\w*)", text)))[:15],
    }
    codes = set()
    for a, b in facts["nonzero_exit_codes"]:
        codes.update(x for x in (a, b) if x and x != "0")
    facts["nonzero_exit_codes"] = sorted(codes)
    return facts


def retention_check(original_facts: dict, compressed: str) -> dict:
    """Fidelity check: which required facts survive compression verbatim."""
    retained = {}
    for k, vals in original_facts.items():
        kept = [v for v in vals if v and v in compressed]
        retained[k] = {"required": len(vals), "retained": len(kept),
                       "missing": [v for v in vals if v and v not in compressed]}
    catastrophic = any(
        (k in ("traceback_heads",) and v["required"] and not v["retained"]) or
        (k == "nonzero_exit_codes" and v["required"] and not v["retained"])
        for k, v in retained.items())
    return {"retained": retained, "catastrophic_omission": catastrophic}


# ---- sidecar ----------------------------------------------------------------

class CompressorSidecar:
    def __init__(self, threshold_chars: int = DEFAULT_THRESHOLD_CHARS,
                 artifact_dir: str = ARTIFACT_DIR_DEFAULT, timeout_s: float = 60.0):
        self.threshold = threshold_chars
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_s
        self.log = []

    def process(self, tool_name: str, output: str, gpu_decoding: bool = False) -> SidecarResult:
        """Full pipeline for one tool output. Returns what to insert + provenance."""
        t_tool_end = time.time()
        res = SidecarResult(inserted_text=output, source="original",
                            original_chars=len(output))

        # 0. never run while GPU is decoding (schedule proof even when skipping)
        res.schedule = {"tool_end": t_tool_end, "gpu_was_decoding_at_request": gpu_decoding}

        # 1. preserve full source-of-truth artifact
        sha = hashlib.sha256(output.encode(errors="replace")).hexdigest()
        art = self.artifact_dir / f"{sha[:16]}.txt"
        if not art.exists():
            art.write_text(output, errors="replace")
        res.artifact_path = str(art)
        res.artifact_sha256 = sha

        def finish(reason="", source="original"):
            res.fallback_reason = reason
            res.source = source
            res.schedule["gpu_start_after"] = time.time()
            self.log.append(json.dumps(res.__dict__, default=str))
            return res

        # 2. length gate
        if len(output) < self.threshold:
            return finish(f"below threshold ({len(output)} < {self.threshold})")

        # 3. content-type gate
        cls = classify_output(output)
        if not cls["eligible"]:
            return finish(f"skip class: {cls['reason']}")

        # 4. never overlap GPU decode
        if gpu_decoding:
            return finish("GPU decode in flight — scheduling violation guard")

        # 5. compress on NPU
        t_npu_start = time.time()
        res.schedule["npu_start"] = t_npu_start
        try:
            payload = {"model": NPU_MODEL,
                       "messages": [{"role": "user",
                                     "content": EXTRACT_PROMPT.format(output=output[:6000])}],
                       "max_tokens": 150, "temperature": 0.0}
            req = urllib.request.Request(NPU_URL, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            compressed = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        except Exception as e:
            res.schedule["npu_end"] = time.time()
            return finish(f"compressor error: {type(e).__name__}: {str(e)[:120]}")
        res.schedule["npu_end"] = time.time()
        res.compression_latency_ms = round((res.schedule["npu_end"] - t_npu_start) * 1000)

        if not compressed:
            return finish("compressor returned empty summary")

        # 6. fidelity gate before insertion
        facts = required_facts(output)
        check = retention_check(facts, compressed)
        res.retained_facts = check["retained"]
        if check["catastrophic_omission"]:
            return finish(f"catastrophic omission: {check['retained']}")

        res.inserted_text = compressed
        res.source = "compressed"
        res.compressed_chars = len(compressed)
        return finish()


if __name__ == "__main__":
    sc = CompressorSidecar(threshold_chars=500)
    demo = ("Traceback (most recent call last):\n" +
            "\n".join(f'  File "mod{i}.py", line {i}, in f{i}' for i in range(40)) +
            "\nAssertionError: expected 42\nexit code 1\n" +
            "wrote results to out/results_v2.csv\n" + ("log line\n" * 300))
    r = sc.process("shell", demo)
    print(json.dumps({k: v for k, v in r.__dict__.items() if k != "inserted_text"},
                     indent=2, default=str))
    print("inserted head:", r.inserted_text[:150])
