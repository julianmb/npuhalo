#!/usr/bin/env python3
"""
sanitize_telemetry.py — Build the sanitized raw-telemetry release archive.

Walks verifier/results/raw/, redacts machine-specific paths and ownership
details from every JSON record, writes a tar.gz with checksums, and prints a
manifest summary. The output is intended to be attached to a GitHub release.

Usage: python3 scripts/sanitize_telemetry.py [--out DIR]
"""

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW_DIR = REPO / "verifier" / "results" / "raw"

REDACTIONS = [
    (re.compile(r"/tmp/setd_[A-Za-z0-9_]+"), "/tmp/setd_<redacted>"),
    (re.compile(r"/tmp/[A-Za-z0-9_.-]{6,}"), "/tmp/<redacted>"),
    (re.compile(r"\buser user\b"), "<user> <group>"),
    (re.compile(r"root root"), "<user> <group>"),
]


def redact_text(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def redact_json(obj):
    if isinstance(obj, dict):
        return {k: redact_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_json(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "dist"))
    args = parser.parse_args()
    if not RAW_DIR.is_dir():
        print(f"error: {RAW_DIR} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"files": [], "redaction_rules": [p.pattern for p, _ in REDACTIONS]}
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "npuhalo-telemetry"
        for src in sorted(RAW_DIR.rglob("*.json")):
            rel = src.relative_to(RAW_DIR)
            dst = staged / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = json.loads(src.read_text())
                cleaned = json.dumps(redact_json(data), indent=2)
            except json.JSONDecodeError:
                cleaned = redact_text(src.read_text())
            dst.write_text(cleaned)
            manifest["files"].append({"path": str(rel), "sha256": sha256(dst)})

        manifest_path = staged / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        archive = out_dir / "npuhalo-v0.1.0-telemetry.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staged, arcname="npuhalo-telemetry")

    sums = out_dir / "npuhalo-v0.1.0-telemetry.tar.gz.sha256"
    sums.write_text(f"{sha256(archive)}  {archive.name}\n")
    print(f"[+] {archive} ({archive.stat().st_size / 1024:.0f} KiB, {len(manifest['files'])} files)")
    print(f"[+] {sums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
