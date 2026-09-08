#!/usr/bin/env python3
"""
sanitize_history_paths.py — Redact machine-specific paths from tracked files.

Used during release preparation (v1.0.0-findings) to strip personal /
machine-specific information from experiment records:

  /home/<user>/...            -> <home>/...
  <redacted-windows>/Users/...      -> <redacted-windows>...
  /tmp/setd_<suffix>          -> /tmp/setd_<redacted>

Idempotent. Run from repo root: python3 scripts/sanitize_history_paths.py
"""

import subprocess
import sys

RULES = [
    ("<redacted-windows>/", "<redacted-windows>/"),
    ("<home>/", "<home>/"),
]


def redact(text: str) -> str:
    # longest-first so <redacted-windows>/Users/... collapses before generic rules
    text = text.replace("<redacted-windows>/", "<redacted-windows>/")
    text = text.replace("<redacted-windows>/", "<redacted-windows>/")
    text = text.replace("<home>/", "<home>/")
    text = text.replace("<home>", "<home>")  # bare form in emitted commands
    return text


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    changed = []
    for f in files:
        if not f.endswith((".json", ".jsonl", ".md", ".py", ".txt", ".yml", ".yaml",
                           ".toml", ".cfg", ".ini", ".sh", ".env")):
            continue
        try:
            original = open(f, encoding="utf-8", errors="surrogateescape").read()
        except (IsADirectoryError, FileNotFoundError):
            continue
        cleaned = redact(original)
        if cleaned != original:
            open(f, "w", encoding="utf-8", errors="surrogateescape").write(cleaned)
            changed.append(f)
    print(f"sanitized {len(changed)} files")
    for f in changed:
        print(" ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
