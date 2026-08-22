#!/usr/bin/env python3
"""
validate_set_d.py — CI task-validation gate for the Set D benchmark.

For every task in set_d_agentic.jsonl:
  1. A reference solution MUST exist in reference_solutions.json (else FAIL).
  2. Build a fresh workspace from the task's shipped files.
  3. Apply the reference solution (file overlays + optional shell commands).
  4. Run the hidden grader (test_cmd). It MUST succeed and print HIDDEN PASS
     where the grader defines one.

If a known-correct solution following the instruction text fails its own
grader, the task is contradictory and this gate fails CI.

Usage: python3 verifier/scripts/validate_set_d.py [--json OUT]
Exit codes: 0 all pass; 1 validation failures; 2 structural errors.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
DATA = Path(REPO) / "verifier" / "data" / "set_d_agentic.jsonl"
REFS = Path(REPO) / "verifier" / "data" / "reference_solutions.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep-workspaces", action="store_true")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    refs = json.loads(REFS.read_text())
    dataset_sha = hashlib.sha256(DATA.read_bytes()).hexdigest()

    results = []
    failures = 0
    for t in tasks:
        tid = t["id"]
        row = {"task": tid, "status": "PASS", "detail": ""}
        ref = refs.get(tid)
        if ref is None:
            row.update(status="FAIL", detail="no reference solution")
            results.append(row)
            failures += 1
            continue

        ws = tempfile.mkdtemp(prefix=f"setd_gate_{tid}_")
        try:
            # 1. ship the task's own files
            for name, content in t.get("files", []):
                p = Path(ws) / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            # 2. apply the reference overlay
            for name, content in ref.get("files", []):
                p = Path(ws) / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            # 3. run reference commands
            for cmd in ref.get("commands", []):
                rc = subprocess.run(["bash", "-c", cmd], cwd=ws, timeout=60,
                                    capture_output=True, text=True)
                if rc.returncode != 0:
                    raise RuntimeError(f"reference command failed: {cmd} :: {rc.stderr[:200]}")
            # 4. hidden grader
            rc = subprocess.run(["bash", "-c", t["test_cmd"]], cwd=ws, timeout=120,
                                capture_output=True, text=True)
            out = rc.stdout + rc.stderr
            expects_marker = "HIDDEN PASS" in t["test_cmd"]
            ok = rc.returncode == 0 and (not expects_marker or "HIDDEN PASS" in out)
            if not ok:
                row.update(status="FAIL",
                           detail=f"grader rc={rc.returncode} out={out.strip()[:180]}")
                failures += 1
        except Exception as e:
            row.update(status="FAIL", detail=str(e)[:200])
            failures += 1
        finally:
            if args.keep_workspaces:
                row["workspace"] = ws

        if ref.get("note") and not any(f.get("note","").startswith("FLAGGED") for f in [ref]):
            row["note"] = ref["note"]
        if ref.get("note", "").startswith("FLAGGED"):
            row["flag"] = ref["note"]
        results.append(row)
        print(f"[{row['status']:4}] {tid}: {row.get('detail') or row.get('flag') or 'ok'}"
              + (f" | {row['note']}" if row.get("note") else ""), flush=True)

    summary = {
        "dataset": str(DATA),
        "dataset_sha256": dataset_sha,
        "tasks": len(tasks),
        "passed": len(results) - failures,
        "failed": failures,
        "flagged_noop_references": [r["task"] for r in results if r.get("flag")],
        "results": results,
    }
    out_path = args.json or str(Path(REPO) / "verifier" / "data" / "set_d_validation.json")
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"\n=== GATE {'PASS' if failures == 0 else 'FAIL'}: "
          f"{len(tasks)-failures}/{len(tasks)} tasks validated ===")
    print(f"dataset sha256: {dataset_sha}")
    if summary["flagged_noop_references"]:
        print("FLAGGED (no-op references need review):", summary["flagged_noop_references"])
    print(f"[+] wrote {out_path}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
