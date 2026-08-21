#!/usr/bin/env python3
"""
replay_parser_guard.py — Offline replay of every recorded agent step through
the incremental tool-call parser.

Acceptance criterion: ZERO false rejects — every step whose reference
parse_tool_call() succeeded must classify VALID_SO_FAR or RECOVERABLE
(never UNRECOVERABLE). Also re-derives the malformed-call failure mass.
"""

import json
import glob
import os
import re
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from toolcall_parser import IncrementalToolCallParser  # noqa: E402

RUN_DIRS = [
    os.path.join(REPO, "verifier", "results", "active_recovery_20260821_1040", "runs"),
    os.path.join(REPO, "verifier", "results", "active_recovery_calibrated_20260821_1637", "runs"),
]
CHUNK = 4  # chars per synthetic streaming chunk


def stream_chunks(text: str):
    for i in range(0, len(text), CHUNK):
        yield text[i:i + CHUNK]


def main():
    stats = {
        "runs": 0, "steps": 0,
        "ref_parsed": 0, "false_rejects": 0,
        "ref_none": 0, "none_prose_no_marker": 0, "none_truncated_midcall": 0,
        "none_meta_mention": 0, "none_true_malformed": 0,
        "unrecoverable_detections": [],
    }
    for d in RUN_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(f))
            stats["runs"] += 1
            for s in r.get("steps", []):
                txt = s.get("raw_text") or ""
                if not txt:
                    continue
                stats["steps"] += 1
                p = IncrementalToolCallParser()
                first_bad = None
                for ci, ch in enumerate(stream_chunks(txt)):
                    st = p.feed(ch)
                    if st.value == "UNRECOVERABLE" and first_bad is None:
                        first_bad = {"chunk_index": ci, "char_index": ci * CHUNK,
                                     "reason": p.error["reason"]}
                        break
                res = p.result()
                ref_ok = s.get("tool_call") is not None
                if ref_ok:
                    stats["ref_parsed"] += 1
                    if res["state"] == "UNRECOVERABLE":
                        stats["false_rejects"] += 1
                        stats["unrecoverable_detections"].append({
                            "task": r["task_id"], "seed": r["seed"], "config": r["config"],
                            "kind": "FALSE_REJECT", "reason": res["error"]["reason"],
                            "text_head": txt[:200],
                        })
                else:
                    stats["ref_none"] += 1
                    if res["state"] == "UNRECOVERABLE":
                        stats["none_true_malformed"] += 1
                        stats["unrecoverable_detections"].append({
                            "task": r["task_id"], "seed": r["seed"], "config": r["config"],
                            "kind": "TRUE_MALFORMED", "first_bad": first_bad,
                            "total_chars": len(txt),
                            "text_head": txt[:200],
                        })
                    elif re.search(r"(<function[=]?|<tool_call>?)\s*$", txt.rstrip()) or \
                            re.search(r"<function=[A-Za-z]*$", txt.rstrip()):
                        stats["none_truncated_midcall"] += 1
                        stats["unrecoverable_detections"].append({
                            "task": r["task_id"], "seed": r["seed"], "config": r["config"],
                            "kind": "truncated_midcall", "char_index": len(txt),
                            "note": "generation already ended; no tokens saveable",
                        })
                    elif "<tool_call>" in txt or "<function" in txt:
                        stats["none_meta_mention"] += 1
                    else:
                        stats["none_prose_no_marker"] += 1


    stats["rederived_failure_mass"] = {
        "successful_parses": stats["ref_parsed"],
        "missing_call_prose": stats["none_prose_no_marker"],
        "truncated_midcall": stats["none_truncated_midcall"],
        "meta_mention_false_positive_traps": stats["none_meta_mention"],
        "true_malformed_calls": stats["none_true_malformed"],
    }
    out = os.path.join(REPO, "verifier", "results", "parser_guard_replay.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "unrecoverable_detections"}, indent=2))
    print("\n=== detections ===")
    for d in stats["unrecoverable_detections"]:
        print(json.dumps(d)[:220])
    print(f"\n[+] FALSE REJECTS: {stats['false_rejects']} "
          f"({'PASS' if stats['false_rejects'] == 0 else 'FAIL — FIX PARSER'})")
    print(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
