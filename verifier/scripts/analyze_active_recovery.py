#!/usr/bin/env python3
"""
analyze_active_recovery.py — Paired-analysis for the active-recovery experiment.

Consumes runs/*.json from an experiment directory and produces summary.json
with paired statistics only from comparable (both-modes-COMPLETED) pairs.

Usage: python3 verifier/scripts/analyze_active_recovery.py --exp-dir DIR
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

CATCHABLE = {"D02", "D03", "D04", "D06", "D18", "D21", "D25", "D28"}


def load_runs(exp_dir: Path):
    runs = defaultdict(dict)  # (task, seed) -> {mode: record}
    for p in sorted((exp_dir / "runs").glob("*.json")):
        try:
            rec = json.load(open(p))
        except Exception:
            continue
        key = (rec.get("task_id"), rec.get("seed"))
        runs[key][rec.get("config")] = rec
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)
    runs = load_runs(exp_dir)

    pairs, infra = [], {"baseline": [0, 0], "active_esc": [0, 0]}
    overflow = {"baseline": [0, 0], "active_esc": [0, 0]}
    for (task, seed), modes in sorted(runs.items()):
        for mode, rec in modes.items():
            if rec.get("outcome") == "INFRA_FAILURE":
                infra[mode][1] += 1
            elif rec.get("outcome") == "CONTEXT_OVERFLOW":
                overflow[mode][1] += 1
        if "baseline" in modes and "active_esc" in modes:
            b, a = modes["baseline"], modes["active_esc"]
            if b.get("outcome") == "COMPLETED" and a.get("outcome") == "COMPLETED":
                pairs.append({
                    "task": task, "seed": seed,
                    "catchable": task in CATCHABLE,
                    "baseline_passed": bool(b.get("passed")),
                    "active_passed": bool(a.get("passed")),
                    "delta_pass": int(bool(a.get("passed"))) - int(bool(b.get("passed"))),
                    "active_rollbacks": a.get("rollbacks", 0),
                    "active_escalations": a.get("escalations", 0),
                    "baseline_steps": b.get("num_steps"),
                    "active_steps": a.get("num_steps"),
                    "latency_baseline_s": b.get("wall_time_s"),
                    "latency_active_s": a.get("wall_time_s"),
                    "latency_delta_s": round((a.get("wall_time_s") or 0) - (b.get("wall_time_s") or 0), 1),
                    # suspected false abort: verifier intervened, active failed, baseline passed
                    "suspected_false_abort": bool(
                        (a.get("rollbacks", 0) > 0) and not a.get("passed") and b.get("passed")),
                    # conversion: baseline failed, active passed
                    "conversion_win": (not b.get("passed")) and bool(a.get("passed")),
                    # harm: baseline passed, active failed
                    "conversion_loss": bool(b.get("passed")) and not a.get("passed"),
                })

    n = len(pairs)
    b_pass = sum(1 for p in pairs if p["baseline_passed"])
    a_pass = sum(1 for p in pairs if p["active_passed"])
    deltas = sorted(p["latency_delta_s"] for p in pairs)
    lat_b = [p["latency_baseline_s"] for p in pairs if p["latency_baseline_s"]]
    lat_a = [p["latency_active_s"] for p in pairs if p["latency_active_s"]]

    def pct(num, den):
        return round(100 * num / den, 1) if den else None

    summary = {
        "exp_dir": str(exp_dir),
        "completed_pairs_comparable": n,
        "baseline_success": {"n": b_pass, "N": n, "rate_pct": pct(b_pass, n)},
        "active_success": {"n": a_pass, "N": n, "rate_pct": pct(a_pass, n)},
        "delta_successful_tasks": a_pass - b_pass,
        "conversion_wins": sum(1 for p in pairs if p["conversion_win"]),
        "conversion_losses": sum(1 for p in pairs if p["conversion_loss"]),
        "false_aborts_suspected": {
            "n": sum(1 for p in pairs if p["suspected_false_abort"]),
            "denominator_note": "pairs where active intervened (rollbacks>0)",
            "interventions": sum(1 for p in pairs if p["active_rollbacks"] > 0),
        },
        "catchable_recall_conversion": {
            "note": "of comparable catchable pairs where baseline failed",
            "baseline_failed_catchable_pairs": sum(
                1 for p in pairs if p["catchable"] and not p["baseline_passed"]),
            "converted_by_active": sum(
                1 for p in pairs if p["catchable"] and not p["baseline_passed"] and p["active_passed"]),
        },
        "latency": {
            "mean_baseline_s": round(statistics.mean(lat_b), 1) if lat_b else None,
            "mean_active_s": round(statistics.mean(lat_a), 1) if lat_a else None,
            "mean_delta_s": round(statistics.mean(deltas), 1) if deltas else None,
            "median_delta_s": round(statistics.median(deltas), 1) if deltas else None,
            "p95_delta_s": round(deltas[int(0.95 * len(deltas)) - 1], 1) if len(deltas) >= 20 else None,
            "p95_note": "p95 requires n>=20 pairs",
        },
        "infra_failure_rate": {
            "baseline": {"n": infra["baseline"][0], "attempts_seen": infra["baseline"][1]},
            "active_esc": {"n": infra["active_esc"][0], "attempts_seen": infra["active_esc"][1]},
        },
        "context_overflow_rate": {
            "baseline": {"n": overflow["baseline"][0]},
            "active_esc": {"n": overflow["active_esc"][0]},
            "note": "overflow is a harness/context-limit class, excluded from outcome deltas",
        },
        "outcome_changes": [p for p in pairs if p["delta_pass"] != 0],
        "all_pairs": pairs,
    }
    out = exp_dir / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    brief = {k: v for k, v in summary.items() if k not in ("all_pairs", "outcome_changes")}
    print(json.dumps(brief, indent=2))
    print(f"\n[+] wrote {out} ({len(summary['outcome_changes'])} outcome changes)")


if __name__ == "__main__":
    main()
