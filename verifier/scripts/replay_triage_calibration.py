#!/usr/bin/env python3
"""
replay_triage_calibration.py — Offline replay of escalation-gate policies on the
FROZEN active-recovery dataset (seeds 1-2 only). No live endpoint calls.

For each candidate policy, computes what the system WOULD have done at every
checkpoint of every active-mode run, using only logged features:
  - majority verdict + vote distribution per checkpoint (step.verdict)
  - recorded judge jeopardy score r where policy A escalated (abort_events)
  - baseline counterpart outcome for harm estimation

Outputs a calibration summary JSON and prints the champion by the
pre-registered ranking (subset-of-A aborts, then largest escalation cut).
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import os  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from triage_policy import TriagePolicy, POLICY_NAMES  # noqa: E402

CATCHABLE = {"D02", "D03", "D04", "D06", "D18", "D21", "D25", "D28"}


def load_frozen(exp_dir: Path):
    runs = defaultdict(dict)
    for p in (exp_dir / "runs").glob("*.json"):
        rec = json.load(open(p))
        runs[(rec["task_id"], rec["seed"])][rec["config"]] = rec
    return runs


def checkpoints_of(rec):
    """Ordered list of checkpoint dicts from an active run record."""
    out = []
    for s in rec.get("steps", []):
        v = s.get("verdict")
        if not v:
            continue
        votes = v.get("votes") or {}
        maj = v.get("verdict")
        out.append({
            "majority": maj,
            "votes": votes,
            "latency_ms": v.get("latency_ms"),
            "escalated": bool(s.get("escalated")),
            "r": None,
        })
    # attach recorded judge scores from abort_events (only places r was stored)
    for ev in rec.get("abort_events", []):
        idx = ev.get("step")
        if isinstance(idx, int) and 1 <= idx <= len(out):
            out[idx - 1]["r"] = ev.get("escalation_score")
            out[idx - 1]["aborted"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=os.path.join(REPO, "verifier", "results",
                                                      "active_recovery_20260821_1040"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)
    runs = load_frozen(exp_dir)

    # ---- measured marginal cost per escalation (least squares on frozen data) ----
    xs, ys = [], []
    for (task, seed), modes in runs.items():
        if "active_esc" not in modes or "baseline" not in modes:
            continue
        a, b = modes["active_esc"], modes["baseline"]
        if a.get("outcome") != "COMPLETED" or b.get("outcome") != "COMPLETED":
            continue
        xs.append(a.get("escalations", 0))
        ys.append((a.get("wall_time_s") or 0) - (b.get("wall_time_s") or 0))
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-9, sum((x - mx) ** 2 for x in xs))
    alpha = my - beta * mx
    print(f"[timing] wall_delta_s ~ {alpha:.1f} + {beta:.2f} * escalations   (n={n})")

    # ---- replay each policy over seeds 1-2 active runs ----
    results = {}
    per_policy_events = {}
    for pname in POLICY_NAMES:
        pol = TriagePolicy(pname)
        agg = {"checkpoints": 0, "suspect_checkpoints": 0, "escalations": 0,
               "aborts": 0, "est_harmful_aborts": 0, "aborted_in_baseline_failed_catchable": 0,
               "npu_latency_ms": 0.0}
        events = []
        for (task, seed), modes in sorted(runs.items()):
            if seed not in (1, 2) or "active_esc" not in modes:
                continue
            rec = modes["active_esc"]
            base_rec = modes.get("baseline", {})
            base_passed = bool(base_rec.get("passed"))
            ckpts = checkpoints_of(rec)
            history = []
            est_aborts_here = []
            for i, c in enumerate(ckpts):
                history.append({"majority": c["majority"], "votes": c["votes"]})
                agg["checkpoints"] += 1
                agg["npu_latency_ms"] += c["latency_ms"] or 0
                if c["majority"] == "SUSPECT":
                    agg["suspect_checkpoints"] += 1
                decision = pol.decide(history)
                would_abort = False
                if decision == "ESCALATE":
                    agg["escalations"] += 1
                    # counterfactual abort only knowable where A already scored the judge
                    if c.get("r") is not None and c["r"] is not None and c["r"] <= 0.50:
                        would_abort = True
                    elif c.get("aborted"):
                        would_abort = True
                    if would_abort:
                        agg["aborts"] += 1
                        est_aborts_here.append(i + 1)
                        if base_passed:
                            agg["est_harmful_aborts"] += 1
                        if task in CATCHABLE and not base_rec.get("passed"):
                            agg["aborted_in_baseline_failed_catchable"] += 1
                events.append({"task": task, "seed": seed, "ckpt": i + 1,
                               "majority": c["majority"], "decision": decision,
                               "would_abort": would_abort})
        n_tasks = sum(1 for (t, s) in runs if s in (1, 2) and "active_esc" in runs[(t, s)])
        agg["escalations_per_task"] = round(agg["escalations"] / max(1, n_tasks), 2)
        agg["suspect_rate_pct"] = round(100 * agg["suspect_checkpoints"] / max(1, agg["checkpoints"]), 1)
        agg["post_policy_escalation_rate_pct"] = round(
            100 * agg["escalations"] / max(1, agg["suspect_checkpoints"]), 1)
        agg["est_latency_saved_s_vs_A"] = None  # filled after A computed
        results[pname] = agg
        per_policy_events[pname] = events

    ref = results["A_reference"]["escalations"]
    for pname, agg in results.items():
        saved_escal = ref - agg["escalations"]
        agg["est_latency_saved_s_vs_A"] = round(saved_escal * beta, 1)
        agg["escalations_cut_vs_A"] = saved_escal

    # champion: abort set subset of A (guaranteed structurally, verify), no added harmful, max cut
    a_harm = results["A_reference"]["est_harmful_aborts"]
    eligible = [p for p in POLICY_NAMES
                if results[p]["est_harmful_aborts"] <= a_harm
                and results[p]["aborts"] <= results["A_reference"]["aborts"]]
    champion = max(eligible, key=lambda p: results[p]["escalations_cut_vs_A"]) if eligible else "A_reference"

    summary = {
        "frozen_dataset": str(exp_dir),
        "seeds_used_for_calibration": [1, 2],
        "escalation_cost_fit": {"alpha_s": round(alpha, 2), "beta_s_per_escalation": round(beta, 2), "n_runs": n},
        "policies": results,
        "champion": champion,
        "champion_rationale": "pre-registered ranking: subset-of-A aborts, then largest escalation cut",
        "replay_limitation": "counterfactual aborts only evaluable where policy A scored the judge; "
                             "B/C can only remove aborts relative to A in replay",
    }
    out = Path(args.out) if args.out else exp_dir / "triage_calibration_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "policies"}, indent=2))
    for pname in POLICY_NAMES:
        print(pname, json.dumps(results[pname]))
    print(f"\n[+] CHAMPION: {champion}")
    print(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
