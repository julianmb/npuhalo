#!/usr/bin/env python3
"""analyze_set_d.py — aggregate all Set D results into the decision metrics."""

import json
import os
import random
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
RAW = os.path.join(REPO, "verifier", "results", "raw", "setd")


def load_run(f):
    with open(os.path.join(RAW, f)) as fp:
        return json.load(fp)


def bootstrap(passes, n_boot=2000, ci=0.95, seed=7):
    rng = random.Random(seed)
    n = len(passes)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rates = sorted(sum([passes[rng.randrange(n)] for _ in range(n)]) / n
                   for _ in range(n_boot))
    return (sum(passes) / n, rates[int((1 - ci) / 2 * n_boot)],
            rates[int((1 + ci) / 2 * n_boot) - 1])


def main():
    runs = defaultdict(list)
    for f in sorted(os.listdir(RAW)):
        if f.startswith("summary") or not f.endswith(".json"):
            continue
        try:
            r = load_run(f)
        except Exception:
            continue
        cfg = r.get("config", "?")
        tid = r.get("task_id")
        runs[(cfg, tid)].append(r)

    present = sorted({c for c, _ in runs})

    out = {}
    for cfg in present:
        recs = [r for (c, _), rs in runs.items() if c == cfg for r in rs]
        if not recs:
            continue
        passes = [1 if r["passed"] else 0 for r in recs]
        p, lo, hi = bootstrap(passes)
        # rollback conversion: among tasks with >=1 rollback, fraction passed
        rb_tasks = [r for r in recs if r.get("rollbacks", 0) > 0]
        converted = sum(1 for r in rb_tasks if r["passed"])
        # false aborts: passing tasks with >=1 abort event
        false_aborts = sum(1 for r in recs
                           if r["passed"] and len(r.get("abort_events", [])) > 0)
        out[cfg] = {
            "pass_rate": round(p, 3),
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
            "n": len(recs),
            "rollback_tasks": len(rb_tasks),
            "rollback_converted": converted,
            "rollback_conversion_rate": round(converted / len(rb_tasks), 3) if rb_tasks else None,
            "false_aborts": false_aborts,
            "mean_rollbacks": round(sum(r.get("rollbacks", 0) for r in recs) / len(recs), 2),
            "mean_steps": round(sum(r.get("num_steps", 0) for r in recs) / len(recs), 1),
        }
        # per-task pass matrix
        out[cfg]["per_task"] = {t: sum(1 for r in rs if r["passed"])
                                for (c, t), rs in runs.items() if c == cfg}

    # quality-per-token proxy: passes per step (steps ~ tokens)
    for cfg in present:
        recs = [r for (c, _), rs in runs.items() if c == cfg for r in rs]
        if recs:
            tot_steps = sum(r.get("num_steps", 0) for r in recs)
            tot_pass = sum(1 for r in recs if r["passed"])
            out[cfg]["quality_per_step"] = round(tot_pass / tot_steps, 4) if tot_steps else 0

    with open(os.path.join(REPO, "verifier", "results", "setd_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
