# Active Recovery Experiment Report

**Experiment dir:** `verifier/results/active_recovery_20260821_1040/`
**Date:** 2026-08-21 · **Commit:** `8470d11` · **Manifest:** `manifest.json` (locked, dataset sha256 `5e816fe5dcffd859…`)
**Decision question:** *Can the shadow-mode verifier signal (4/4 catchable detection, 0/22 false alarms) be converted into a safe, measurable end-to-end agent improvement when allowed to abort and recover?*

## 1. Did the complete active run finish?

**Yes — 64/64 task/mode evaluations completed** (32 comparable pairs: 16 tasks × 2 modes × 2 seeds; 2 pairs were the Phase-4 smoke runs, 62 from the main run).
- `INFRA_FAILURE`: **0** · `CONTEXT_OVERFLOW`: **0**
- The resilient harness (health-check → verified restart-wait → single retry) never needed to invoke recovery during the run; zero server crashes occurred under this workload. All retry/restart machinery was validated in Phase-4 smoke tests instead.

## 2. Paired results (comparable pairs only, n=32)

| Metric | Result |
|---|---|
| Baseline success | **24/32 (75.0%)** |
| Active success | **20/32 (62.5%)** |
| Δ successful tasks | **−4** |
| Conversion wins (base F → act P) | **1** (D25 s1 — but with **0 rollbacks**, i.e. not attributable to recovery) |
| Conversion losses (base P → act F) | **5** (2 attributable to abort intervention: D21 s1, D28 s2; 3 had zero rollbacks — stochastic + overhead) |
| Suspected false aborts | **2 / 9 interventions** (both harmful: baseline passed, active aborted then failed) |
| Catchable recall→conversion | 1 of 8 baseline-failed catchable pairs "converted" — and that one had no rollback, so **attributable recovery wins: 0/8** |
| Mean latency Δ | **+11.6 s** active (61.9 → 73.5 s, +19%) |
| Median latency Δ | +4.5 s · p95 Δ +59.2 s (n=32 ≥ 20, reported) |
| Infra failure rate | baseline 0/N, active 0/N |

Statistical note: discordant pairs 6 (1 win, 5 losses); exact binomial p ≈ 0.11 — directionally negative, not individually significant at n=32.

## 3. Per-task autopsies (every outcome change)

| Task/seed | Change | Verifier path taken | Did recovery change the result? | Was the verifier right? |
|---|---|---|---|---|
| D03 s1 | P→F | 4 escalations, **0 rollbacks** — judge never aborted | No — trajectory identical in kind; failed by step budget with verifier latency overhead (+124 s) | No intervention occurred; NPU SUSPECT votes were step-level noise |
| D03 s2 | P→F | 4 escalations, 0 rollbacks | No — same as above | Same |
| D18 s2 | P→F | 10 escalations, 0 rollbacks | No — overhead only (+40 s) | Same |
| D21 s1 | P→F | **2 rollbacks**, 11 escalations | **Yes — harm.** Rollbacks consumed budget; task died by exhaustion (17 steps) | Detection dubious: most flagged steps had exit 0; looks like false-positive triage pressure, then bad recovery |
| D25 s1 | F→P | 4 escalations, **0 rollbacks** | **No — not a recovery win.** Pure stochastic variation; verifier did not alter the path | n/a |
| D28 s2 | P→F | **2 rollbacks**, 13 escalations | **Yes — harm.** 21.9 s → 135.8 s; rollbacks derailed a passing trajectory | Same dubious pattern: SUSPECT on healthy steps, rollback, budget death |

## 4. Separated conclusions (no cross-category leakage)

- **Verifier classification quality:** unchanged from shadow evidence — NPU triage is sensitive but step-level SUSPECT-happy (SUSPECT on exit-0 steps); judge escalates often, aborts rarely.
- **Rollback/recovery correctness:** **poor.** 9 interventions: 2 harmed passing trajectories, 7 neutral, 0 attributable recoveries. Recovery-as-implemented (checkpoint rewind + re-run within a fixed step budget) converts detections into budget exhaustion rather than passes.
- **Serving reliability:** excellent once hardened — 64/64 completed, 0 infra failures. The original sweep's death was harness fragility, now fixed and validated.
- **End-to-end agent value:** **negative point estimate** (−12.5 pp success, +19% mean latency). Not statistically significant at n=32, but the entire delta comes from the active arm, and every attributable intervention effect was harmful or neutral.

## 5. Recommendation

> **SHADOW ONLY: signal is useful but recovery value remains unproven**

The shadow-mode detection signal remains real and worth logging. But in this paired trial, allowing abort-and-recover produced **zero attributable recoveries and two attributable harms**, plus a +19% latency tax. If the 2/9 harmful-intervention rate replicates in any larger sample, the correct verdict becomes ARCHIVE. Do not enable active aborts on real workloads on this evidence.

Acceptance criteria check: harness lost/changed no tasks ✅ · every pair has raw event records (`events.jsonl`, 64 run artifacts) ✅ · controls + known-catchable trajectories included ✅ · locked identical manifest/settings both modes ✅ · failure/retry behavior separately measured (0 events this run; machinery smoke-tested) ✅ · conclusion from completed comparable pairs only ✅ · no threshold/prompt/model/tool changes during comparison ✅
