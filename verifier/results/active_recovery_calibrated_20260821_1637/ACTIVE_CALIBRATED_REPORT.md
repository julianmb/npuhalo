# Active Calibrated Recovery Report — Frozen Policy B_consecutive

**Experiment dir:** `verifier/results/active_recovery_calibrated_20260821_1637/`
**Date:** 2026-08-21 · **Commit:** `4f426f4` · **Policy sha256:** `16a215f2…92940`
**Held-out data:** seeds 3–5 only (never used for tuning) · **Calibration:** seeds 1–2 frozen dataset (`active_recovery_20260821_1040/`)

## 1. Calibration phase (seeds 1–2 replay, offline)

Replaying the frozen 32-pair dataset against candidate escalation gates:

| Policy | Escalations (/174 ckpts) | Abort set vs A | Est. harmful | Verdict |
|---|---:|---|---:|---|
| A_reference | 174 (100%) | — | 6 ckpt-level | reference |
| B_consecutive | 142 (−18%) | **identical** (12, incl. all 6 catchable-failure aborts) | 6 (unchanged) | **CHAMPION** |
| C_unanimous | 0 (−100%) | empty — disarms verifier | 0 | rejected by AMENDMENT 1 (recall preservation) |

Key calibration facts: NPU majority-SUSPECT rate was **100%** (174/174); zero
unanimous checkpoints existed; measured marginal cost per escalation
**0.94 s**. Champion B_consecutive = escalate only on ≥2 consecutive
majority-SUSPECT checkpoints; isolated SUSPECT → ABSTAIN (snapshot advances as
CONTINUE); judge remains sole abort authority.

## 2. Held-out results (seeds 3–5, live, 48 comparable pairs)

| Metric | Baseline | Active calibrated |
|---|---|---|
| Success rate | **31/48 (64.6%)** | **32/48 (66.7%)** |
| Infra failures | 0 | 0 |
| Mean latency | 71.0 s | 79.9 s (**Δ +9.0 s**, median +3.6 s, p95 +69.1 s) |

Δ successful tasks: **+1** (parity within ±1 ✓).

Gate behavior: 259 checkpoints, SUSPECT rate still **100%**; gate escalated
**211** (4.40/run vs 5.44 ungated reference, −19% — matching the calibrated
prediction), abstained 48 (exactly one per run: the isolated first checkpoint).

## 3. Paired outcome changes (all 5, with autopsies)

| Task/seed | Change | Rollbacks | Attributable? |
|---|---|---|---|
| D28 s4 | F → **P** | 2 | **YES — attributable recovery.** Judge aborted twice; recovered trajectory passed where baseline failed |
| D25 s4 | F → P | 0 | No — stochastic (verifier never intervened) |
| D18 s4 | F → P | 0 | No — stochastic |
| D25 s5 | P → **F** | 2 | **Harmful abort.** Rollbacks derailed a passing trajectory (10.4 s → 187.2 s) |
| D28 s3 | P → **F** | 2 | **Harmful abort.** Same pattern (130.0 s → 127.6 s, budget death) |

## 4. Required numbers

- Success: baseline **31/48**, active **32/48**
- False/harmful aborts: **2/14 interventions** (14%; was 2/9 = 22% in calibration)
- Harmful abort tasks: **D25 s5, D28 s3**
- Recovered tasks: **D28 s4**
- Judge escalations/task: **4.40** (was 5.44)
- SUSPECT rate: **100%** (unchanged — the vote itself is untouched)
- Latency Δ: mean **+9.0 s**, median +3.6 s, p95 +69.1 s
- Infra failures: **0** in all 96 runs. One environmental interruption at 85/96
  (host disk exhausted by an unrelated 53 GB `/tmp/inv` artifact) — resumed via
  `--resume`, zero tasks lost, reported separately per protocol.

## 5. Did the harm signal replicate?

**Partially — the exact pre-registered rules do not fire:**

- Rule 1 (ARCHIVE): requires ≥2 harmful **and** 0 recoveries → not met (1 recovery exists).
- Rule 3 (PROCEED): requires 0–1 harmful **and** ≥2 recoveries → not met (2 harmful).
- Rule 2 (SHADOW ONLY): requires 0 harmful → not literally met.

**Verdict: INCONCLUSIVE between ARCHIVE and PROCEED — stop and state why.**

What improved: net outcome flipped from −4 to +1 task; harm-per-intervention
fell 22% → 14%; escalations fell 19% at zero detection cost; the first ever
attributable recovery (D28 s4) was observed.
What did not: harmful aborts replicated in absolute count (again 2), again on
the volatile D25/D28 family; latency remains materially worse (+9.0 s mean).

## 6. Operational recommendation

Remain **SHADOW ONLY** in production. The calibrated policy is strictly better
than the uncalibrated one and produced the first evidence that recovery *can*
work (D28 s4), but 2 harmful aborts in 48 pairs is still too high to enable
aborting real agent sessions. The binding constraint is no longer triage
precision (SUSPECT is uninformative at 100% rate) — it is **which trajectories
get aborted and how recovery re-enters the task**, i.e., rollback checkpoint
quality and judge threshold behavior on volatile tasks.
