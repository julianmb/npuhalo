# Triage Calibration Protocol — 2026-08-21

**Status:** Pre-registered before any policy evaluation on the frozen dataset.
**Frozen calibration dataset:** `verifier/results/active_recovery_20260821_1040/` (32 paired evaluations, seeds 1–2). **Held-out:** seeds 3–5. Seeds 3–5 will not be looked at until the champion is frozen and committed.

## 1. What counts as a SUSPECT step now?

A *checkpoint* is one Tier-2 verification call (`_verify_step_npu`, k=3 samples,
temperature 0.7, model `lfm2.5-tk:1.2b`). A checkpoint is **SUSPECT** iff the
majority of its 3 votes is SUSPECT (i.e., ≥2/3), exactly as recorded in
`step.verdict.votes`. This definition is unchanged from the frozen run.

## 2. What new rule will reduce SUSPECT escalation rate?

The NPU vote itself is untouched (model, prompt, k, temperature fixed). What
changes is the **escalation gate** — which SUSPECT checkpoints are allowed to
wake the Tier-3 judge. Candidates (all computable from logged features):

| Policy | Rule | Logged feature used |
|---|---|---|
| **A (reference)** | Escalate on every majority-SUSPECT checkpoint (current behavior) | `verdict.verdict` |
| **B (consecutive)** | Escalate only if the **last 2 checkpoints** were both majority-SUSPECT. An isolated SUSPECT (previous checkpoint not SUSPECT) → ABSTAIN | sequence of `verdict.verdict` |
| **C (unanimity)** | Escalate only if the checkpoint's votes are **unanimous SUSPECT (3/3)**. A 2/3 split → ABSTAIN | `verdict.votes` |

## 3. Abstain / LOW-risk bucket

A gated-off checkpoint becomes **ABSTAIN (LOW-risk)**: it is logged, produces no
judge call, and — pre-registered here — **advances the rollback snapshot exactly
as a CONTINUE verdict does** (the alternative, blocking snapshot advance, would
change recovery semantics and confound the comparison).

## 4. Abort rule using the calibrated output

Unchanged and non-negotiable: **only the Tier-3 judge may abort** (jeopardy
r ≤ 0.50), with `max_rollbacks = 2`. The policy gates *escalation*, never
directly aborts. A raw NPU ABORT vote (never observed in practice) still
aborts directly, as today.

## 5. Pre-registered success/failure criteria for seeds 3–5

Definitions (paired, per task/seed, both modes COMPLETED):
- **Attributable recovery:** active PASS with ≥1 rollback where baseline FAILED.
- **Harmful abort:** active FAIL with ≥1 rollback where baseline PASSED.

Decision rules (exact):
1. ≥2 harmful aborts on seeds 3–5 **and** 0 attributable recoveries → **ARCHIVE active abort**.
2. 0 harmful aborts, success parity within ±1 task, but latency materially worse than baseline → **SHADOW ONLY** unless latency also fixed.
3. 0–1 harmful aborts, ≥2 attributable recoveries, latency delta not worse than the frozen active mode (+11.6 s mean) → **PROCEED to hardened pilot**.
4. Fewer than 20 valid paired runs complete → **BLOCKED**.

Champion selection (seeds 1–2 replay only), ranked:
1. No additional estimated harmful aborts vs policy A (abort set must be a subset of A's).
2. Largest reduction in judge escalations per task.
3. Tie-break: simpler rule.

Known replay limitation, stated in advance: counterfactual abort decisions can
only be evaluated at checkpoints where the judge was actually scored under
policy A (r values exist only there). Policies B/C therefore can only *remove*
aborts relative to A in replay, never add them. Live held-out runs are the real
test.

## Measured timing basis for latency estimates

Marginal cost per escalation will be estimated by least-squares fit of
`wall_time_delta_s ~ escalations` across the 32 frozen active runs (measured,
not assumed). NPU checkpoint latency is taken from `verdict.latency_ms`.

---

## AMENDMENT 1 (pre-held-out, documented 2026-08-21 after seeds-1–2 replay, before any seed-3–5 contact)

Replay on the calibration set exposed a degenerate case the original ranking
did not anticipate: **100% of the 174 checkpoints were majority-SUSPECT and
none was unanimous (3/3)**, so policy C_unanimous escalates **never** — it
disarms the verifier entirely while retaining its ~1 s/checkpoint latency cost,
and discards all 6 detection-aborts on baseline-failed catchable tasks.

Amended champion rule (adds one constraint to the pre-registered ranking):

> **0. Recall preservation (new, highest priority):** the champion must retain
> the complete policy-A abort set on baseline-failed catchable tasks
> (`aborted_in_baseline_failed_catchable` unchanged vs A).

Rationale: the experiment's purpose is to test whether *calibrated* triage
preserves recovery value at lower cost; a policy that cannot recover anything
answers the question trivially and wastes the held-out set. This amendment was
made strictly between calibration and held-out phases; seeds 3–5 remain
unexamined.

**Champion under amended rule: B_consecutive** (identical abort set to A — all
12 aborts including all 6 catchable-failure aborts — while cutting 32/174
escalations, ≈30 s saved per 32-run block at the measured marginal escalation
cost of 0.94 s).
