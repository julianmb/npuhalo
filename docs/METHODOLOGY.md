# METHODOLOGY — How to run agent-inference experiments you can trust

This document describes the experimental machinery used across every finding
in [FINDINGS.md](FINDINGS.md), so others can replicate the rigor (or audit it).
The machinery is not decoration: it is what makes negative results believable.

## 1. Pre-registered decision gates

Before any experiment, write a protocol that states:

- the **hypothesis** and the **decision question** in one sentence;
- exact **pass/fail thresholds** chosen *before* looking at results
  (e.g., "convert >=2 of the capability-limited tasks", "regression <= +/-2 tasks");
- **stop rules** — conditions under which you stop and report rather than expand scope.

Real examples: `verifier/results/triage_calibration_protocol_2026-08-21.md`
(includes a numbered amendment made *before* held-out contact, with rationale),
`verifier/results/compressor_production_20260822/PREREGISTRATION.md`.

Rules of practice:
- Never tune on held-out data. Split calibration vs held-out seeds up front.
- If a threshold proves degenerate after calibration but before held-out,
  amend the protocol **in writing** with rationale (see AMENDMENT 1 in the
  triage protocol). Never amend silently.
- If no pre-registered gate matches an outcome exactly, report INCONCLUSIVE
  and say why. Do not pick the nearest flattering verdict.

## 2. Locked manifests and dataset versioning

- Every run writes a `manifest.json`: task IDs, dataset SHA-256, sampling
  configuration, endpoints, model identifiers, timestamps.
- The manifest is **immutable once written**; later invocations load it rather
  than rewrite it (`run_active_recovery.py::build_manifest`).
- Datasets are versioned with a hash file
  (`verifier/data/set_d_version.json`) and superseded versions are archived
  with defect notes (`verifier/data/archive/`).

Why: two experiments measured against "Set D" meant different bytes when v1
contained contradictory tasks. The hash makes that visible forever.

## 3. Task validation as CI

Every benchmark task ships a known-correct reference solution
(`verifier/data/reference_solutions.json`). CI applies each solution to a
fresh workspace and requires the hidden grader to pass
(`verifier/scripts/validate_set_d.py`, currently 30/30).

This catches the worst failure class in agentic benchmarks: tasks whose
instruction and grader contradict each other. Two such tasks survived multiple
review rounds and only fell to the reference-solution gate. If a reference
following the instruction fails its own grader, the task is wrong — fix or
quarantine before measuring anything on it.

## 4. Paired designs

- Same task IDs, same seeds, same sampling settings across arms; only the
  treatment differs.
- Interleave arm order per task to cancel warm-state drift
  (`modes = args.modes if i % 2 == 0 else reversed(...)`).
- Compare only pairs where **both** arms completed. Report incomplete runs
  separately, never as failures.
- For stochastic agents at temperature > 0, treat single-seed deltas as noise:
  prefer >= 2 seeds and report per-task autopsies for every outcome change.

## 5. Resume-safe harnesses and honest failure accounting

Pattern (see `verifier/scripts/run_active_recovery.py`):

- **Locked manifest** + `--resume`: completed (task, mode, seed) triples are
  never recomputed; interrupted runs continue from disk.
- **Health-check before every task** and after any failed request; wait for a
  supervised server restart, retry the same request at most once.
- **Explicit outcome classes**: `COMPLETED`, `INFRA_FAILURE`
  (transport/crash), `CONTEXT_OVERFLOW` (server context limit). A failed task
  is never silently dropped and never counted as a model failure.
- **Append-only JSONL event log** per run: request timestamps, endpoint
  health, HTTP status + response body, server-restart count, task lifecycle.
- `--dry-run` validates health, paths, and one known-safe request before
  touching real data.

The value was demonstrated twice: an ENOSPC interruption at run 85/96 resumed
with zero lost tasks, and a server-crash class that killed earlier sweeps
became a documented, retried event.

## 6. Fidelity gates with unconditional fallback

Any mechanism that rewrites content entering an agent context must:

1. Preserve the full original as a source-of-truth artifact (path + sha256).
2. Extract required facts deterministically (paths, exit codes, traceback
   heads, failing tests) and verify they survive verbatim post-rewrite.
3. Fall back to the original on **any** failure — transport, empty output, or
   fidelity loss. Never insert a truncated result.

Reference implementation: `verifier/src/compressor_sidecar.py`; evaluation
pattern: `verifier/scripts/eval_compressor_fidelity.py` (stratified real
samples, per-sample autopsies, explicit catastrophic-omission labels).

## 7. Re-measure inherited claims

Treat numbers inherited from earlier phases or other repos as hypotheses.
Three inherited claims failed reproduction during this project (TTFT handoff
1.8x win, router latency gap, compressor ~1 s latency) — each because the
stack had moved or the original measurement omitted a confound. Date every
number, keep the raw records committed, and state which artifact supports
each claim in your write-ups.

## 8. Reporting conventions

- Exact numerator/denominator everywhere ("31/48", not "~65%").
- Separate classification quality from recovery correctness from serving
  reliability from end-to-end value; a good number in one category must never
  imply another.
- Disclose judge models and their limits when LLM scoring is involved.
- Commit raw per-run records next to summaries; document anything excluded
  and why (see FINDINGS Appendix A).
- Write the stop-and-report memo when a gate blocks you. A blocked experiment
  with a diagnosed cause is a result.
