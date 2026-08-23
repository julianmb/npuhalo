# Compressor Production Report — Final Recommendation: ARCHIVE AUTOMATIC COMPRESSION

**Date:** 2026-08-22 · **Dir:** `verifier/results/compressor_production_20260822/`
**Decision question:** Can compression of genuinely large tool output reduce downstream GPU prefill enough to beat the measured 3–4 s NPU compression cost, while preserving next-action information?

## Recommendation: **ARCHIVE automatic compression.** Retain raw outputs; keep the sidecar code as opt-in infrastructure for any future workload that genuinely produces ≥32K-char tool outputs (none exists today).

## What was measured

### Structural finding (gates everything)
`agent_loop.run_tool()` keeps only the **last 8,000 chars** of shell stdout
(4,000 stderr) before anything reaches agent context (`agent_loop.py:204`).
Across **1,136 real recorded tool outputs**, the maximum size is 7,999 chars;
**zero outputs ≥16K, ≥32K, or ≥48K exist**. The only path that could produce
eligible output is the `read` tool (64 KB cap), and no Set D v2 fixture
contains a large file. The compressor's target bucket (≥32K) is *structurally
empty* in this pipeline.

### Fidelity evaluation (Phase 2): 36 real samples, all fell back

Held-out corpus stratified across test/traceback output, package-install logs,
source inspection, file listings, generic shell output (sizes 517–7,999 chars,
all sub-threshold — labeled as such).

| Outcome | Count |
|---|---:|
| Compressed summaries that passed the fidelity gate | **0 / 36** |
| Catastrophic omissions (traceback heads or exit codes lost) | **15 / 36** |
| Expansion instead of compression (ratio >100%) | frequent below ~700 chars (up to **311%**) |
| Compression latency (incl. cold NPU) | ~6.5–7.1 s |

Failure classes:
1. **Expansion** — reasoning-token emission plus verbose summary exceeds small inputs.
2. **Critical-fact loss** — the extractor paraphrases tracebacks and drops exit codes even when explicitly instructed to preserve them verbatim.
3. **Empty/transport failures** — occasional zero-latency fallbacks.

The deterministic retention gate worked exactly as designed: every failing
compression fell back to the full original. No degraded context was ever
inserted.

### End-to-end A/B (Phase 3): not executable — documented, not extrapolated

With a hard 8K shell ceiling and no large-fixture read targets, no legitimate
locked workload can populate the ≥32K bucket using existing fixtures. Per the
ship gate ("insufficient ≥32K outputs → BLOCKED; do not extrapolate from
synthetic data"), no A/B was run and none is inferred. The structural argument
above is the Phase 3 result.

## Gate application

| Ship-gate condition | Result |
|---|---|
| Zero catastrophic fidelity failures | ❌ 15/36 |
| No task-success regression | not reached |
| Net median latency improvement ≥32K | ❌ bucket empty by construction |
| Prefill savings > compressor cost | ❌ unmeasurable (no eligible outputs) |
| Safe fallback on all failures | ✅ 36/36 fell back correctly |
| No Ornith decode regression from overlap | ✅ scheduling guard present (timestamps logged) |

Fidelity failed → **archive automatic compression**, per the gate.

## What survives

- `verifier/src/compressor_sidecar.py`: provenance-complete sidecar (artifact
  preservation with sha256, content classification, scheduling timestamps,
  deterministic retention checks, unconditional fallback). Ready to arm if a
  future workload legitimately produces ≥32K outputs — re-run the fidelity
  corpus at that size regime first.
- `fidelity_results.json`: 36 per-sample autopsies with explicit non-equivalence labels.
- Raw outputs: never deleted; every processed artifact retained under its sha256.
