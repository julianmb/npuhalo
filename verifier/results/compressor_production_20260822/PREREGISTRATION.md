# Compressor Production Integration — Pre-registration

**Date:** 2026-08-22 · **Dir:** `verifier/results/compressor_production_20260822/`
**Decision question:** Can compression of genuinely large tool output reduce downstream GPU prefill enough to beat the measured 3–4 second NPU compression cost, while preserving the information needed for the next agent action?

## Frozen prior evidence (cited artifacts)

Sources: `verifier/results/compressor_breakeven.json`, `verifier/scripts/run_compressor.py`,
`verifier/results/final-report.md` §Phase 6, `verifier/results/final_verdict.md`.

| Quantity | Prior measurement | Artifact |
|---|---|---|
| Compression ratio | **80–98% reduction** (e.g., 8,001 → 123 chars) on real tool outputs >500 chars | `final-report.md` §Phase 6 |
| Latency distribution | Original prototype claimed ~1.0 s; **2026-08 re-measure: 3,014–3,853 ms median** per size bucket (reasoning-token emission) | `compressor_breakeven.json` |
| Breakeven threshold | Net saving negative below 16 K chars; **positive only at ~32 K chars** (+3,745 ms there); prefill isolated via max_tokens=1 | `compressor_breakeven.json` |
| Model/endpoint | `lfm2.5-tk:1.2b` on FastFlowLM v0.9.46, `:8001`, structured-extraction prompt, temp 0 | `run_compressor.py` |
| Fidelity | **Never scored** — the gap this task closes | — |
| GPU competition | **Never measured** — NPU calls were issued without scheduling evidence | — |

### New Phase-0 discoveries (change the problem statement)

1. **The shell tool hard-truncates output before context insertion:**
   `agent_loop.run_tool()` keeps only the LAST 8,000 chars of stdout and 4,000
   of stderr (`agent_loop.py:204`). The ≥32 K-char bucket is therefore
   *structurally empty* for shell output in this pipeline.
2. The `read` tool caps at MAX_FILE_BYTES = 64,000 bytes — the only path that
   can produce eligible output; no Set D v2 fixture contains a large file.
3. Recorded corpus scan (1,136 real tool outputs across all run archives):
   p50=121, p90=588, p99=5,230, **max=7,999 chars; zero outputs ≥16K**.

Consequence, stated before any measurement: the end-to-end ≥32 K bucket cannot
be populated from existing data or fixtures. Per the ship gate, absent a
legitimate large-output workload the verdict trends BLOCKED regardless of
fidelity results. Fidelity will still be measured on the largest available
real samples (labeled sub-threshold).

## Pre-registered decisions

- **Candidate thresholds:** 16K / 24K / 32K / 48K characters.
- **Default policy hypothesis:** compress only at ≥32K characters unless new
  data justifies another threshold.
- **Scheduling:** compression is never invoked while Ornith is actively
  decoding; every invocation logs tool-end / NPU-start / NPU-end / GPU-start
  timestamps as proof.
- **Skip classes (never compressed):** binary or high non-ASCII content,
  patch/diff payloads (`^diff --git`, `^--- `, `^+++ `, `^@@` density),
  compiler diagnostics requiring exact text, base64/hex blobs — unless
  preservation tests show safety.
- **Failure policy:** any compressor failure or fidelity-check failure falls
  back to the ORIGINAL output. A truncated result is never inserted.
- **Fidelity bar for shipping anything:** zero catastrophic omissions (a
  critical fact present in the original but absent after compression, where
  that fact determines the correct next action).

## Planned phases

1. Safe sidecar integration (`verifier/src/compressor_sidecar.py`) with full
   provenance logging; raw output never deleted; agent tool protocol untouched.
2. Held-out fidelity corpus drawn from recorded real tool outputs, stratified
   by type; deterministic fact-retention scoring + judge spot-checks.
3. End-to-end paired A/B — expected BLOCKED on bucket availability; documented
   either way.
