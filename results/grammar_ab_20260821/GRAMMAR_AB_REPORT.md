# Grammar A/B Report — Forced Tool-Call Structure vs Free Generation

**Date:** 2026-08-21 · **Dir:** `verifier/results/grammar_ab_20260821*/`
**Mode A:** parser guard only (`grammar_ab_20260821/runs`) · **Mode B:** parser guard + GBNF tool-call grammar (`grammar_ab_20260821_grammar/runs`)
**Subset:** 8 tasks × 2 seeds (D01–D04, D18, D21, D25, D28), matched manifest, identical sampling

## Phase 1 feasibility (all gates passed)

| Gate | Result |
|---|---|
| GBNF via `grammar` field on /v1/chat/completions | ✅ enforced (verified: output always terminates in complete call) |
| Tool schemas compile to grammar | ✅ `verifier/src/grammar_compiler.py` |
| ROCmFP4 quantization interaction | ✅ none observed |
| TTFT overhead | ✅ **0%** (34 ms both arms) |
| Decode overhead | −8.6% (76.3 → 69.8 tok/s) |

Implementation note: under grammar, streamed output arrives as
`reasoning_content` deltas (the template never closes `<think>` because the
grammar forbids emitting its tags); the harness reads both delta fields.

## Phase 2 results (16 paired runs per arm)

| Metric | Mode A (no grammar) | Mode B (grammar) |
|---|---:|---:|
| **Missing-call prose steps** | **67** (37.9% of 177 generations) | **0** (0%) |
| Task success | **10/16 (62.5%)** | **4/16 (25.0%)** |
| Mean wall time | 100.1 s | 44.7 s |

## Why success collapsed (per-task autopsy)

All 6 regressions (D02 s1/s2, D03 s1/s2, D18 s2, D28 s1) show the same pattern:
under grammar the model produces mechanically perfect but **semantically
shallow** calls — fast loops of `ls -la` / premature `test` — because the
bounded tag-free prose region (400 chars, no `<`) strips the extended reasoning
space Ornith uses on hard tasks. The very prose that looked like "waste" in the
failure-mass analysis is often load-bearing reasoning: D03 passed at 99 s /
52 s without grammar and failed at 43 s / 60 s with it. Faster runs, wrong work.

This confirms the brief's warned failure mode: *the grammar forces tool calls
where the model semantically wanted to reason*.

## Decision against pre-registered Phase 3 gate

| Gate | Threshold | Measured | Pass? |
|---|---|---|---|
| Prose reduction | ≥80% | 100% | ✅ |
| Success regression | ≤ ±2 tasks | **−6 tasks** | ❌ |
| Latency overhead | <5% | decode −8.6% | ❌ |

**STOP** per pre-registered rule: forced calls reduced success by 6 tasks (>2).

## Conclusion

Generation-time structural constraint eliminates missing-call prose completely
but is net-harmful: the prose is not waste, it is reasoning. The correct
boundary is between *formatting enforcement after reasoning* (which the
deterministic parser guard already provides at zero cost) and *forcing
emission before reasoning is done* (harmful). Do not enable grammar constraint
by default. If revisited, the only defensible variant is a much larger prose
budget or triggering the grammar only after N consecutive call-less steps —
both untested here and blocked by this experiment's stop rule.
