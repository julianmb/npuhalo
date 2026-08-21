# Parser Guard — Replay Validation & Re-derived Failure Mass

**Date:** 2026-08-21 · **Parser:** `verifier/src/toolcall_parser.py` · **Replay:** `verifier/scripts/replay_parser_guard.py`
**Corpus:** 160 recorded runs (`active_recovery_20260821_1040` + `active_recovery_calibrated_20260821_1637`), 1,418 steps

## Acceptance criterion #1: False rejects — PASS (0/1,004)

Every step the reference `parse_tool_call()` accepted (1,004 steps) was streamed
through the incremental parser in 4-char chunks: **zero UNRECOVERABLE
classifications**. Unit-level property also verified: no proper prefix of a
valid call ever classifies UNRECOVERABLE.

Two parser bugs were found and fixed during development (both caught by tests,
never reached live use):
1. Bare `<` starting `<parameter=…` failed as "unexpected content".
2. Schema-strict param validation rejected calls the reference parser tolerates
   (`<function=write><parameter=command>…`). Resolution: schema mismatches are
   recorded as `schema_warnings`, never parse failures — structural breakage is
   the parser's domain; argument errors belong to the tool layer.

## Re-derived failure mass (stop-rule #3 triggered)

The "~50% of failures are malformed native `<tool_call>`" estimate is **STALE**.
Actual classification of all 1,418 steps:

| Category | Count | Share |
|---|---:|---:|
| Successful parses | 1,004 | 70.8% |
| Missing-call prose (no `<tool_call>` emitted at all) | 410 | 28.9% |
| Truncated mid-call (generation ended at max_tokens mid-structure) | 2 | 0.14% |
| Meta-mentions of the literal tag inside prose (false-positive traps) | 2 | 0.14% |
| **True malformed calls** | **0** | **0%** |

Implications:
- **Projected token savings from early abort ≈ 0** on this corpus. The two
  truncation cases were already complete when generation ended; nothing was
  saveable.
- The dominant addressable failure mode is **missing-call prose** (410 steps,
  each wasting a full generation), which is a prompting/policy problem, not a
  parsing problem.
- The parser remains valuable as **zero-cost structural insurance**: a correct
  deterministic parser cannot false-reject valid JSON, so it protects future
  traffic at no measured risk.

## Live validation summary

See `parser_guard_20260821/` (16 shadow + 16 active runs, seeds 1–2, 8 tasks):
- Shadow: 0 live detections, 0 post-hoc detections, **100% agreement**, 0 false aborts.
- Active (abort+re-prompt armed): **0 detections fired** — parser fully
  invisible to healthy trajectories; pass-rate delta vs recorded baselines
  (−3/16) is sampling noise with no causal path from the parser (it never
  intervened).
