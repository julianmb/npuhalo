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

---

## Addendum (2026-08-22): D06 s1 anomaly resolved — Task 0 closure

**Anomaly:** one live-vs-post-hoc disagreement during the escalation-session
capability runs (D06 s1, active_esc arm).

**Root cause — two stacked issues, neither a live-parser bug:**
1. **Comparison bug (harness):** `posthoc_agree()` parsed only the 200-char
   `text_head`, while the live parser saw all 4,383 chars. The violation
   (nested `<function>` inside an unterminated first call, char ≈4173) was
   invisible in the head → guaranteed false mismatch. Fixed: post-hoc now
   parses the full stored text; `full_text` is persisted per event.
2. **Parser semantics gap (real, fixed):** the live parser finalized
   UNRECOVERABLE at the corrupted first call even though the generation later
   contained a complete valid `<function=test>` call that the reference
   extractor accepted (`has_parsed_call: true`). Aborting on that signal would
   have discarded a usable call — a latent false-reject class.

**Fix — resync-on-error semantics (parser v2):** a structural violation marks a
*provisional* UNRECOVERABLE and is recorded in `errors`; the scanner jumps to
the next `<tool_call>` opener; if a later complete valid call parses, the final
verdict upgrades to VALID_SO_FAR carrying `protocol_violation_recovered`
warnings. Only violation-without-recovery is finally UNRECOVERABLE. This makes
the parser reference-equivalent by construction: anything the reference
extractor accepts never ends UNRECOVERABLE.

**Validation:** new permanent regression file `tests/test_toolcall_parser.py`
(13 cases incl. the synthetic D06 pattern in both char-stream and single-chunk
forms); full suite **38/38 green**, prefix-safety property re-verified.
