# Escalation Tier Evaluation — Qwen3.8-27B as Failure Escalation for Ornith

**Date:** 2026-08-22 · **Dir:** `verifier/results/escalation_20260822/`
**Decision question:** does a stronger dense model tier convert Ornith's persistent failures, justifying an escalate-on-exhaustion tier?

## Verdict: **STOP — escalation tier NOT justified; harness problem diagnosed instead**

Per the pre-registered gate: *"27B also fails them → these tasks are
harness/environment-limited, not model-limited. STOP and diagnose the harness."*
The gate fired. Phase 3 (wiring) was not built.

## 1. Tier bring-up (Phase 1)

| Item | Result |
|---|---|
| Model | `unsloth/Qwen3.8-27B-GGUF` **UD-Q4_K_M**, 16.4 GB (`Q4_K_M` exact name not in repo; UD-Q4_K_M is unsloth's dynamic Q4_K_M-class quant) |
| Serving | llama-server ROCmFPX fork, Vulkan0, `:8014`, ctx 16384 |
| Co-residency | ✅ Ornith :8012 + 27B :8014 healthy simultaneously; **61 GB UMA free** of 124 GB; zero instability |
| Perf @2K prompt | TTFT median 121 ms (range 120–2,437), decode 12.1 tok/s |
| Perf @8K prompt | TTFT median 274 ms (one pathological 57 s outlier), decode 7.9 tok/s |
| >16K prompts | rejected with `exceed_context_size_error` (correct ctx-limit behavior) |

Escalation latency reality: a full 15-step 27B attempt costs **8–16 minutes**
at ~8–12 tok/s decode — acceptable only if it converts failures Ornith cannot
pass.

## 2. Capability test (Phase 2 — the decisive question)

Capability-limited set (≤25% pass across all 160 corpus runs): D03 (20%), D04
(0%), D06 (0%), D21 (10%). Each task × 2 seeds on the 27B, identical harness,
tools, budget (15 steps), temperature.

### Task × model capability matrix

| Task/seed | Ornith baseline | Qwen3.8-27B direct | Conversion? |
|---|---|---|---|
| D03 s1 | P (68 s) | **P** (223 s, 6 steps) | n/a — Ornith passes it |
| D03 s2 | P (136 s) | F (951 s) | — |
| **D04 s1** | F | **F** (803 s) | ✗ |
| **D04 s2** | F | **F** (679 s) | ✗ |
| **D06 s1** | F | **F** (770 s) | ✗ |
| **D06 s2** | F | **F** (850 s) | ✗ |
| D21 s1 | P (139 s) | F (477 s) | regression on this seed |
| D21 s2 | F (139 s) | F (547 s) | ✗ |

**Conversions of tasks Ornith never passes: 0 of 4 attempts** (D04×2, D06×2).
Gate required ≥2. Not met.

## 3. Harness diagnosis (the stop-rule finding)

Trajectory autopsies show both models fail D04/D06 with the **identical
signature**: fix the visible bug, pass the visible test, then loop — *"the
visible test passes but the hidden grader fails"* — until budget death.
Two architecturally different models (35B-A3B MoE FP4 vs dense 27B Q4_K_M)
converging on the same wall indicates the tasks, not the models.

Root cause found in `verifier/data/set_d_agentic.jsonl`:

- **D04**: instruction says restocking a missing item should be *ignored*;
  the hidden grader asserts `i.restock('zz',3); i.items.get('zz')==3` —
  demanding the item **was added**. Instruction and grader contradict.
- **D06**: instruction says empty lines "should be preserved"; grader expects
  `parse_records('#x\n\n y \n') == [' y ']` — the empty line is **absent** from
  its own expected output.

These tasks are unsolvable-as-specified: following the instruction fails the
grader; only guessing hidden semantics could pass. This deflates absolute Set D
pass rates for every model and every prior arm symmetrically (comparisons
between arms remain internally valid), but absolute capability numbers carry
this caveat going forward.

Secondary findings:
- The one parser live/post-hoc mismatch (D06 s1) was a flaw in my comparison
  code (post-hoc parsed only the 200-char head), not in the parser; live
  detection was correct (double `<function>` block at char 4173).
- D04 trajectories also show tool timeouts ("find timed out") worth hardening.

## 4. Ship/don't-ship

**Don't ship the escalation tier.** It converts nothing Ornith cannot already
pass, costs 8–16 minutes per attempt, and the "persistent failures" it would
target are provably task-authoring bugs. The correct next action is fixing the
D04/D06 task specifications and re-measuring — after which the capability
question can be re-asked cleanly.
