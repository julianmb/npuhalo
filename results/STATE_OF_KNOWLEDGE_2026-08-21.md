# State of Knowledge — Strix Halo Heterogeneous Agent-Inference Project

**Date:** 2026-08-21
**Prepared by:** Lead ML Systems Engineer
**Repository:** `npuhalo` @ `dbd36d2` (+ companion repo `~/source/strix-halo-speculative-decoding-tests` @ `414e4f6`, branch `update/flm1-ornith`)
**Method:** Raw artifacts inspected first; README/report claims treated as claims until traced to telemetry. Where sources disagree, both numbers are reported with reproduction status.

---

## The Decision Question (RESOLVED 2026-08-21)

> **Can the existing shadow-mode verifier signal — already showing 4/4 catchable failures detected and 0/22 false alarms — be converted into a safe, measurable end-to-end agent improvement when it is allowed to abort and recover?**

**Answer: NO — the verifier line is CLOSED.** Final measured state across 80
paired evaluations:

- Uncalibrated active mode (seeds 1–2): net **−4/32 tasks**, 2 harmful aborts,
  0 attributable recoveries, +11.6 s mean latency.
- Calibrated active mode (frozen `B_consecutive`, seeds 3–5): net **+1/48
  tasks**, 2 harmful aborts on volatile tasks (D25/D28 family), 1 attributable
  recovery (D28 s4), +9.0 s mean latency.
- Triage votes are **information-free**: 100% SUSPECT rate on 433 checkpoints
  across both runs; the vote carries no signal to threshold.

**Final state:** shadow-only logging with frozen policy `B_consecutive`
(sha256 `16a215f2…`), retained purely as a data collector. No further verifier
experiments. Reports: `active_recovery_20260821_1040/` and
`active_recovery_calibrated_20260821_1637/`.

## Structural-constraint line (CLOSED 2026-08-22)

> **Can tool-call structure be forced at the sampler without degrading reasoning? NO.**

GBNF forcing is feasible on this stack (TTFT overhead 0%, decode −8.6%) and
eliminates missing-call prose completely (**67 → 0 steps**), but task success
collapsed **10/16 → 4/16** in matched A/B. The missing-call prose is
**load-bearing reasoning**, not formatting waste: forcing emission before
reasoning completes converts thoughtful passes into shallow fast failures.
The deterministic parser guard remains the correct formatting boundary
(0 false rejects on 1,004 replayed calls; invisible on healthy traffic).
No further structural-constraint experiments.
Reports: `results/grammar_ab_20260821/`.

---

## 1. Hardware and Runtime Truth

Only measured or directly observable facts.

| Component | Hardware/runtime | Model | Endpoint | Measured prefill | Measured decode | Memory/power if known | Evidence file |
|---|---|---|---|---:|---:|---|---|
| Primary generator | llama-server, ROCmFPX fork, Vulkan0 (RADV STRIX_HALO, KHR_coopmat) | `Ornith-1.5-35B-A3B-ROCmFP4.gguf` (19 GB), ctx 16384, `--no-context-shift -np 1` | `:8012` | **151.4 ms TTFT** (short prompt); long-prompt TTFT varies 176 ms–16 s depending on cache state (see §2.13) | **72.04 tok/s** (single source, `verifier/results/benchmark.md`); spec-dec repo measured equivalent ~50 tok/s on short gens (674 ms); range "56–80 tok/s" asserted in two docs, not independently reproduced | ~85 W class (doc claim); 19 GB weights in 128 GB UMA | `verifier/results/benchmark.md`, `docs/baselines.md`, spec-dec `REPORT.md` |
| Generator + NPU concurrently | same GPU server + FLM generating | qwen3.5-0.8b-FLM | :8012 + :8001 | — | GPU decode **−16.5%** (38.7 → 32.3 tok/s); NPU itself barely slowed (42.9 → 40.1 tok/s) | — | `scripts/phase0_real_npu.py` results in `docs/final_verdict.md` §6.1 |
| NPU runtime | FastFlowLM **v0.9.46** (binary query), XDNA 2 via `/dev/accel/accel0`, amdxdna + SVA (`iommu=pt iommu.passthrough=0`) | `lfm2.5-tk:1.2b` (verifier/compressor), `qwen3.5:0.8b` (drafter), 30+ models in store | `:8001` | long-prompt prefill slow (handoff A/B: NPU TTFT ~700–1430 ms incl. prefill) | **~43 tok/s** verifier; **40–43 tok/s** 0.8B; **cold start ~74 s** (graph compile + weight load) | ~2 W (doc claim, consistent across sources) | `standalone-eval/results/serving.md`, `docs/npu_0.8b_bench_real.json`, README caveat #4 |
| Escalation judge | llama-server (ROCmFPX), Vulkan0, native logprobs ✅ | `Qwen3.5-2B-Q4_K_M.gguf`, ctx 8192 | `:8013` | fast (short prompts) | adequate; thinking-mode burns tokens unless `enable_thinking:false` | — | live config; `verifier/src/escalator.py` |
| CPU reference | PyTorch in-process, Zen 5 16C/32T | LFM2.5-1.2B-Thinking bf16 | in-process | — | **18–24 tok/s** | — | `standalone-eval/results/serving.md` |
| Logprobs/grammar support | FLM stock returns `"logprobs": null`; **patched** via `patches/0003` (**logprobs unmerged upstream**); GBNF grammars submitted as **PR #487** (status unknown). llama.cpp `echo=true`+prompt-logprobs via `patches/0002` (**unmerged**, PRs #15189/#17935 open) | — | — | — | — | — | spec-dec `README.md` patch matrix |

Contention was **directly measured**, not modeled: proxy DRAM stress −11.0% @ ~8 GB/s,
−17.2% @ ~16 GB/s added traffic vs a pre-registered 5% gate (`scripts/phase0_contention.py`,
`docs/final_verdict.md` §6); real concurrent NPU generation −16.5% (above).

CPU capabilities beyond the PyTorch reference were **not** benchmarked.

---

## 2. Results Inventory

| # | Experiment | Hypothesis | Dataset/workload | Metric | Result | Sample size | Reproduced? | Conclusion |
|---|---|---|---|---|---|---:|---|---|
| 2.1 | Shadow-mode NPU checkpoint verification | NPU triage flags real failures without false alarms | Set D, 30 agentic tasks | Recall on catchable failures / false alarms | **4/4 recall, 0/22 false alarms**; shadow pass 22/30 | 30 tasks, 8 failures, 4 catchable | Single run, raw autopsies preserved | Detection signal is real but n is small; LFM votes SUSPECT, never ABORT |
| 2.2 | Active abort-and-recover verification | Recovery converts detected failures into passes | Narrow set: 8 failing tasks × 5 seeds × 3 arms | Pass-rate delta, rollback conversion | **INVALID FOR OUTCOME CLAIMS** — died 59/120 (server crash → HTTP 400, unhandled). baseline 40/40 done, parser_guard 24/40, active_esc 0/40 | 59/120 | No | **Open decision question.** Infrastructure repair required before any claim |
| 2.3 | Parser guard (deterministic tool-call schema validation) | Catches malformed `<tool_call>` stalls | Same narrow set | Stall recovery | Partial: 24 parser_guard runs exist, **unanalyzed**. The "fixed 50% of stalls" claim is confounded — 4/8 silent failures were `max_tokens=420` truncation fixed by harness change (1024 + `reasoning_content`), explicitly not creditable to the guard | 24/40 runs | No | Claim withdrawn from README; true guard value unmeasured |
| 2.4 | Qwen escalation judge calibration | CONTINUE/ABORT logprob mass separates broken from clean trajectories | 2 hand-built trajectories | r = P(continue prefix mass) | Broken r=0.000, clean r=0.806 | **n=2** | No | Directionally useful, statistically meaningless; generalization unknown |
| 2.5 | Standalone pairwise judging (NPU-verifier quality) | Small model can grade trajectory pairs | 30 curated SE pairs | Pairwise accuracy | Hard rules 90%; **LFM2.5 holistic 96.7%** (29/30); Qwen2.5-1.5B holistic 100% | 30 | Single run, raw JSON kept | LFM2.5 is a competent *pairwise* judge; note this does not establish single-trajectory verdict quality |
| 2.6 | Raw-logprob discrimination (LFM) | Zero-shot letter-logprobs separate correct from incorrect answers | 6 canonical scenarios | Expected score separation | **FAILS**: correct-math 0.022 vs incorrect-math 0.025 (indistinguishable) | 6 | Single run | Distribution-based scoring on LFM needs reasoning-first prompting, not raw logits |
| 2.7 | Tool-output compressor | NPU extraction saves context at acceptable latency | Real tool outputs >500 chars; synthetic breakeven sizes 1K–32K chars | Size reduction; net wall-clock | Reduction 80–98%. Latency: **~1.0 s original vs ~3–4 s re-measured** (thinking tokens). Net saving negative <16K chars; breakeven **~32K chars** (+3.7 s there) | prototype n≈dozens; breakeven 6 sizes × 3 trials | Re-measured ✅ | Only worthwhile for very large outputs; ">500 chars" guidance was wrong by ~64× |
| 2.8 | NPU-direct fast lane / routing | Router sends short queries to NPU cheaper | 28-query A/B (20 short-answerable, 8 GPU-appropriate) | Routing accuracy; answer accuracy; latency | **Routing accuracy 25%** (sends nearly everything to GPU; one reasoning query to NPU). Answers: NPU 78% vs GPU 100%; latency parity (~1.9 vs ~2.2 s). Original "1,384 ms vs 14,635 ms" was latency-only, different config | 28 queries × 3 calls | Re-measured ✅ (contradicts latency-only claim) | Fast lane blocked on classifier quality |
| 2.9 | NPU→GPU speculative decoding | Draft on NPU, verify on GPU beats GPU-only | Short-gen benchmarks, multiple prompt types | End-to-end latency; acceptance | **Negative, twice independently**: npuhalo harness 2,640 vs 1,313 ms (~2× slower); spec-dec native C++ zero-HTTP: chunked(8) 1,425 ms vs GPU-only 674 ms, acceptance 66% chunked / 14% batch | dozens of runs each | ✅ Reproduced across two codebases | Structural: MoE target decodes faster than NPU drafter; closed |
| 2.10 | TTFT handoff (NPU burst → GPU continue) | Instant first tokens improve perceived latency | 12 prompts (8 short, 4 long-prefix), blind 2B judge | TTFT; coherence | Original claim 347 ms vs 1,587 ms (1.8× win) **does not reproduce**: handoff loses ~1,430 vs ~730 ms even on long prompts (NPU prefills slowly). Quality tie (judge 7.3 v 7.3; 92% bursts end mid-sentence, continuations recover) | 12 paired cases | Re-measured ✅ (contradicts original) | Archived; do not deploy on current firmware |
| 2.11 | Verifier concurrency capacity | How many streams can one NPU verifier serve | Async client sweep | Median/p90 verdict latency | 1,147 ms @1 → 1,922 @2 → 3,607 @4 → 5,237 @6; saturation **3.6–3.8 streams** | 6+12+24+36 samples per level | Single sweep | Capacity fine for ≤3 concurrent agent sessions |
| 2.12 | GPU/NPU bandwidth contention | Concurrent NPU work degrades GPU decode | Controlled DRAM stress + real NPU load | Decode delta | −11.0% @8 GB/s, −17.2% @16 GB/s (proxy, 45 s settle, medians of 3, 5% gate); **−16.5% real** | 3 runs/level + real test | ✅ Both methods agree | Any concurrent-NPU-during-decode design pays double-digit GPU tax |
| 2.13 | Prefix/KV cache behavior | Prompt cache helps agentic reuse | repeat/shared-prefix/multi-turn patterns | Prefill time | Identical repeats: 9,268→88 ms. Shared-prefix varying-tail: **works only without draft-MTP** (175–369 ms vs 16.3 s miss). Multi-turn: 273–350 ms/turn | patterned bench | Single run | Cache interacts badly with MTP spec decoding; relevant to serving config choices |
| 2.14 | Embedded MTP depth sweep | K depth tuning | 16-config grid | gen tok/s | K=4 optimal (33.8–38.2); K=6 regresses, K=8 catastrophic | 16 configs | Consistent across docs | Serving config settled |

### Cross-source discrepancies

| Quantity | Source A | Source B | Resolution |
|---|---|---|---|
| Ornith decode throughput | 72.04 tok/s (`verifier/results/benchmark.md`) | ~50 tok/s implied (spec-dec REPORT, 674 ms short gens); "56–80 tok/s" prose | Not reconciled; different dates/configs/context lengths. Treat as **50–72 tok/s band** until one controlled re-measurement exists |
| Compressor latency | ~1.0 s (original prototype) | ~3–4 s (2026-08 breakeven harness) | Both real: re-measured run captures reasoning-token emission; original likely measured post-warmth non-thinking path. Use 3–4 s for planning |
| Handoff TTFT | 347 ms win (original) | Loses by ~2× (2026-08 A/B) | Re-measurement supersedes; firmware/llama-server TTFT improved since |
| Router latency gap | 1,384 vs 14,635 ms | parity ~1.9 vs 2.2 s | Different configs (GPU thinking mode); A/B supersedes for current stack |
| Shadow pass rate 73.3% vs baseline 80% | shadow autopsy | calibrated baseline | Harness confound: `max_tokens=420` truncation fixed between runs; not model variance |

---

## 3. Gaps and Open Questions

1. **The decision question (§ top): active recovery value — unmeasured.** Everything else is secondary.
2. Escalation judge separation rests on n=2 trajectories; no distribution over failure types.
3. Parser-guard value unmeasured independent of the max_tokens harness fix.
4. Compressor output fidelity never scored against ground truth.
5. All agentic results are on synthetic Set D; no real-traffic measurement.
6. Ornith throughput lacks one controlled modern number.
7. ROCmFPX `llama-server` crashes (`common_context_seq_rm` abort) under rapid varying-length prefills — documented, workaround = supervisor + client retry; blocks long unattended evals (direct cause of 2.2 invalidation).

## 4. Preliminary ROI Ranking (pre-decision)

1. **Repair active-mode experimental validity and run the decisive paired test** ← chosen as next action (this document's follow-on task).
2. Compressor latency kill (non-thinking/grammar-constrained extraction) — deferred per scope.
3. Router classifier improvement — deferred.
4. Upstream patch PRs — deferred.

*Items 2–4 are explicitly deferred until the active-verification decision question is answered.*
