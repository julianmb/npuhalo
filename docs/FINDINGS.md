# FINDINGS — Complete Experimental Record

**Project:** heterogeneous NPU + iGPU LLM inference on AMD Strix Halo
**Branch/tag:** `release/findings-v1` → `v1.0.0-findings`
**Method:** every line below ran under pre-registered decision gates with locked manifests, paired seeds, and committed raw records. Numbers are exact numerator/denominator; each section ends with artifact paths. Negative results are first-class.

**Testbed (constant across all experiments):** Ryzen AI Max+ 395 · 128 GB LPDDR5X-8000 (~273 GB/s shared) · Ornith-1.5-35B-A3B ROCmFP4 on Radeon 8060S iGPU via llama.cpp (`:8012`, ctx 16384) · FastFlowLM v0.9.46 on XDNA2 NPU (`:8001`, `lfm2.5-tk:1.2b` / `qwen3.5:0.8b`) · Qwen3.5-2B-Q4_K_M judge (`:8013`) · CPU agent loop, tools, parsing.

---

## 1. Parser guard — SHIPPED

**Hypothesis.** A deterministic incremental parser for the Qwen-native `<tool_call>` format can catch malformed tool calls with zero false rejects, because valid structure never fails a correct parser.

**Method.** State-machine parser over streamed text (`verifier/src/toolcall_parser.py`): classifies VALID_SO_FAR / RECOVERABLE / UNRECOVERABLE per position; validates structure only (schema mismatches become warnings — replay proved schema strictness creates false rejects). Replay harness streams all recorded generations through it in 4-char chunks.

**Results.**

| Check | Result |
|---|---|
| False rejects on replay | **0 / 1,004 successful reference parses** |
| Corpus scanned | 160 runs · 1,418 steps |
| Regression tests | **38/38** (`tests/test_toolcall_parser.py`) |
| Prefix safety | no proper prefix of a valid call is ever UNRECOVERABLE |

Replay also re-derived the failure mass honestly: of 414 unparseable steps, 410 were *missing-call prose* (no `<tool_call>` emitted at all), 2 were mid-call truncations, 2 were prose mentioning the literal tag — **zero true malformed calls**. The guard ships as structural insurance, not a measured token-saver.

Anomaly closure: the D06 s1 live-vs-post-hoc mismatch was root-caused to a comparison-harness bug (post-hoc parsed a 200-char head) plus latent over-strict abort semantics. Fix: **resync-on-error** — violations are provisional; the scanner jumps to the next opener; only violation-without-recovery stays UNRECOVERABLE. Reference-equivalent by construction.

**Verdict: ARMED / SHIP NOW.**
Artifacts: `verifier/results/parser_guard_replay.json`, `verifier/results/parser_guard_20260821/` (+ `PARSER_GUARD_REPLAY.md` addendum).

---

## 2. Speculative decoding — DEAD, twice over

**Hypothesis.** An NPU drafter feeding an iGPU verifier beats GPU-only decode.

**Method & results.** Two independent implementations:

| Implementation | Best config | GPU-only baseline |
|---|---|---|
| npuhalo HTTP harness | draft+verify | **1,313 ms vs 2,640 ms** (2x slower) |
| spec-dec native C++ (zero-HTTP) | chunked, chunk=8, acceptance 66% | **674 ms vs 1,425 ms** |

Acceptance ranged 100% (trivial factual) down to 0–38% (creative), never enough: a ~3B-active-parameter MoE target decodes at ~50–72 tok/s while the fastest NPU drafter manages ~30–34 tok/s — the "verifier" already outpaces the "draft". Embedded MTP (K=4) at 33.8–38.2 tok/s remains the ceiling and needs no second silicon.

**Verdict: DEAD — structural, not fixable by engineering.**
Artifacts: `docs/final_verdict.md`, companion repo `strix-halo-speculative-decoding-tests`, `docs/mtp_sweep_results.json`.

---

## 3. TTFT handoff — DEAD on current firmware

**Hypothesis.** Streaming an NPU burst as instant first tokens, then letting the iGPU continue, improves perceived latency 1.8x (original measurement: 347 ms vs 1,587 ms).

**Method.** Paired A/B, 12 prompts (8 short control + 4 long-prefix), blind randomized judge scoring (Qwen3.5-2B; disclosed weak instrument), streaming TTFT measurement.

**Results.**

| Regime | Pure GPU TTFT | Handoff TTFT |
|---|---:|---:|
| Short prompts (mean) | 198 ms | 719 ms |
| Long prompts (mean) | ~730 ms | ~1,430 ms |
| Judge quality (blind) | 7.3/10 | 7.3/10 (preference 1/9/1 gpu/tie/handoff) |
| Bursts ending mid-sentence | — | 92% (continuations recovered) |

The original win does not reproduce: current llama-server TTFT improved dramatically, and the NPU must prefill the same long context more slowly than the iGPU. Output coherence was unaffected either way.

**Verdict: DEAD on current stack — archived.**
Artifacts: `verifier/results/handoff_coherence.json`, `docs/HYBRID_NPU_PIPELINE.md` (archived banner), script `verifier/scripts/eval_handoff_coherence.py`.

---

## 4. Query router — DEAD

**Hypothesis.** A small NPU classifier can route trivial queries to the NPU ("fast lane", previously claimed at 1,384 ms vs 14,635 ms) without quality loss.

**Method.** 28-query A/B (20 short-answerable with objective gold answers + 8 reasoning/code queries); routing decisions vs ground truth; answer accuracy graded objectively; latency measured on both paths.

**Results.**

| Metric | Value |
|---|---|
| Routing decision accuracy | **25% (7/28)** — sends nearly everything to GPU; one reasoning query routed to NPU |
| Answer correctness | NPU-direct **77.8% (14/18)** vs GPU **100% (18/18)** gradable |
| Latency | near-parity: ~1,946 vs ~2,210 ms |

The earlier latency gap was measured on a different configuration without any accuracy check.

**Verdict: DEAD — blocked on classifier quality, not speed.**
Artifacts: `verifier/results/router_ab.json`, script `verifier/scripts/eval_router_ab.py`.

---

## 4b. Cross-cutting physics — bandwidth contention

Concurrent NPU work shares the ~273 GB/s bus with iGPU decode:

| Experiment | GPU decode delta |
|---|---|
| Proxy DRAM stress @~8 GB/s added traffic | -11.0% |
| Proxy DRAM stress @~16 GB/s | -17.2% |
| **Real concurrent NPU generation** | **-16.5%** (38.7 -> 32.3 tok/s; NPU itself barely slowed: 42.9 -> 40.1 tok/s) |

This is why every in-loop coupling above failed: the tax is physical.
Artifacts: `docs/final_verdict.md` section 6, `scripts/phase0_contention.py`, `scripts/phase0_real_npu.py`.

---

## 5. NPU live verifier — SHADOW ONLY

**Hypothesis.** The shadow verifier's detection signal (recall 4/4 on catchable failures, 0/22 false alarms across 30 agentic tasks) could convert into end-to-end improvement when allowed to abort and recover.

**Method.** Three-tier design: deterministic parser guard -> NPU triage (LFM2.5-tk, k=3 majority vote, checkpoints every <=250 tokens) -> iGPU judge (Qwen3.5-2B logprob jeopardy, r<=0.50 aborts, max 2 rollbacks). Pre-registered gates; locked manifests; interleaved paired modes; seeds held out from calibration; INFRA_FAILURE accounted separately.

**Phase results.**

| Stage | Result |
|---|---|
| Shadow autopsy (30 tasks) | recall **4/4**, false alarms **0/22**, shadow pass 22/30 |
| Escalator calibration | broken trajectory r=0.000 vs clean r=0.806 (**n=2** — indicative only) |
| Uncalibrated active (32 pairs, seeds 1-2) | baseline **24/32** vs active **20/32**; harmful aborts 2/9 interventions; attributable recoveries 0 |
| Calibration replay | SUSPECT rate **100%** (174/174 checkpoints); zero unanimous votes -> policy C rejected via documented amendment; champion `B_consecutive` frozen (sha256 `16a215f2...`) |
| Calibrated held-out (48 pairs, seeds 3-5) | baseline **31/48** vs active **32/48** (**net +1**); harmful aborts **2/14** interventions; attributable recovery **1** (D28 s4); latency delta mean +9.0 s, p95 +69 s |

**Key mechanistic finding:** the NPU triage vote is information-free — it voted SUSPECT on effectively every checkpoint (100% in both runs, 433 checkpoints total), so no threshold policy can extract signal from it. The judge escalates often but rarely aborts; when it does, rollbacks consumed step budgets faster than they recovered them.

**Contamination caveat:** these numbers were measured against Set D v1, which contained two unsolvable tasks (see section 9). Arm-vs-arm comparisons remain valid; absolute rates are deflated. Verdict direction unchanged by the correction.

**Verdict: SHADOW ONLY** — keep as a data collector with the frozen policy; do not enable aborts; do not reopen active-abort work without new signal.

Artifacts: `verifier/results/active_recovery_20260821_1040/` (FAILURE_ANALYSIS.md, events.jsonl, 64 run records, summary.json), `verifier/results/active_recovery_calibrated_20260821_1637/` (ACTIVE_CALIBRATED_REPORT.md, 96 run records), `verifier/results/triage_calibration/frozen_policy_B_consecutive_v1.0.json`, `verifier/results/triage_calibration_protocol_2026-08-21.md`.

---

## 6. Generation-time grammar constraint — NET-HARMFUL

**Hypothesis.** Forcing tool-call structure via GBNF at the sampler eliminates missing-call prose (29% of recorded steps) without degrading reasoning.

**Method.** Matched A/B on 8 tasks x 2 seeds: Mode A parser-guard only vs Mode B parser-guard + GBNF forcing bounded-prose-then-complete-call structure. Feasibility gates passed first: GBNF enforced end-to-end, TTFT overhead **0%** (34 ms both arms), decode -8.6%.

**Results.**

| Metric | Mode A | Mode B (grammar) |
|---|---:|---:|
| Missing-call prose steps | 67 | **0** |
| Task success | **10/16 (62.5%)** | **4/16 (25.0%)** |

Every regression shows the same signature: mechanically perfect but shallow call loops. The prose targeted as "waste" is load-bearing reasoning — D03 passed at 99 s / 52 s ungrammared and failed at 43 s / 60 s grammared.

**Verdict: NET-HARMFUL — stop rule fired (regression 3x the +/-2-task gate). Prose-is-reasoning.**

Artifacts: `results/grammar_ab_20260821/GRAMMAR_AB_REPORT.md` + `summary.json`, compiler `verifier/src/grammar_compiler.py`.

---

## 7. 27B escalation tier — ARCHIVED

**Hypothesis.** A stronger dense tier (Qwen3.8-27B UD-Q4_K_M, co-resident on the iGPU) converts Ornith's persistent failures when triggered by exhaustion rather than prediction.

**Method.** Capability-limited set derived from corrected data (section 9): D18 (1/5 = 20%), D21 (0/5 = 0%). Gate: convert >=2 of them with the 27B directly. Both models resident on the iGPU — stable with 61 GB UMA free; 27B decode 8-12 tok/s; attempt cost 8-16 min.

**Results (27B direct, identical harness/tools/budget).**

| Task | Ornith (v2) | 27B attempts | Conversions |
|---|---|---|---|
| D18 | 1/5 | 0/4 passes | 0 |
| D21 | 0/5 | 1 pass in 6 attempts (seed 3, 360 s) | counted once |
| **Total** | | | **1 of required >=2** |

**Verdict: ARCHIVED PERMANENTLY** — economics fail (6-14 min per attempt against a 75%-per-task failure rate), and most historical "failures" were benchmark bugs already fixed (section 9). Roadmap redirects to compressor shipping, distillation, multi-tenant serving.

Artifacts: `verifier/results/escalation_20260822/ESCALATION_REPORT.md`, `capability_qwen38/` + `capability_qwen38_v2/summary.json`.

---

## 8. NPU tool-output compressor — ARCHIVED

**Hypothesis.** Compressing genuinely large tool outputs on the NPU saves enough downstream GPU prefill to beat its own 3-4 s cost while preserving next-action information.

**Prior evidence frozen in pre-registration:** ratio 80-98%; latency re-measured 3,014-3,853 ms (the original ~1.0 s claim omitted reasoning-token emission); breakeven negative below 16K chars, positive only at ~32K (+3,745 ms there).

**New structural finding.** The shell tool hard-truncates output before context insertion: `agent_loop.run_tool()` keeps only the last 8,000 chars of stdout and 4,000 of stderr (`agent_loop.py:204`). Across **1,136 real recorded tool outputs**, max size = 7,999 chars; **zero outputs >=16K exist**. Only the `read` tool (64 KB cap) could produce eligible output, and no Set D fixture contains a large file. The compressor's target bucket is structurally empty in this pipeline.

**Fidelity evaluation (36 stratified real samples).**

| Outcome | Count |
|---|---:|
| Compressed summaries passing the deterministic retention gate | **0 / 36** |
| Catastrophic omissions (traceback heads or exit codes lost verbatim) | **15 / 36** |
| Expansion instead of compression (ratio >100%, below ~700 chars) | frequent, up to **311%** |
| Compression latency | ~6.5-7.1 s |

The retention gate worked exactly as designed: every failing compression fell back to the full original; no degraded context was ever inserted.

**Verdict: ARCHIVED automatic compression** (fidelity gate failed; end-to-end >=32K bucket additionally unpopulatable without inventing tasks). The sidecar (`verifier/src/compressor_sidecar.py`) remains as opt-in infrastructure with full provenance logging for any future workload that genuinely produces large outputs — re-run fidelity at that regime first.

Artifacts: `verifier/results/compressor_production_20260822/` (PREREGISTRATION.md, COMPRESSOR_PRODUCTION_REPORT.md, fidelity_results.json), prior `compressor_breakeven.json`.

---

## 9. Set D v2 — the benchmark was broken first

**Discovery path.** The escalation-tier stop rule ("if the stronger model also fails these tasks, diagnose the harness") led to the root cause: two Set D v1 tasks were unsolvable-as-specified.

**Defect history.**

| Task | Defect | Fix (v2) |
|---|---|---|
| D04 | instruction demanded *ignoring* missing-item restock; hidden grader asserted the item was created (`items['zz']==3`) | fixture bugs re-authored to match grader semantics; instruction rewritten to canonical interpretation |
| D06 | instruction said blank lines are preserved; grader omitted one from its own expected output; inline-comment semantics ambiguous | whole-line comments only, whitespace preserved verbatim, blanks excluded — all three parties consistent |
| D17 | shipped `casefile.py` used `os.environ` without importing os (found by the validation gate, not by review) | import added |

**Validation gate.** `verifier/scripts/validate_set_d.py`: every task ships a known-correct reference solution; CI applies it to a fresh workspace and requires the hidden grader to pass. **30/30 pass** on every push. Contradiction = CI failure.

**Corrected Ornith baseline (v2, sha256 `ee46b12d...`, 72 runs, 0 infra failures):** overall **58/72 = 80.6%**. Notable per-task: D04 0%->100%, D06 0%->100% (pure task-bug conversions), D03 20%->60%, D28 70%->100%; D18 1/5 (20%), D21 0/5 (0%) are the only remaining capability-limited tasks.

**Standing caveat:** every absolute pass rate measured against v1 — including all verifier-era numbers in section 5 — is deflated by the unsolvable tasks. Arm-vs-arm comparisons remain internally valid.

Artifacts: `verifier/data/set_d_agentic.jsonl` + `set_d_version.json` + `reference_solutions.json` + `set_d_validation.json`, `verifier/data/archive/` (v1 + defect notes), corrected baseline `verifier/results/rebaseline_v2_20260822/summary.json`.

---

## 10. Lessons

1. **Bandwidth-bound decode physics dominates shared UMA.** One chip, one bus: concurrent small-model work costs the big model double-digit throughput (-16.5% measured) while saving microseconds. Any design that couples an NPU into the decode loop pays the tax; any design that keeps it off-bus (parser guard, shadow logging) pays nothing.

2. **In-loop NPU coupling fails for a second reason: latency asymmetry.** Even where bandwidth were free, the 1.2B NPU model needs ~0.9-4 s per decision against a generator that produces a step in ~2 s. Sidecar timing windows this tight invert value the moment anything degrades.

3. **Prose-is-reasoning.** Agent "rambling" between tool calls predicted task success, not failure: forcing structure at the sampler cut prose to zero and success by 37.5 points. Measure what text *does* before classifying it as waste.

4. **Task-validation gates are load-bearing.** Two of ten tasks were unsolvable and one shipped crashing code — found not by review but by requiring a reference solution to pass every hidden grader in CI. Without that gate, the escalation-tier experiment would have "proven" the 27B incapable.

5. **Re-measure inherited claims.** Three results inherited from earlier phases (TTFT win, router gap, compressor latency ~1 s) failed reproduction on the current stack. Every number in this report carries its measurement date and artifacts for exactly this reason.

6. **Negative results need the same rigor as positive ones.** Pre-registered gates, held-out seeds, paired designs, and INFRA_FAILURE accounting are what make "this does not work" believable — see [METHODOLOGY.md](METHODOLOGY.md).

---

## Appendix A — Artifact exclusions

Everything cited above is committed. Excluded from Git (by `.gitignore`):

| Path | Contents | Summary available at |
|---|---|---|
| `verifier/results/raw/narrow/`, `raw/setd/` | per-run JSON records of the interrupted v1-era narrow sweep (59 runs) and early Set D raw runs | aggregates superseded by `active_recovery_*` dirs (committed in full) |
| `docs/benchmark_results.json` | local benchmark matrix generated by `scripts/npu_benchmark.py` | `docs/mtp_sweep_results.json` (committed) covers the same sweep |
| `models/` | GGUF weights (~16 GB Qwen3.8-27B + MTP heads); re-downloadable via `hf download unsloth/Qwen3.8-27B-GGUF` | n/a |

Per-sample fidelity autopsies, event logs (JSONL), manifests with dataset hashes, and summaries for every experiment are committed under their experiment directories.
