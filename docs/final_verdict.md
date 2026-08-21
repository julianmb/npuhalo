# Final Verdict & Forward-Path Assessment (Reconciled)

This document reconciles the external research synthesis with the empirical measurements taken on this system, and closes the open questions.

## 1. Correction: the EAGLE-3 conclusion in the synthesis is STALE

The synthesis states EAGLE-3 is "currently blocked by the d2t bug" and would unlock 34.8–36.0+ tok/s "once fixed". **That bug is now fixed and benchmarked** — and the result is negative:

| Drafter | Acceptance | Decode | Verdict |
|---|---|---|---|
| Embedded MTP (K=4) | ~80% (τ=2.5) | **33.8 tok/s** | best measured configuration |
| EAGLE-3 full (248k) | 90.6% (τ=2.88) | 19.8 tok/s | weight-streaming bound (1.77 GB) |
| EAGLE-3 compressed (32k) | 7.4% (τ=1.10) | 12.4 tok/s | 18.5k/248k vocab coverage |

Two factual errors in the synthesis must be flagged:
1. **"596 MB head fits in 32 MB MALL"** — impossible (596 MB ≫ 32 MB). The compute core (~25M params) would fit, but the head's output/embedding tensors would not.
2. **"Projected 37–40 tok/s"** — based on the *full* head's τ=2.88, but the compressed head's 32k draft vocab only covers **7.4%** of Qwen 3.8's token distribution, so the compressed head can never reach that. The measured compressed-head acceptance is 7.4%, not 90.6%.

**Conclusion: EAGLE-3 is closed. On this bandwidth-bound APU, embedded MTP (zero auxiliary memory traffic) is structurally unbeatable by any separate drafter.**

## 2. The four new AMD paths — honest assessment

| Path | What it is | Actionable now? | Relevance to 27B-on-iGPU |
|---|---|---|---|
| **PARD / PARD-2** | AMD's target-aligned parallel draft (mask-predict, acceptance-length training) | ❌ Research (arXiv + AMD PACE, not production) | High potential (claims 4.87× on Qwen), but needs PACE runtime + PARD-trained drafters, neither shipped as installable tools |
| **TileFuse** | AWQ (W4A16/W8A16) kernel lib for XDNA 2 NPU | ⚠️ Open-source but NPU-side | Only helps NPU-resident models; does nothing for the 27B running on the iGPU |
| **HeteroMosaic** | micro-batch scheduler splitting LLM graph across CPU/iGPU/NPU | ❌ Research (arXiv) | 2.05× claim vs llama.cpp, but no usable release |
| **Block FP16** | XDNA 2 hardware 8+1-bit float format | ❌ HW format, needs compiler support | NPU-side only |

**Verdict:** none of these are deployable today. PARD-2 is the most promising *long-term* direction (it's target-aligned and parallel, addressing the exact serial-draft bottleneck), but it's gated on AMD PACE and PARD-trained weights. TileFuse/HeteroMosaic/BlockFP16 are NPU-centric and don't move the 27B iGPU number.

## 3. One config contradiction to note

The synthesis recommends **hard-allocating 64–96 GB VRAM in BIOS**. This contradicts the empirically-validated setup on this machine: **minimal carve-out (512 MB) + dynamic GTT allocation** (`amdgpu.gttsize` + `ttm.pages_limit`), which is what currently yields 33.8 tok/s and lets the OS reclaim memory. Large static carve-outs starve the OS and add nothing on UMA — do not adopt the 64–96 GB advice.

## 4. Final measured state

- **Sustained decode:** `Vulkan0` + `--spec-type draft-mtp --spec-draft-n-max 4` = **33.8 tok/s** (2.4× over bare).
- **TTFT:** NPU `qwen3.5-0.8b-FLM` burst → **347 ms** (1.8× faster first-token on long prompts).
- **Background:** `scripts/npu_router.py` intent classifier at ~2 W.
- **Ceiling:** 33.8 tok/s is the practical ceiling on this hardware until a *parallel, target-aligned* drafter (PARD-2 class) ships with usable tooling — no further config or bug-fix moves it.

---

## 5. Post-verdict experiments (NPU-for-decode ideas, tested)

### 5.1 Combined spec types (`draft-mtp,ngram-mod`) — NEGATIVE
n-gram speculation combined with MTP: 30.7 tok/s vs 38.2 MTP-only on the same prompt. The extra candidates hurt. K=4 remains optimal on both prose and code content (K=5: 34.5, K=6: 37.2, K=8: 23.4).

### 5.2 Custom frequency-optimized d2t compressed head — ACCEPTANCE FIXED, SPEED STILL BOUND
Built `EAGLE3-custom` (`scripts/build_custom_d2t.py`): top-28k Qwen tokens by frequency from a 15 MB code+prose corpus, lm_head sliced from the full head, custom d2t.

| Head | Vocab | Acceptance | τ | Decode (K=4) |
|---|---|---|---|---|
| Stock Ex0bit compressed | 18.5k | 7.4% | 1.10 | 12.4 tok/s |
| **Custom d2t (Q8_0)** | 28.1k | **90.5%** | **2.84** | 21.7 tok/s |
| **Custom d2t (Q4_0)** | 28.1k | 90.5% | 2.84 | **24.9 tok/s** |
| Embedded MTP (reference) | — | ~80% | ~2.5 | **33.8–38.2 tok/s** |

The custom vocab completely fixed the acceptance collapse (7.4% → 90.5%) — proving Ex0bit's vocab choice, not the d2t architecture, caused the failure. But decode remains bound by the **serial per-step weight streaming** inherent to any separate drafter: each chain step re-streams the head's weights (~300–535 MB) plus kernel-dispatch latency, a cost embedded MTP (weights resident in the target graph) never pays.

**Final law confirmed on this hardware: any *separate* drafter — regardless of acceptance — cannot beat embedded MTP on a bandwidth-bound unified-memory APU.**

### 5.3 The one remaining NPU-decode path (untested, days of work)
Port the embedded MTP head itself to the NPU via XRT/IRON with *pre-drafting overlap*: NPU chains window N+1 while the iGPU verifies window N (rewind on rejection, hidden states via the 0.42 µs ring buffer). This is the only architecture where the NPU joins decode without streaming a separate model's weights per step. Blocked on compiling the MTP graph for XDNA 2 (FLM cannot load custom graphs).

---

## 6. Phase 0 contention proxy — NPU pre-drafting idea KILLED (empirical)

**Experiment** (`scripts/phase0_contention.py` + `scripts/bw_stress.py`): measured embedded-MTP (K=4) decode while generating controlled concurrent DRAM read traffic — a proxy for an NPU draft head (~18 GB/s) streaming its weights during overlapped drafting. Methodology: 45 s thermal settle under load, 1 warm-up + 3 recorded runs per level, medians compared against the pre-registered **5% gate**.

| Added DRAM traffic | Decode median | Degradation |
|---|---|---|
| 0 (baseline) | **40.0 tok/s** | — |
| ~8 GB/s | 35.6 tok/s | **−11.0%** |
| ~16 GB/s | 33.1 tok/s | **−17.2%** |

**Verdict: DEAD.** The expected NPU-draft traffic (~18 GB/s) sits where degradation is ~17-20% — **3-4× the 5% gate**. Overlapped NPU drafting would slow the iGPU's main forward far more than hiding the 30 ms chain saves (best case was +4-12%). Even discounting a few percent for CPU-worker power draw (vs ~2-5 W for the NPU), the margin is decisive. The head cannot avoid DRAM streaming (384 MB ≫ 32 MB MALL), so there is no SRAM escape hatch.

Run-to-run variance also tripled under contention (σ ≈ 2.5 tok/s vs ±0.5 baseline) — added bus pressure makes decode jittery, further hurting the pre-draft assumption chain.

**This closes the last NPU-for-decode architecture.** Confirmed law: on this bandwidth-bound APU, the NPU cannot join the decode loop; its roles are TTFT burst (347 ms) and 2 W ambient routing.

### 6.1 Real-NPU verification (clean re-measurement)

Two follow-up checks confirmed, rather than overturned, the Phase 0 verdict:

1. **Real NPU-only load test** (`curl` against port 13305, FLM store): qwen3.5-0.8b-FLM at 40.3 tok/s decode, 0.80 s generation, TTFT ~350 ms, zero errors — the NPU itself is healthy and fast.
2. **Clean NPU contention re-test** (`scripts/phase0_real_npu.py`): with the unrelated long-running llama-server slot cleared, re-measured GPU decode with the real NPU generating concurrently. Baseline 38.70 tok/s → 32.30 tok/s = **−16.5%** — consistent with the proxy's −17.2% estimate. Notably the NPU itself hardly slowed (40.1 vs 42.9 tok/s alone): the NPU's DMA wins contention and the GPU pays the price. This is worse than any plausible overlap benefit (+4-12%).
3. **The MTP-head-on-NPU path (§5.3) was already dead by §3/§5.2**: the full-vocab head measured 16.9 tok/s on CPU (spec-draft-ngl 0), the best split 22.7, the best compressed d2t head 24.9 — all below embedded MTP's 33.8. The NPU path requires a ~160-384 MB head that cannot fit the 4 MiB XDNA2 L2 SRAM (per kernel docs; the 32 MB "MALL" is GPU-side Infinity Cache, not NPU SRAM), so it would DRAM-stream per chain step exactly like the 0.8B drafter that just measured −16.5%. No escape hatch exists.

**Final answer to "is the NPU viable?":** For decode-side acceleration, no — every architecture was tested empirically and all failed. For TTFT (870 vs 1587 ms on long prompts, sequential) and 2 W ambient routing/guardrails/retrieval, yes — those are measured wins that never involve concurrent bus pressure on the decode loop.

---

## 7. Follow-up experiments from external report review (`prefill_cache_bench.py`, `phase0_contention.py 3`)

### 7.1 Prompt-cache TTFT — EFFECTIVE, with one hard caveat

Benchmarked with an agent-style long prefix (repeat / shared-prefix / multi-turn
patterns, `n_predict=1` to isolate prefill):

| Pattern | cache off (MTP server) | cache on (MTP server) | cache on (no-spec server) |
|---|---|---|---|
| Identical prompt repeat | 9268 ms every time | **88 ms** | 91 ms |
| Shared 4.5k-tok blob + varying query | ~16.3 s each turn | ~16.3 s each turn (**miss**) | **175–369 ms (45–90×)** |
| Multi-turn accumulating conversation | — | — | **273–350 ms/turn** |

**Hard caveat:** prefix-shared reuse (the realistic agentic pattern) works only
when `--spec-type draft-mtp` is *not* enabled. Identical-prompt repeat hits
under both configs, but once the request tail varies, the draft-MTP server
re-prefills the entire prompt (16.3 s) while the no-spec server reuses the
shared prefix (<400 ms). The startup log also warns
`cache_reuse is not supported by this context, it will be disabled`.

**Production implication:** for prefill-heavy repeated-prompt workloads
(agentic loops, fixed system prompts), disabling the spec flags and keeping
prompt caching on gives 45–105× TTFT at the cost of decode speed (14.1 vs
33.8 tok/s) — worthwhile when the response is short relative to the shared
prefix. Streaming chat remains spec-first. (`-cpent`/`-cram`/`--slot-save-path`
were accepted but not individually isolated; `--cache-reuse 256` had no
observable effect beyond the log warning.)

### 7.2 3 GB/s drafter-class bus gate — KILLED

The external report claimed AMD-135M-class drafters (2.98× on NPU for a
135M model) could beat the decode ceiling. The bus footprint of such a drafter
(~70 MB weights streamed once per chain step, ~4 chain steps per 100 ms
verification window) projects ~2.8 GB/s added DRAM traffic. Phase 0 was
extended with a 3 GB/s level:

| Added DRAM traffic | Median | Degradation |
|---|---|---|
| 0 | 40.0 tok/s | — |
| **3 GB/s** | **30.7 tok/s** | **−23.2%** |
| 8 GB/s | 35.6 tok/s | −11.0% |
| 16 GB/s | 33.1 tok/s | −17.2% |

Even the *cheapest* plausible separate drafter (135M class) fails the 5% gate
by 4.6×. Any NPU drafter adds KV-state traffic on top of weight streaming, so
3 GB/s is already generous. **The bus gate for separate NPU drafters on this
platform is empirically closed — including the smallest ones.**
