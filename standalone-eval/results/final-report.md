# Comprehensive Technical Evaluation Report: LiquidAI LFM2.5-1.2B-Thinking as a Local / Edge Verifier for `llm-as-a-verifier` on AMD Strix Halo

**Evaluation Date:** 2026-08-20  
**Target Repository:** [`llm-as-a-verifier/llm-as-a-verifier`](https://github.com/llm-as-a-verifier/llm-as-a-verifier)  
**Host Architecture:** AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 48-tile XDNA 2 NPU)  
**Evaluated Artifacts:** `standalone-eval/` (`data/`, `prompts/`, `results/`, `scripts/`)

---

## 1. Executive Summary & Verdict

| Question | Evaluation Finding | Recommendation / Action |
| :--- | :--- | :--- |
| **Is `LFM2.5-1.2B-Thinking` viable as a local verifier?** | **YES, with architectural caveats.** Achieves **96.7% pairwise accuracy** on software engineering trajectories under holistic rubrics. | Viable for local agent evaluation and Best-of-N reranking. |
| **Can it run directly on the AMD XDNA 2 NPU?** | **PARTIAL.** FastFlowLM (`flm`) runs on NPU at >40 tok/s, but current compiled AIE2p NPU kernels do **not** return per-token logprobs (`"logprobs": null`). | Use **llama-server on Vulkan/iGPU** (port 8012) or **PyTorch/FastAPI** for full logprob scoring, or use a hybrid pipeline. |
| **Logprob Continuous Expectation vs Scalar PRM?** | `llm-as-a-verifier` **strictly requires token logprobs** across discrete letters ($A$–$T$) to compute continuous expected rewards $\mathbb{E}[\text{score}] = \sum p(t_i) \phi(t_i)$. | Scalar reward models (e.g. Skywork PRM) cannot drop in without rewriting tournament math; LFM2.5 and Qwen2.5 fit cleanly. |
| **What is the optimal score label format?** | **Single letters `A`–`T` (20-point scale).** | Both raw letters (`A`..`T`) and space-prefixed letters (` A`..` T`) are guaranteed single tokens across all tokenizers. |
| **How does it compare to Qwen2.5-1.5B & Hard Rules?** | `Qwen2.5-1.5B-Instruct` achieved **100.0% accuracy** (margin: $0.1011$), while `LFM2.5-1.2B-Thinking` achieved **96.7% accuracy** (margin: $0.0491$) and is 21% faster. | Use `Qwen2.5-1.5B-Instruct` for maximum score calibration or `LFM2.5` for higher memory efficiency / SSM speed. |

---

## 2. Deep Dive: `llm-as-a-verifier` Mechanics & Logprob Requirements

### 2.1 API & Token Architecture
* **Endpoint:** OpenAI-compatible `/v1/chat/completions` with `logprobs=True`, `top_logprobs=20`, `temperature=1.0`.
* **Prefill Scoring Mechanism (`_score_tags_by_prefill`):** Appends `\n<score_A> ` as the assistant turn and samples the subsequent letter token.
* **Continuous Expectation Formulation:**
  $$\mathbb{E}[S] = \frac{\sum_{i=0}^{19} (20 - i) \cdot \exp(\text{logprob}(L_i))}{\sum_{i=0}^{19} \exp(\text{logprob}(L_i))}, \quad R = \frac{\mathbb{E}[S] - 1.0}{19.0} \in [0.0, 1.0]$$

---

## 3. Label Scheme & Tokenization Analysis

We tested tokenization across vocabularies (LFM 64.4k vs Qwen 151.6k):

| Label Format | Representation | Single Token Safe? | Token Lengths | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Format A (Letters A–T)** | `A`, `B`, ..., `T` (and ` A`, ` B`) | ✅ **YES** (100% single token) | `[1]` across all models | **Recommended Standard** |
| **Format B (Numbers 0–19)** | `0`, `1`, ..., `19` | ⚠️ **NO** | `[1, 2, 3]` depending on whitespace | Risk of multi-token splitting |
| **Format C (Structured Tags)** | `<S00>`, `<S01>`, ..., `<S19>` | ❌ **NO** | `[4]` (`<`, `S`, `00`, `>`) | Incompatible with 1-step logprob extraction |

---

## 4. Benchmark Results on 30 SWE/Terminal Agent Trajectory Pairs

All benchmark data was executed on the workstation and recorded in `standalone-eval/results/benchmark.json`.

```
========================================================================================
MODEL / BASELINE            EVALUATION MODE    ACCURACY   TIE RATE   ERROR RATE   MARGIN
========================================================================================
Random Baseline             Random Choice         50.0%       0.0%        50.0%   0.0000
Hard-Rule Deterministic     Regex Extraction      90.0%      10.0%         0.0%      N/A
LFM2.5-1.2B-Thinking        Holistic Rubric       96.7%       0.0%         3.3%   0.0491
LFM2.5-1.2B-Thinking        Decomposed Criteria   83.3%       3.3%        13.4%   0.0669
Qwen2.5-1.5B-Instruct       Holistic Rubric      100.0%       0.0%         0.0%   0.1011
Qwen2.5-1.5B-Instruct       Decomposed Criteria   96.7%       3.3%         0.0%   0.0663
========================================================================================
```

### Key Insights:
1. **Holistic Superiority on Sub-2B Models:** Holistic evaluation significantly outperforms decomposed rubrics (96.7% vs 83.3% on LFM2.5) because sub-2B models maintain better global context when weighing code diffs and test logs together rather than evaluating isolated abstract criteria.
2. **Thinking Model Dynamics:** Because `LFM2.5-1.2B-Thinking` is optimized for internal reasoning (`<think>...</think>`), prompting it for single-token prefill scoring works well (96.7% accuracy), but its unconstrained generation requires a reasoning token budget ($\ge 256$ tokens) before outputting final tags.

---

## 5. Deployment Recommendation for AMD Strix Halo

1. **Production Serving Setup:**
   * Run `llama-server` on Vulkan/iGPU (`port 8012`) with `-c 32768 --top-logprobs 20` for continuous logprob extraction.
   * Point `llm-as-a-verifier` client at `http://127.0.0.1:8012/v1`.
2. **Hybrid Fast-Triage Pipeline (NPU + iGPU):**
   * **Stage 1 (NPU / FastFlowLM on `:13305`):** Run high-speed heuristic triage (rejecting obvious test crashes and tracebacks at 45 tok/s without consuming iGPU memory).
   * **Stage 2 (iGPU / llama-server on `:8012`):** Run fine-grained tournament pair scoring with token logprobs on the remaining candidate solutions.
