# Phase 5: Pairwise Trajectory Benchmark Results

**Evaluation Dataset:** 30 Curated Software Engineering / Terminal Agent Pairs (`standalone-eval/data/pairs.jsonl`)  
**Hardware Host:** AMD Strix Halo (Ryzen AI Max+ 395, 128 GB UMA)  
**Evaluation Date:** 2026-08-20  
**Machine-Readable Data:** `standalone-eval/results/benchmark.json`

---

## 1. Comparative Performance Matrix

| Model / Baseline | Evaluation Mode | Pairwise Accuracy | Tie Rate | Error / Inversion Rate | Mean Margin ($\Delta$) | Latency / Call | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Chance** | N/A | 50.0% (15/30) | 0.0% | 50.0% | 0.0000 | 0.0 ms | N/A |
| **Hard-Rule Baseline** | Regex Heuristics | 90.0% (27/30) | 10.0% (3/30) | 0.0% | N/A | 0.1 ms | >100k tok/s |
| **LFM2.5-1.2B-Thinking** | **Holistic Rubric** | **96.7% (29/30)** | **0.0%** | **3.3% (1/30)** | **0.0491** | **410.2 ms** | **300.9 tok/s** |
| **LFM2.5-1.2B-Thinking** | **Decomposed Criteria** | **83.3% (25/30)** | **3.3% (1/30)** | **13.4% (4/30)** | **0.0669** | **417.2 ms** | **286.2 tok/s** |
| **Qwen2.5-1.5B-Instruct** | **Holistic Rubric** | **100.0% (30/30)** | **0.0%** | **0.0%** | **0.1011** | **519.9 ms** | **249.2 tok/s** |
| **Qwen2.5-1.5B-Instruct** | **Decomposed Criteria** | **96.7% (29/30)** | **3.3% (1/30)** | **0.0%** | **0.0663** | **544.8 ms** | **228.6 tok/s** |

---

## 2. Granular Analysis & Diagnostic Observations

1. **Holistic vs. Decomposed Criteria:**
   * For both sub-2B models, the **Holistic Prompt** outperformed the Decomposed Criteria (96.7% vs 83.3% on LFM2.5; 100% vs 96.7% on Qwen2.5).
   * *Reason:* Decomposing into 4 separate criteria forces the small 1.2B model to score narrow sub-aspects (such as "Contradiction Signals" on a positive trace) where lack of negative keywords leads to ambiguous letter logits. Holistic evaluation allows the model to synthesize code changes and execution test logs simultaneously.
2. **Confidence Margin & Separation:**
   * `Qwen2.5-1.5B-Instruct` displayed over $2\times$ greater confidence margin ($0.1011$ vs $0.0491$) compared to `LFM2.5-1.2B-Thinking`.
   * LFM2.5 tends to have a more uniform logit spread across the top tokens without explicit chain-of-thought generation, whereas Qwen2.5's instruction tuning concentrates probability mass decisively on high-scoring letters for passing traces.
3. **Throughput & Latency:**
   * LFM2.5 achieved **300.9 tok/s** evaluation throughput with an average call latency of **410.2 ms**, making it ~21% faster than Qwen2.5-1.5B due to its hybrid SSM/Conv/Attention architecture.
