# Benchmark Results: Live Streaming Verification vs Baseline & Streaming BoN

> **Status: preliminary benchmark, superseded for deployment decisions by
> [`final-report.md`](final-report.md).** Later live measurements found −8.2%
> steady-state overhead, not the preliminary <3.5% estimate below.

**Platform:** AMD Strix Halo (Ryzen AI Max+ 395, 128 GB UMA LPDDR5X-8000, Radeon 8060S iGPU, 48-tile XDNA 2 NPU)  
**Generator:** `Ornith-1.5-35B-A3B-ROCmFP4` (1.5B active MoE params, served on `llama-server` port 8012)  
**Verifier:** `LiquidAI/LFM2.5-1.2B-Thinking` (1.2B Hybrid Conv/Transformer verifier)  
**Available datasets:** Set A (30 math tasks), Set B (20 Python tasks), Set C (20 planted-error trajectories). **Executed sample in the table below:** 10 Set A tasks and 10 Set B tasks; all 20 Set C trajectories were evaluated separately.

---

## 1. Executive Summary Table

| Metric | BASELINE (Unverified) | LIVE STREAMING VERIFICATION | STREAMING BoN ($N=4$) |
| :--- | :--- | :--- | :--- |
| **Math Accuracy (Set A)** | 90.0% (9/10) | 90.0% (9/10) | 90.0% (9/10) |
| **Python Code Pass@1 (Set B)** | 50.0% (5/10) | 50.0% (5/10) | 50.0% (5/10) |
| **Overall Pass Rate** | **70.0%** (14/20) | **70.0%** (14/20) | **70.0%** (14/20) |
| **Total GPU Generation Tokens** | 7,250 tokens | 7,217 tokens | 29,565 tokens |
| **Tokens per Solved Task** | 517.9 tokens/solve | **515.5 tokens/solve** | 2,111.8 tokens/solve |
| **Total Wallclock Execution Time** | 192.16 s | 192.66 s | 816.25 s |
| **Streaming Pipeline Overhead** | 0.00 ms (Direct yield) | +0.0095 ms / token | N/A (Parallel stream pool) |
| **Memory Contention Degradation** | 0.0% | **< 3.5%** | **~12.1%** |

---

## 2. Planted Error Detection & Abort Precision (Set C)

Evaluated against 20 controlled trajectories (10 synthetic bugs / hallucinations / division-by-zero, 10 ground-truth valid checkpoints):

- **True Positives (Correctly aborted invalid code/reasoning):** 10 / 10
- **False Positives (Erroneously aborted clean code):** 0 / 10
- **True Negatives (Allowed clean steps to proceed):** 10 / 10
- **False Negatives (Missed bugs):** 0 / 10
- **Abort Precision:** **100.0%** ($\ge 90\%$ target achieved)
- **Abort Recall:** **100.0%**

---

## 3. Systems & Latency Profiling

- **Generator Native Throughput (Radeon 8060S):** 72.04 tok/s (TTFT = 151.4 ms)
- **Verifier Latency per Checkpoint:**
  - 150-token chunk: 3,222 ms
  - 300-token chunk: 3,489 ms
  - 600-token chunk: 4,241 ms
- **Checkpoint Synchronization Balance:**
  - Generating 250 tokens on GPU takes: $250 / 72.04 = 3.47\text{ s}$
  - Verifying 250 tokens asynchronously takes: $\approx 3.48\text{ s}$
  - **Verdict:** Verifier lag matches GPU generation speed almost exactly (lag ratio $\approx 1.00$).

---

## 4. Resource & Contention Profile

1. **LPDDR5X Memory Bandwidth Contention:**
   - Under single-stream LIVE verification, the generator throughput dipped by only **2.8% to 3.5%** during NPU/CPU verifier prefill phases (well below the 15% degradation limit).
   - Under 4-way parallel Streaming BoN, concurrent memory access pressure created an aggregate **12.1% throughput dip** on the iGPU.

2. **Quality per GPU Token:**
   - Single-stream LIVE verification achieves equal solution quality to unverified generation while saving GPU tokens on invalid runs (**515.5 vs 517.9 tokens/solve**).
   - Streaming BoN ($N=4$) consumes $4.08\times$ more GPU compute without raising pass rate beyond the generator's intrinsic capacity limit.
