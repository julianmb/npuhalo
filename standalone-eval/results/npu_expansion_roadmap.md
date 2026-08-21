# Verification Pipeline Expansion & NPU Adapter Roadmap

**Host:** AMD Strix Halo (Ryzen AI Max+ 395, 128 GB UMA, 48-tile XDNA 2 NPU)  
**Date:** 2026-08-20  
**Associated Tools:** [`npu_logit_adapter.py`](../scripts/npu_logit_adapter.py), [`position_bias_and_calibration.py`](../scripts/position_bias_and_calibration.py)

---

## 1. Production Multi-Stage Deployment Architecture

```
                               +---------------------------------------------+
                               |              30B Agent Model                |
                               |    (Main Discrete GPU / ROCm PyTorch)       |
                               |   Generates K candidate agent traces/diffs  |
                               +---------------------------------------------+
                                                      |
                                                      v
                               +---------------------------------------------+
                               |         STAGE 1: NPU Fast Screening         |
                               |      Model: LiquidAI LFM2.5-1.2B-Thinking   |
                               |      Engine: FastFlowLM (/dev/accel/accel0) |
                               |      Speed: ~43 tok/s | Memory: <2 GB       |
                               |      Task: Fast rejection of merge conflicts|
                               |            syntax errors, and tracebacks    |
                               +---------------------------------------------+
                                                      | (Top 3-4 surviving traces)
                                                      v
                               +---------------------------------------------+
                               |          STAGE 2: Final Pairwise Ranker     |
                               |      Model: Qwen2.5-1.5B-Instruct           |
                               |      Engine: llama-server (iGPU Vulkan :8012)|
                               |      Logprobs: Full A-T 20-token logits     |
                               |      Task: Continuous expected score        |
                               |            tournament selection             |
                               +---------------------------------------------+
```

---

## 2. NPU Final-Step Logit Extractor Adapter

To migrate Stage 2 directly to the NPU once FastFlowLM exposes final-step logits, we built [`npu_logit_adapter.py`](../scripts/npu_logit_adapter.py):

* **Mechanism:** Queries unnormalized logits $z$ at the `<score_A>` rating position across the 20 letter token IDs.
* **Math:**
  $$P(t_i) = \frac{\exp(z_i)}{\sum_{j \in \{A..T\}} \exp(z_j)}, \quad \mathbb{E}[S] = \sum_{i \in \{A..T\}} P(t_i) \phi(t_i)$$
* **OpenAI Payload Formatter:** Serializes the computed distribution into standard `choices[0].logprobs.content[0].top_logprobs` matching `llm-as-a-verifier`'s exact schema.

---

## 3. Large-Scale Expansion Protocol (200–500 Real Trajectories)

1. **Artifact Collection:** Ingest real SWE-Bench Verified and Terminal-Bench agent run traces (uncompressed git diffs, stdin/stdout traces, and exit codes).
2. **Best-of-N Simulation:** Evaluate verifier selection across $N \in \{4, 8, 16, 32\}$ candidates to measure true Pass@1 uplift:
   $$\Delta \text{Pass@1} = \text{Pass@1}_{\text{verifier}} - \text{Pass@1}_{\text{raw}}$$
3. **Decile Calibration:** Monitor probability bins $[0.0, 0.2), \dots, [0.8, 1.0]$ against hidden oracle execution outcomes.
