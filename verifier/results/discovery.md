# Phase 0: Discovery & Baseline Profiling Report

**Workstation:** AMD Strix Halo (Ryzen AI Max+ 395, 128 GB UMA LPDDR5X-8000, Radeon 8060S iGPU, 48-tile XDNA 2 NPU)  
**Date:** 2026-08-20  

---

## 1. Serving Architecture & Endpoints

| Role | Model | Backend Engine | Device / Offload | Endpoint / Port | Streaming Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Generator** | `Ornith-1.5-35B-A3B-ROCmFP4` | `llama-server` (b215) | Radeon 8060S (Vulkan0, 99 layers) | `http://127.0.0.1:8012/v1` | ✅ SSE (`text/event-stream`) |
| **Verifier** | `LiquidAI/LFM2.5-1.2B-Thinking` | FastFlowLM / PyTorch | AIE2p NPU (`/dev/accel/accel0`) / In-process | `http://127.0.0.1:13305/v1` / Local | ✅ Token generation |

---

## 2. Generator Throughput (Ornith-1.5 Alone)

* **Decode Speed:** **72.04 tok/s**
* **Time To First Token (TTFT):** **151.4 ms**
* **Prompt Cache:** Enabled (8192 MiB RAM prompt cache limit)
* **Concurrency:** 4 server slots configured (`-c 16384`)

---

## 3. Verifier Latency vs Checkpoint Granularity (LFM2.5-1.2B)

| Target Chunk Size | Actual Input Tokens | Output Tokens | Mean Verdict Latency | Effective Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **150 tokens** | 193 | 15 | **3222.0 ms** | **64.6 tok/s** |
| **300 tokens** | 343 | 15 | **3489.1 ms** | **102.6 tok/s** |
| **600 tokens** | 643 | 15 | **4241.1 ms** | **155.1 tok/s** |

---

## 4. Key Takeaways & Checkpoint Sizing

1. **Generation vs. Verification Window:**
   * At 72.04 tok/s, generating a 250-token chunk takes **~3.47 seconds** on the GPU.
   * LFM2.5 evaluates a 300-token checkpoint in **~3.49 seconds**.
   * This yields a **near 1:1 balance** between GPU generation time and asynchronous verification time, meaning the verifier can run continuously without falling significantly behind or stalling the generation stream.
2. **Streaming Protocol:**
   * SSE chunks deliver `delta.content` and `delta.reasoning_content` in real time, enabling the stream tap to detect syntax and structural boundaries with zero buffer delays.
