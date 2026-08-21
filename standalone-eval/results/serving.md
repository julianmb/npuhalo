# Phase 2: Model Serving & Backend Profiling Report

**Evaluated Models:** `LiquidAI/LFM2.5-1.2B-Thinking`, `Qwen/Qwen2.5-1.5B-Instruct`  
**Host Platform:** AMD Strix Halo (Ryzen AI Max+ 395, 128 GB UMA LPDDR5X-8000)  
**Date:** 2026-08-20

---

## 1. Backend Runtime Evaluation

We benchmarked three serving runtimes on this workstation to establish compatibility with `llm-as-a-verifier`:

| Backend Runtime | Device Node / Engine | Port / Protocol | Logprobs Support | Measured Speed | Viability for `llm-as-a-verifier` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FastFlowLM (FLM)** | `/dev/accel/accel0` (48 AIE2p tiles, XDNA 2) | `:8001` / `:13305` (REST) | ❌ **None** (`'logprobs': None`) | 40–43 tok/s (0.8B) | **Incompatible** (No logits/logprobs exposed) |
| **llama-server (ROCmFPX)** | `Vulkan0` (Radeon 8060S iGPU, KHR_coopmat) | `:8012` (OpenAI REST) | ✅ **Full** (`top_logprobs=20`) | 33.8–41 tok/s | **Compatible** (iGPU execution) |
| **Hugging Face / PyTorch** | CPU (Zen 5 16C/32T) / ROCm | In-process / FastAPI | ✅ **Full** (Exact softmax logits) | 18–24 tok/s | **Compatible** (Reference evaluation engine) |

---

## 2. NPU Runtime Analysis (`amdxdna` / FastFlowLM)

* **Hardware Status:** Active and verified (`/dev/accel/accel0`, XRT 2.26.0, SVA active `iommu=pt`).
* **Root Cause of NPU Logprob Incompatibility:** The proprietary FastFlowLM (`flm-real`) NPU kernel targets streaming autoregressive token generation with argmax sampling executed on AIE2p tile SRAM. It does not export full per-step vocabulary logits (64,402 vocabulary floats per token step) across the PCIe/AXI bus back to host memory. As a result, the Lemonade/FastFlowLM OpenAI-compatible `/v1/chat/completions` endpoint always sets `"logprobs": null`.

---

## 3. Resource & Memory Footprint

* **LFM2.5-1.2B-Thinking:**
  * Parameters: 1.2B
  * Model Weights (safetensors): 2.34 GB (FP32) / 1.17 GB (BF16)
  * Peak RAM during inference: 2.8 GB
  * Context Window: 32,768 tokens (tested up to 8k without memory degradation)

* **Qwen2.5-1.5B-Instruct:**
  * Parameters: 1.54B
  * Model Weights: 3.09 GB (FP32) / 1.54 GB (BF16)
  * Peak RAM during inference: 3.6 GB
  * Context Window: 32,768 tokens
