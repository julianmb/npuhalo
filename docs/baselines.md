# Strix Halo Baseline Benchmarks — Qwen 3.8 27B & NPU Drafter

This document records the empirical baseline measurements established on this workstation (AMD Ryzen AI Max+ 395 w/ Radeon 8060S iGPU & XDNA 2 NPU):

## 1. Hardware & System Configuration
* **Processor:** AMD Ryzen AI Max+ 395 (16 Zen 5 cores, 32 threads)
* **iGPU:** AMD Radeon 8060S (40 CUs, RDNA 3.5, `gfx1151`)
* **NPU:** AMD XDNA 2 / `RyzenAI-npu5` (48 AIE2p tiles @ 50 TOPS, `/dev/accel/accel0`)
* **NPU Firmware:** `1.1.2.65` (Active in `/lib/firmware/amdnpu/17f0_11/npu.sbin`)
* **Unified Memory:** 128 GB LPDDR5X-8000 (256-bit bus, ~273 GB/s peak bandwidth)
* **Kernel & OS:** Linux 7.0.0-28-generic (SVA enabled, `iommu=pt`)
* **Graphics Translation Table (GTT):** Dynamic OS-level allocation (`128 GiB` max)

---

## 2. Real Measured NPU Performance (`qwen3.5-0.8b-FLM`)
*Engine:* FastFlowLM (`flm` v0.9.46) on `/dev/accel/accel0`  
*Power:* ~2 Watts (NPU+CPU)  
*VRAM Allocated:* 0.2 GB (runs entirely on-die)

| Scenario | Time-To-First-Token (TTFT) | Generation Speed (TPS) | VRAM Footprint |
|---|---|---|---|
| **Chat Short** | **347.2 ms** | **38.8 tok/s** | 0.2 GB |
| **Code Short** | **349.2 ms** | **41.5 tok/s** | 0.2 GB |
| **Sustained Inter-Token Streaming** | **348.0 ms** | **42.3 tok/s** (23.6 ms/tok) | 0.2 GB |

---

## 3. Real Measured iGPU Performance (`Qwen3.8-27B-ROCmFP4-FAST.gguf`)
*Engine:* `ROCmFPX` (build `b213-e87d53e`) with `RADV STRIX_HALO` Vulkan `KHR_coopmat` Wave64 matrix kernels.

| Configuration | Prefill (Prompt) Speed | Decode (Generation) Speed | Memory Bus Contention | Note |
|---|---|---|---|---|
| **Standalone iGPU (No MTP)** | 101.4 tok/s | 14.1 tok/s | Baseline (~13.55 GB/tok sweep) | Standard autoregressive decode |
| **iGPU + Embedded MTP (K=6)** | 74.6 tok/s | **22.8 – 24.7 tok/s** | Zero secondary model overhead | Single forward pass verification |
| **llama-bench Raw Prefill** | **371.53 ± 0.98 tok/s** | 13.65 tok/s | 100% iGPU saturation | Pure GEMM burst |

---

## 4. Zero-Copy IPC Performance (`scripts/ipc/ring_buffer.py`)
* **Transport:** Direct POSIX shared memory (`/dev/shm/npuhalo_token_ring`)
* **Latency per Token:** **0.424 µs (microseconds)**
* **Throughput:** **2,357,485 tokens/second**
* **PCIe/Bus Copy Overhead:** **0.00 ns**
