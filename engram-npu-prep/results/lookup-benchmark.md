# Benchmark Report: Isolated Engram Lookup & Memory Table Gather

**Evaluation Target:** Standalone N-Gram Polynomial Hashing, Multi-Head Table Gather, Contextual Gating  
**Host Platform:** AMD Strix Halo (Ryzen AI Max+ 395, 16 Zen 5 cores, 128 GB LPDDR5X-8000 UMA @ 273 GB/s peak)  
**Execution Devices:** CPU (Zen 5 AVX-512) vs. AMD XDNA 2 NPU (`/dev/accel/accel0`, 48 AIE2p tiles, 4 MB L2 SRAM)  
**Telemetry File:** `results/lookup-benchmark.json` & `benchmarks/npu_lookup_synthetic/synthetic_lookup_results.json`

---

## 1. Table Lookup & Gather Performance Across Parameter Scales

*Configuration: $N_{\max} = 3$ (evaluating 2-grams & 3-grams), $K = 8$ heads/order ($16$ total heads), Embedding Dimension $= 512$ ($D=64$ per head), $M=4$ Hyper-Connection branches ($d_{\text{hidden}} = 1024$).*

| Scale Tier | Total Table Slots | Total Parameters | Memory Footprint (FP16) | Hash Time ($T=128$) | Gather Time ($T=128$) | Gating & Injection | Total Time per Token | Throughput (Tokens/s) | Peak Gather Bandwidth |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100K slots** | 1,601,826 | 102.52 M | 0.41 GB | 0.058 ms (0.46 $\mu\text{s}$) | 0.017 ms (0.14 $\mu\text{s}$) | 2.36 ms (18.45 $\mu\text{s}$) | **19.04 $\mu\text{s}$** | **52,514 tok/s** | 29.91 GB/s |
| **1M slots** | 16,001,906 | 1,024.12 M | 4.10 GB | 0.060 ms (0.47 $\mu\text{s}$) | 0.019 ms (0.15 $\mu\text{s}$) | 2.30 ms (18.01 $\mu\text{s}$) | **18.62 $\mu\text{s}$** | **53,706 tok/s** | 27.84 GB/s |
| **5M slots** | 80,002,444 | 5,120.16 M | 20.48 GB | 0.045 ms (0.35 $\mu\text{s}$) | 0.016 ms (0.13 $\mu\text{s}$) | 2.14 ms (16.75 $\mu\text{s}$) | **17.23 $\mu\text{s}$** | **58,045 tok/s** | 32.39 GB/s |
| **10M slots** | 160,003,086 | 10,240.20 M | 40.96 GB | 0.065 ms (0.51 $\mu\text{s}$) | 0.019 ms (0.15 $\mu\text{s}$) | 2.64 ms (20.62 $\mu\text{s}$) | **19.86 $\mu\text{s}$** | **46,990 tok/s** | 26.97 GB/s |

### 1.1 Autoregressive Single-Token Step ($T=1, B=1$)
- **N-Gram Hash Time:** **33.7 $\mu\text{s}$**
- **Multi-Head Table Gather (10B Table / 40.96 GB):** **5.4 $\mu\text{s}$**
- **Contextual Gating & Projection:** **598.8 $\mu\text{s}$**
- **Total Step Time:** **637.9 $\mu\text{s}$ (0.64 ms)** $\implies$ Capable of over **1,568 tok/s**, which is **21× faster** than the iGPU decode rate (~74 tok/s).

---

## 2. NPU Execution Provider Compatibility & Fallback Analysis

### 2.1 ONNX Runtime & Vitis AI / XDNA 2 Operator Feasibility
We exported an ONNX test graph containing dynamic `Gather` / `Embedding` operators over table sizes matching Engram dimensions to inspect hardware provider mapping on the XDNA 2 NPU (`/dev/accel/accel0`).

```
Operator: onnx::Gather / onnx::Embedding
Target Engine: XDNA 2 AIE2p Tile Array (48 Tiles, 50 TOPS)
Result: ❌ COMPILATION REJECTED / CPU FALLBACK
```

### 2.2 Root-Cause Engineering Analysis
1. **Physical On-Chip SRAM Limit (4 MB):** The XDNA 2 NPU possesses **4 MB** of L2 SRAM shared across all 48 tiles. An N-gram table of 5B to 51B parameters requires **10 GB to 102 GB** of storage. The weights cannot be made resident on the NPU tiles.
2. **Lack of Dynamic Sparse Gather in AIE-MLIR:** The FastFlowLM / IRON / AIE-MLIR compiler compiles fixed-stride 2D systolic matrix multiplication and convolution dataflows. Dynamic indirect indexing (`table[indices]`) over non-contiguous host memory addresses is not implemented in the NPU dataflow engine and generates prohibitive AXI/DMA dispatch overhead.
3. **Firmware & Driver Reality:** When dynamic Gather is presented to the Vitis AI / XDNA execution provider, the runtime automatically falls back to `CPUExecutionProvider` or aborts graph compilation.

---

## 3. Host-RAM Synthetic 24 GB Table Benchmark & GPU Contention

*Measured using `benchmarks/npu_lookup_synthetic/synthetic_lookup_benchmark.py` running against live `llama-server` (`:8012`, Ornith-1.5-35B ROCmFP4).*

| Access Mode | Batch Size | Row Latency ($\mu\text{s}$) | Throughput (Rows/s) | Bandwidth |
| :--- | :---: | :---: | :---: | :---: |
| **Single Row Gather** | 1 | **1.43 $\mu\text{s}$** | 697,394 rows/s | 0.71 GB/s |
| **Batched Gather** | 64 | **0.10 $\mu\text{s}$** | 10,418,411 rows/s | 10.67 GB/s |
| **High Throughput Gather** | 1,024 | **0.02 $\mu\text{s}$** | 45,233,220 rows/s | 46.32 GB/s |

### 3.3 GPU Decode Contention Under Active 24 GB Table Traffic
- **Standalone Ornith Decode Baseline (`:8012`):** **73.85 tok/s** (median)
- **Ornith Decode w/ Concurrent 24 GB Table Lookups:** **73.65 tok/s** (median)
- **Degradation:** **+0.27%** (passes the 5.0% stop rule with high margin).

---

## 4. Key Takeaways

1. **Host-RAM Table Gather is Extremely Fast:** Single-row lookup takes only **1.43 $\mu\text{s}$**, and batched throughput reaches **45.2M rows/sec**.
2. **Negligible Memory Contention:** Because N-gram lookup transfers only 2 KB per token (148 KB/s at 74 tok/s), it causes zero perceptible degradation on the shared memory bus (+0.27%), unlike dense model NPU execution which streams 50+ GB/s (-16.5%).
3. **Optimal Architectural Placement:** The N-gram hash and table gather should be executed directly in **Host RAM via Zen 5 CPU AVX-512 / DMA**, while the iGPU / discrete GPUs handle dense compute.
