# Capacity & Memory Planning: Qwen3.8-Flash-Next

**Target Hardware Profiles:**
1. **AMD Strix Halo:** Ryzen AI Max+ 395, 128 GB LPDDR5X-8000 Unified Memory (~124.9 GiB usable, ~120 GB safe ceiling).
2. **Discrete GPU Cluster:** 4× NVIDIA GeForce RTX 3090 (4 × 24 GB = **96.0 GB VRAM** total).
3. **Hybrid Setup:** 4× RTX 3090 (96 GB VRAM) for Dense MoE Compute + AMD Strix Halo (128 GB Host RAM) for Table Offload.

**Leaked Architecture Specification:**
- **Main Dense/MoE Backbone:** ~125B Total Parameters (~6B Active per token, ~61 Layers).
- **N-Gram Lookup Memory Table:** ~51B Total Parameters ($K=8$ heads, $N_{\max}=3$ to $4$, $D=64$–$128$).

---

## 1. Master Capacity Planning Matrix

| Scenario | Main MoE Format (125B) | N-Gram Table Format (51B) | Main MoE Footprint | Table Footprint | 16K KV Cache (MLA) | OS & Runtime Headroom | Total Memory Required | Strix Halo (128 GB UMA) Status | 4x RTX 3090 (96 GB VRAM) Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **Q4_K_M** (~4.5 bpw) | **INT4** (0.5 B/p) | **58.0 GB** | **25.5 GB** | **1.5 GB** | **6.0 GB** | **91.0 GB** | 🟢 **FEASIBLE (+29.0 GB Free)** | 🟢 **FEASIBLE (GPU: 59.5 GB, Table in RAM: 25.5 GB)** |
| **B** | **Q4_K_M** (~4.5 bpw) | **INT8 / FP8** (1.0 B/p) | **58.0 GB** | **51.0 GB** | **1.5 GB** | **6.0 GB** | **116.5 GB** | 🟡 **TIGHT (+3.5 GB Free, <8GB warning)** | 🟢 **FEASIBLE (GPU: 59.5 GB, Table in RAM: 51.0 GB)** |
| **C** | **Q4_K_M** (~4.5 bpw) | **FP16 / BF16** (2.0 B/p) | **58.0 GB** | **102.0 GB** | **1.5 GB** | **6.0 GB** | **167.5 GB** | 🔴 **EXCEEDS 128 GB (Impossible on single node)** | 🟡 **FEASIBLE ONLY with 128GB Host RAM Offload** |
| **D** | **Q5_K_M** (~5.5 bpw) | **INT4** (0.5 B/p) | **85.0 GB** | **25.5 GB** | **1.5 GB** | **6.0 GB** | **118.0 GB** | 🟡 **TIGHT (+2.0 GB Free)** | 🟢 **FEASIBLE (GPU: 86.5 GB, Table in RAM: 25.5 GB)** |
| **E** | **Q5_K_M** (~5.5 bpw) | **INT8** (1.0 B/p) | **85.0 GB** | **51.0 GB** | **1.5 GB** | **6.0 GB** | **143.5 GB** | 🔴 **EXCEEDS 128 GB** | 🟡 **FEASIBLE (GPU: 86.5 GB, Table in Host RAM: 51 GB)** |
| **F** | **Q6_K** (~6.5 bpw) | **INT4** (0.5 B/p) | **100.0 GB** | **25.5 GB** | **1.5 GB** | **6.0 GB** | **133.0 GB** | 🔴 **EXCEEDS 128 GB** | 🔴 **EXCEEDS GPU VRAM (100 GB > 96 GB)** |
| **G** | **Q8_0** (~8.5 bpw) | **INT4** (0.5 B/p) | **131.0 GB** | **25.5 GB** | **1.5 GB** | **6.0 GB** | **164.0 GB** | 🔴 **EXCEEDS 128 GB** | 🔴 **EXCEEDS GPU VRAM** |
| **H** | **FP16** (16.0 bpw) | **FP16** (2.0 B/p) | **250.0 GB** | **102.0 GB** | **1.5 GB** | **6.0 GB** | **359.5 GB** | 🔴 **EXCEEDS 128 GB** | 🔴 **EXCEEDS CLUSTER** |

---

## 2. Granular Component Breakdown

### 2.1 Main MoE Backbone (125B Parameters)
- Total parameter count: ~125B.
- Active parameter count per token: ~6B (sparse MoE activation).
- **Quantization Footprints:**
  - `Q4_K_M` (4.50 bits/param): **58.0 GB** (recommended default)
  - `Q5_K_M` (5.50 bits/param): **85.0 GB**
  - `Q6_K` (6.56 bits/param): **100.0 GB**
  - `Q8_0` (8.50 bits/param): **131.0 GB**
  - `FP16 / BF16` (16.0 bits/param): **250.0 GB**

### 2.2 51B N-Gram Lookup Table Footprint
- **INT4 (0.5 bytes/param):** **25.5 GB** (4-bit linear or vector quantization).
- **INT8 / FP8 (1.0 byte/param):** **51.0 GB**.
- **FP16 / BF16 (2.0 bytes/param):** **102.0 GB** (🚨 **Flagged as the likely worst-case default** in initial upstream Hugging Face weight releases).

### 2.3 KV Cache Calculation at 16K Context
Using DeepSeek / Qwen Multi-Head Latent Attention (MLA) architecture ($d_c = 512, d_R = 64$, 61 layers):
$$\text{Memory per Token} = 61 \times (512 + 64) \times 2 \text{ bytes} = 70,272 \text{ bytes} \approx 70.3 \text{ KB/token}$$
$$\text{Total 16K KV Cache} = 16,384 \times 70,272 \text{ bytes} = 1,151,336,448 \text{ bytes} \approx \mathbf{1.15 \text{ GB}} \ (\text{or } \mathbf{1.5 \text{ GB}} \text{ with overhead})$$

---

## 3. Hardware Deployment Configurations

### 3.1 Scenario 1: Standalone AMD Strix Halo (128 GB Unified Memory)
- **Feasible Configurations:**
  - `Q4_K_M MoE (58 GB) + INT4 Table (25.5 GB)` $\implies$ **91.0 GB Total** (**+29.0 GB safe headroom**).
  - `Q4_K_M MoE (58 GB) + INT8 Table (51.0 GB)` $\implies$ **116.5 GB Total** (**+3.5 GB headroom** — viable with minimal desktop/OS footprint).
- **Infeasible Configurations:**
  - Any configuration where the table is unquantized BF16 (102 GB) or where main weights are Q6/Q8.

### 3.2 Scenario 2: 4× RTX 3090 Cluster (96 GB VRAM)
- **Pure GPU Residency:**
  - `Q4_K_M MoE (58 GB) + INT4 Table (25.5 GB) + 16K KV (1.5 GB)` = **85.0 GB VRAM** $\implies$ **Fits 100% inside 96 GB VRAM** (21.25 GB per GPU across 4 GPUs).
- **GPU + Host RAM Split (Recommended for higher table precision):**
  - **4x3090 VRAM (96 GB):** Hosts Q4/Q5 MoE weights (58–85 GB) + KV Cache (1.5 GB).
  - **Host RAM (128 GB):** Hosts the 51B N-gram Table in INT8 (51 GB) or BF16 (102 GB).
  - Asynchronous PCIe prefetch transfers the 2 KB retrieved vector per token with <0.15 $\mu\text{s}$ transfer latency.

---

## 4. Key Deployment Rules

1. **The BF16 Table Red Line:** If the official Qwen3.8-Flash-Next checkpoint ships with an unquantized BF16 N-gram table (102 GB) and no quantized INT4/INT8 GGUF/Safetensors table is available on Day 0, **it CANNOT run on a single 128 GB Strix Halo node**.
2. **Immediate Quantization Action:** Day-0 preparation must include an immediate quantization script to convert the 51B embedding table from BF16 $\to$ INT4 / INT8.
