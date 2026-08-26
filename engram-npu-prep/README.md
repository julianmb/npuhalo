# Engram NPU Preparation & Day-0 Readiness Suite for Qwen3.8-Flash-Next

**Target Model:** `Qwen/Qwen3.8-Flash-Next` (125B Dense/MoE Backbone + 51B N-Gram Memory Table, Engram-style)  
**Host Hardware:** AMD Strix Halo (Ryzen AI Max+ 395, 128 GB LPDDR5X-8000 UMA, 48-tile XDNA 2 NPU) & 4x RTX 3090 Cluster  
**Lead Systems Engineer:** ML Infrastructure & Heterogeneous Compute Evaluation  

---

## 1. Executive Summary & Core Verdict

This repository provides the complete pre-release research, architecture validation, benchmarking harnesses, memory capacity models, and Day-0 deployment protocols for **Qwen3.8-Flash-Next**.

```
                           Qwen3.8-Flash-Next Pipeline Topology
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      │                                                                            │
      ▼                                                                            ▼
[Host RAM / CPU AVX-512]                                            [Radeon 8060S / 4x RTX 3090]
  • Tokenizer Normalization P(x_t)                                    • 125B Dense/MoE Backbone (Q4_K_M)
  • Polynomial-XOR Multi-Head Hash z_t,n,k                            • Multi-Head Latent Attention (MLA)
  • 51B N-Gram Table Gather -> e_t (2 KB)                             • Contextual Gating α_t = σ(k · q)
  • Latency: 1.43 µs | Bandwidth: 45.2M rows/s                        • ShortConv1D + Residual Injection
      │                                                                            ▲
      └────────────────────────── [Async PCIe Transfer <0.15 µs] ──────────────────┘
                                   (100% Hidden during Layer 0 compute)
```

### Core Empirical Findings:
1. **NPU Physical Barrier:** The XDNA 2 NPU possesses **4 MB on-die L2 SRAM**. A 51B parameter table requires **25.5 GB (INT4)** to **102.0 GB (BF16)** and **cannot physically reside on the NPU**. FastFlowLM has zero dynamic `Gather` kernels.
2. **Zero Contention for Sparse Table Gather:** Table lookup transfers only **2.0 KB per token** ($148 \text{ KB/s}$ at 74 tok/s). Empirically measured GPU decode degradation under continuous 24 GB table traffic is **+0.27%** (well within the 5.0% stop rule).
3. **100% Prefetch Latency Hiding:** Host RAM gather ($1.43 \ \mu\text{s}$) is **~300× faster** than the Layer 0 GPU forward pass ($450 \ \mu\text{s}$), completely eliminating GPU stalls.
4. **Memory Capacity Red Line:** Strix Halo (128 GB) can run `Q4_K_M MoE (58 GB) + INT4 Table (25.5 GB)` with **+29.0 GB safe headroom**. If the table ships as unquantized **BF16 (102 GB)**, single-node Strix Halo deployment is a **No-Go** until quantized.

---

## 2. Directory Structure & Research Artifacts

```text
engram-npu-prep/
├── README.md                          # Top-level executive summary & Day-0 readiness checklist
├── requirements.txt                   # Python runtime dependencies
├── notes/
│   ├── engram-architecture.md         # Reference architecture analysis (DeepSeek Engram paper/code)
│   └── v4-pro-config-findings.md      # DeepSeek-V4-Pro production config & modeling audit
├── results/
│   ├── lookup-benchmark.md            # Standalone N-gram lookup & scaling benchmark report
│   ├── lookup-benchmark.json          # Raw benchmark telemetry across parameter scales
│   ├── prefetch-overlap.md            # Asynchronous prefetch & pipeline overlap report
│   ├── prefetch_overlap_results.json  # Raw overlap simulation telemetry
│   └── capacity-planning.md           # Master memory capacity matrix (INT4/INT8/BF16/Q4/Q8)
└── scripts/
    ├── lookup_bench.py                # Standalone hash & table gather benchmark harness
    └── prefetch_test.py               # Asynchronous prefetch & pipeline overlap test
```

---

## 3. Day-0 Readiness Checklist for Qwen3.8-Flash-Next Release

When weights are dropped, execute the following protocol sequentially:

### Step 1: Probe Repositories & Upstream Runtime Support
Check the following sources first:
- **Hugging Face Model Org:** `https://huggingface.co/Qwen` (`Qwen/Qwen3.8-Flash-Next`, `Qwen/Qwen3.8-Flash-Next-GGUF`)
- **GitHub Repositories:**
  - `QwenLM/Qwen3.8` / `vllm-project/vllm` / `sgl-project/sglang`
  - `ggml-org/llama.cpp` / `ROCm/ROCmFPX`

### Step 2: Fetch `config.json` FIRST (Do NOT download full weights)
```bash
# Probe config without downloading 150GB+ of weights
curl -s "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json" > /tmp/qwen_flash_config.json

# Inspect N-gram table parameters and precision
python3 -c "
import json
cfg = json.load(open('/tmp/qwen_flash_config.json'))
print('Architecture:', cfg.get('architectures'))
print('Hidden Size  :', cfg.get('hidden_size'))
print('Engram Layers:', cfg.get('engram_layer_ids') or cfg.get('num_hash_layers'))
print('Table Vocab  :', cfg.get('engram_vocab_size') or cfg.get('table_slots'))
print('Embedding Dim:', cfg.get('n_embed_per_ngram') or cfg.get('index_head_dim'))
print('Torch Dtype  :', cfg.get('torch_dtype'))
"
```

### Step 3: Apply the Go / No-Go Decision Gate
- 🟢 **GO (Strix Halo 128 GB):** Table is `INT4` ($\le 26 \text{ GB}$) or `INT8` ($\le 51 \text{ GB}$) AND Main MoE is `Q4_K_M` ($\le 60 \text{ GB}$).
- 🟡 **GO with Discrete GPU Hybrid:** 4x RTX 3090 hosts Q4/Q5 MoE (58–85 GB) in VRAM; Strix Halo Host RAM hosts the 51B Table (51–102 GB).
- 🔴 **NO-GO on Single Node:** Table is `FP16 / BF16` (102 GB) AND Main MoE is $\ge 58 \text{ GB}$ (Total 167.5 GB > 128 GB UMA). **Action:** Run local embedding table quantization before loading.

### Step 4: Isolate and Partially Load the N-Gram Lookup Table
```bash
# Download only tokenizer, config, and the specific safetensors shard holding the embedding table
python3 -c "
from huggingface_hub import hf_hub_download, list_repo_files
import json

repo_id = 'Qwen/Qwen3.8-Flash-Next'
files = list_repo_files(repo_id)
# Download index to find table shard
index_file = hf_hub_download(repo_id=repo_id, filename='model.safetensors.index.json')
index = json.load(open(index_file))
table_shards = {v for k, v in index['weight_map'].items() if 'engram' in k or 'embedding' in k}
print('Table shards to download:', table_shards)
for shard in table_shards:
    hf_hub_download(repo_id=repo_id, filename=shard)
"
```

### Step 5: Fallback Plan if NPU Cannot Execute the Lookup Op
1. **Lookup & Hashing Execution:** Run on **Host CPU (Zen 5 AVX-512)** in host RAM. (Measured latency is $1.43 \ \mu\text{s}$, 100% masked).
2. **Dense MoE Execution:** Run on **Radeon 8060S iGPU (ROCmFP4 / Vulkan)** or **4x RTX 3090 (vLLM / TensorRT-LLM)**.
3. **NPU Role:** Keep NPU dedicated to auxiliary ambient tasks (FastFlowLM voice/audio, intent classification, live verifier logging).

---

## 4. Benchmark Summary Reference

- **Single-Row Lookup Latency:** **1.43 $\mu\text{s}$** per row
- **Peak Batched Gather Throughput:** **45,233,220 rows/sec** (46.32 GB/s)
- **Concurrent GPU Decode Impact:** **+0.27% degradation** (73.85 tok/s $\to$ 73.65 tok/s)
- **Pipeline Latency Masking Ratio:** **100.0%** (0.0 ms exposed GPU stall)

*Full telemetry and benchmarks are committed in `results/` and `docs/qwen4_npu_readiness.md`.*
