# npuhalo — NPU + iGPU Heterogeneous Inference on AMD Strix Halo

[![Hardware](https://img.shields.io/badge/Hardware-AMD_Ryzen_AI_Max%2B_395-red)](#hardware-testbed)
[![NPU Architecture](https://img.shields.io/badge/NPU-AMD_XDNA_2_(50_TOPS)-orange)](#breakthrough-minicpm5-2b-on-xdna-2-npu)
[![iGPU Architecture](https://img.shields.io/badge/iGPU-Radeon_8060S_(RDNA_3.5)-purple)](#primary-model-ornith-15-a3b--rocmfpx-quantization)
[![Model Weights](https://img.shields.io/badge/HuggingFace-MiniCPM5--2B--NPU2-blue.svg)](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

Rigorous experimental evaluation of **heterogeneous NPU + iGPU agentic inference** on **AMD Strix Halo** (Ryzen AI Max+ 395, 128 GB shared UMA). 

This repository documents the empirical mechanics of co-locating large agent generators on the **Radeon 8060S iGPU** with ultra-low-power verification, triage, and compression filters on the **XDNA 2 NPU** (`/dev/accel/accel0`).

---

## Key Results & Takeaways

1. **The NPU Advantage: Zero GPU Contention Background Verifier (~2–4 W)**:
   The XDNA 2 NPU runs completely independent of the graphics/compute pipeline. Offloading verification and triage filters to the NPU consumes just **2–4 W** and preserves **100% of the Radeon 8060S compute capacity** for the primary generator.
2. **The Physics of Shared UMA**:
   On shared-memory APUs (~273 GB/s bus), tightly coupling an NPU drafter *in-loop* with GPU decode pays a physical memory contention tax (**-16.5% GPU decode throughput**). Consequently, speculative decoding and in-loop bursts fail; **asynchronous background verification ("Shadow Verifier")** is the winning architectural pattern.
3. **Primary Model & ROCmFPX Acceleration**:
   The primary agent runs **Ornith-1.5-35B-A3B** (active 3B MoE slice) quantized via **ROCmFPX (FP4 block floating-point)** on the iGPU. ROCmFPX delivers **~50–72 tok/s sustained decode** (a **~1.8x – 2.2x speedup** over unoptimized FP16 baselines) with negligible perplexity loss.
4. **Breakthrough MiniCPM5-2B Port for XDNA 2**:
   We ported OpenBMB's **MiniCPM5-2B** to FastFlowLM on XDNA 2 using mathematical $4\times$ KV head replication ($2 \to 8$ heads, $8:1 \to 2:1$ GQA) and unit QK-norm injection. It achieves **63.6 tok/s sustained decode** on the NPU and is publicly available on [Hugging Face](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2).

---

## Architectural Paradigm: "LLM as Verifier" on Heterogeneous Silicon

In agentic systems, generation errors compound exponentially. A faulty tool argument, hallucinated command, or early task completion declaration can poison the entire trajectory.

Drawing upon the **LLM as Verifier** paradigm ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168); [Lightman et al., 2023](https://arxiv.org/abs/2305.20050); [Weng et al., 2024](https://arxiv.org/abs/2402.05120)), `npuhalo` evaluates whether secondary language models can perform real-time outcome and process supervision on edge silicon without stalling the primary agent.

```
                           [User Agent Task]
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │     Radeon 8060S iGPU (RDNA 3.5 UMA)             │
          │     Primary Generator: Ornith-1.5-35B-A3B        │
          │     Quantization: ROCmFPX (FP4) → 50–72 tok/s     │
          └────────────────────────┬─────────────────────────┘
                                   │ Streamed tokens
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │     Tier 1: Incremental Parser Guard (CPU)       │
          │     Deterministic structure check                │
          │     0 false rejects / 1,004 calls                │
          └────────────────────────┬─────────────────────────┘
                                   │ Checkpoints (every ≤250 tokens)
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │     Tier 2: AMD XDNA 2 NPU Verifier (~2-4W)      │
          │     MiniCPM5-2B-NPU2 / LFM2.5 (63.6 tok/s)       │
          │     Zero iGPU compute contention                 │
          └────────────────────────┬─────────────────────────┘
                                   │ Flagged suspect states
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │     Tier 3: iGPU Logprob Escalator / Judge       │
          │     Selective rollback & recovery decision       │
          └──────────────────────────────────────────────────┘
```

---

## Primary Model: Ornith 1.5 A3B & ROCmFPX Quantization

The primary generation engine is **Ornith-1.5-35B-A3B**, a state-of-the-art mixture-of-experts model activating ~3 billion parameters per token.

### ROCmFPX Performance Optimization
To maximize throughput on the unified memory subsystem of Strix Halo, the model weights were quantized to **ROCmFPX (FP4 block floating-point)**:

| Quantization Format | Active Footprint | Decode Throughput | Prefill Speed | Memory Bandwidth Pressure |
|---|---|---|---|---|
| **FP16 / BF16** | ~35.0 GB | ~31.4 tok/s | ~180 tok/s | High (bus saturation) |
| **Q8_0** | ~18.2 GB | ~44.1 tok/s | ~290 tok/s | Moderate |
| **ROCmFPX (FP4)** | **~9.2 GB** | **~50.2 – 72.4 tok/s** | **~520 tok/s** | **Ultra-low (optimal cache streaming)** |

**Key Advantage**: ROCmFPX reduces memory bus traffic by **~73%**, unlocking the full compute rate of the RDNA 3.5 compute units on the Radeon 8060S and delivering up to **2.2x higher generation speed**.

---

## Breakthrough: MiniCPM5-2B on XDNA 2 NPU

As part of this research, we ported **[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)** natively to FastFlowLM on the AMD XDNA 2 NPU.

* 🌐 **Hugging Face Model**: **[julianmb/MiniCPM5-2B-NPU2](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)**
* **Sustained Decoding Speed**: **63.1 – 63.6 tok/s**
* **Prefill Speed (TTFT)**: **81.5 – 128.1 tok/s** (~420 ms TTFT)
* **Power Draw**: **~2–4 W** (active)
* **Contention**: **0% GPU compute contention**

### How We Solved the GQA Firmware Incompatibility
MiniCPM5-2B has 16 Query heads and 2 Key/Value heads ($16:2 = 8:1$ GQA ratio). The FastFlowLM AIE firmware (`libmha.so`) has no native $8:1$ kernel for $d_{head}=128$.
1. **$4\times$ KV Head Replication**: We replicated the 2 KV heads $4\times$ along dimension 0 into 8 KV heads ($16:8 = 2:1$ GQA ratio). Under Grouped Query Attention, this maintains **exact bit-for-bit mathematical equivalence** while matching the native `_gen_mha_seq_d128_q2` AIE kernel.
2. **Qwen3 Runtime Routing**: FastFlowLM's Llama engine hardcodes $d_{head}=64$ for `hidden_size == 2048`. By routing through the Qwen3 engine (`libqwen3_npu.so`), $d_{head}=128$ is dynamically dispatched when `intermediate_size == 6144`.
3. **Identity QK-Norm Injection**: Synthetic unit RMSNorm tensors ($\gamma = 1.0$) were injected across all 42 layers in `model.q4nx`, allowing RMSNorm to act as a transparent identity operation.

Conversion scripts and verification tools are available in [`ports/minicpm5-2b/`](ports/minicpm5-2b/).

---

## Experimental Record & Verdicts

Every experiment below was conducted under pre-registered decision gates with locked manifests and paired random seeds on the **AMD Strix Halo** testbed:

| # | System Component | Hypothesis | Measured Outcome | Empirical Verdict |
|---|---|---|---|---|
| **1** | **Incremental Parser Guard** | Streaming state-machine prevents syntax aborts | **0 / 1,004 false rejects** across 160 runs · 38/38 tests pass | **ARMED / SHIPPED** |
| **2** | **Speculative Decoding** | NPU drafter speeds up iGPU target | **1,313 ms vs 2,640 ms (2x slower)**; embedded MTP (38 tok/s) wins | **DEAD (Structural)** |
| **3** | **TTFT Handoff Burst** | Streaming first tokens from NPU reduces TTFT | Original win failed to reproduce on current stack (1,430 ms vs 730 ms) | **DEAD (Archived)** |
| **4** | **Query Router** | Fast NPU lane for trivial queries | Routing accuracy only 25% (7/28); parity latency (~1.9s vs ~2.2s) | **DEAD (Accuracy)** |
| **4b**| **Physical UMA Contention** | Concurrent NPU + iGPU memory traffic | **-16.5% GPU decode throughput drop** during concurrent NPU traffic | **PHYSICAL LAW** |
| **5** | **Live Verifier (Shadow)** | NPU triage flags bad trajectories early | Recall **4/4** on catchable failures; **0/22 false alarms** in shadow | **SHADOW ONLY** |
| **6** | **Grammar Constraints** | GBNF removes tool-call syntax errors | Syntax errors $\to$ 0, but task success crashed (**62.5% $\to$ 25.0%**) | **NET-HARMFUL** |
| **7** | **27B Escalation Tier** | Dense 27B model converts persistent failures | Converted only 1/2 target tasks; extreme latency cost (8–16 min) | **ARCHIVED** |
| **8** | **NPU Context Compressor** | NPU compresses large outputs to save prefill | Tool outputs capped at 8K; summaries failed retention gate (0/36) | **ARCHIVED** |
| **9** | **Benchmark Validation (v2)**| Set D v1 benchmark tasks had authoring bugs | CI reference validation gate: **30/30 tasks pass**; baseline is **80.6%** | **VALIDATED (v2)** |

Full reproduction logs, autopsies, and analysis protocols are committed in [`docs/FINDINGS.md`](docs/FINDINGS.md) and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Repository Structure

```text
npuhalo/
├── docs/                      # Experimental records, methodology, and findings
│   ├── FINDINGS.md            # Complete experimental record and physics findings
│   ├── METHODOLOGY.md         # Pre-registration protocol and statistical rigor
│   └── ROADMAP.md             # Research roadmap
├── verifier/                  # Live verification & parser guard framework
│   ├── src/                   # Production runtime modules
│   │   ├── toolcall_parser.py # Incremental parser guard (0 false rejects)
│   │   ├── gated_escalator.py # Multi-tier verifier escalation policy
│   │   └── compressor_sidecar.py # Provenance-preserving context compressor
│   ├── scripts/               # Validation and evaluation scripts
│   │   └── validate_set_d.py  # CI benchmark validation gate (30/30 pass)
│   ├── data/                  # Set D v2 benchmark definitions
│   └── results/               # Curated benchmark summaries and calibration data
├── ports/minicpm5-2b/         # MiniCPM5-2B XDNA 2 porting scripts
│   ├── expand_kv_heads.py     # 4x KV replication tool (2 -> 8 heads)
│   ├── inject_qk_norm.py      # Unit RMSNorm injection script
│   └── test_quality.py        # 5-test reasoning validation suite
├── scripts/                   # Contention and telemetry benchmarking scripts
├── tests/                     # Unit test suites (38 parser guard tests)
└── pyproject.toml             # Python package definition
```

---

## Hardware Testbed

* **APU**: AMD Ryzen AI Max+ 395 (16 Zen 5 cores, 32 threads, up to 5.1 GHz)
* **NPU**: AMD XDNA 2 (48 AIE-ML tiles, 50 TOPS, `/dev/accel/accel0`)
* **iGPU**: AMD Radeon 8060S (40 RDNA 3.5 Compute Units, 128 GB UMA LPDDR5X-8000, ~273 GB/s shared)
* **OS / Drivers**: Ubuntu 24.04 LTS (Linux 6.11+) · ROCm 6.2+ / XRT 2.18+ · FastFlowLM v1.0.2

---

## Quickstart

### 1. Run Unit Tests (Parser Guard)
```bash
pytest tests/test_toolcall_parser.py
```

### 2. Validate Benchmark Suite (Set D v2)
```bash
python3 verifier/scripts/validate_set_d.py
```

### 3. Serve MiniCPM5-2B on XDNA 2 NPU
Download weights from [Hugging Face](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2) and run:
```bash
flm serve minicpm5:2b --host 127.0.0.1 --port 8001
```

---

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
