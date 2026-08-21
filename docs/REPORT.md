# Strix Halo NPU + iGPU — Full Technical Report: Tools, Code, Findings, Paths

> **Status: historical experiment log.** Some prototypes and local model
> artifacts described here are not distributed by this repository. The NPU
> burst pipeline is a text handoff, not speculative verification. Use
> [`final_verdict.md`](final_verdict.md) and the root README for current conclusions.

This document is the complete reference for the work done in this repository (`npuhalo`) on accelerating **Qwen 3.8 27B** inference by combining the **AMD XDNA 2 NPU** (`/dev/accel/accel0`) with the **Radeon 8060S iGPU** (`Vulkan0` / `KHR_coopmat`) on **AMD Strix Halo (Ryzen AI Max+ 395)**.

---

## 1. Hardware & System Context

| Component | Detail |
|---|---|
| Processor | AMD Ryzen AI Max+ 395 (16 Zen 5 cores) |
| iGPU | Radeon 8060S (40 CUs, RDNA 3.5, `gfx1151`) |
| NPU | AMD XDNA 2 (`RyzenAI-npu5`, 48 AIE2p tiles @ 50 TOPS, `/dev/accel/accel0`) |
| Unified Memory | 128 GB LPDDR5X-8000, 256-bit bus, ~273 GB/s peak |
| Kernel / OS | Linux 7.0.0-28-generic (SVA enabled, `iommu=pt`) |
| NPU firmware | `1.1.2.65` (updated during this work) |
| XRT | 2.26.0 at `/opt/xilinx/xrt/` |
| GPU engine | ROCmFPX fork of llama.cpp |
| Target model | `Qwen3.8-27B-ROCmFP4-FAST.gguf` (13.55 GiB, 27.32B) |

### External tools used
- **Lemonade** (`/usr/bin/lemonade`) — local AI server (port 13305) that drives the NPU via FastFlowLM.
- **FastFlowLM (`flm`)** — NPU inference runtime (v0.9.46), bundled at `/var/lib/lemonade/.cache/lemonade/bin/flm/npu/flm`.
- **XRT** (`xrt-smi`, `xrt-smi examine`) — Xilinx Runtime for NPU management.
- **`hf`** (Hugging Face CLI) — model downloads.
- **`safetensors` / `torch`** — model weight inspection.
- **`gguf-py`** — GGUF reader/writer (part of the ROCmFPX repo).

---

## 2. Best Measured Configuration

### 2.1 Sustained decode — the winner (iGPU only)

```bash
# llama-server from the ROCmFPX fork (build-strix-rocmfp4/bin)
llama-server \
  -m /path/to/Qwen3.8-27B-ROCmFP4-FAST.gguf \
  --device Vulkan0 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.0 \
  -ngl 99 -fa 1 -c 32768 -b 2048 -ub 2048 --no-mmap --reasoning off
```

Result: **33.8 tok/s** (2.4× over the 14.1 tok/s bare baseline).

Key tuning discovery: **draft depth `K=4` is the sweet spot**. `K=6` regresses (bus saturation), `K=8` is catastrophic (18.2 tok/s). See `docs/mtp_sweep_results.json`.

### 2.2 Archived handoff prototype (NPU burst → GPU continuation)

`scripts/run_pipeline.py` serves an OpenAI-compatible API on port **11435**:

```bash
python3 scripts/run_pipeline.py --device Vulkan0 --draft-n 4 --npu-burst-tokens 24
# daemonized via:
python3 scripts/launch_pipeline.py --device Vulkan0 --draft-n 4
```

Health check: `curl http://127.0.0.1:11435/health`

---

## 3. All Deliverables — Code & Paths

### `scripts/` (workspace scripts)

| File | Purpose |
|---|---|
| `scripts/npu_status.py` | Validates `/dev/accel/accel0`, SVA access, `amdxdna` driver, XRT status |
| `scripts/npu_drafter.py` | Async daemon serving `qwen3.5-0.8b-FLM` draft tokens through a user-private UDS |
| `scripts/npu_router.py` | Always-on 2 W intent classifier (chat/code/translation + length) on the NPU |
| `scripts/run_pipeline.py` | Experimental NPU-to-iGPU text handoff; local OpenAI-compatible API on :11435 |
| `scripts/launch_pipeline.py` | Double-fork daemonizer for `run_pipeline.py` |
| `scripts/npu_benchmark.py` | Benchmark suite (bare vs MTP vs hybrid vs EAGLE-3 matrix) |
| `scripts/sweep_mtp.py` | MTP parameter sweep (K × strict-qwen × p_min) |
| `scripts/export_eagle_head.py` | EAGLE-3 head architecture scaffold (ONNX export attempt) |
| `scripts/extract_mtp_gguf.py` | GGUFWriter-based MTP head extraction (superseded by slicer) |
| `scripts/slice_mtp_gguf.py` | Binary GGUF slicer (raw bytes, no re-quant) — produces head-only GGUF |
| `scripts/ipc/ring_buffer.py` | POSIX shared-memory token ring buffer (0.42 µs/token, 2.35M tok/s) |
| `scripts/ipc/async_pipeline.py` | Async double-buffering coordinator |
| `scripts/ipc/__init__.py` | Package init |

### Local `models/` artifacts (not committed or distributed)

| Path | Size | Description |
|---|---|---|
| `models/Qwen3.8-27B-MTP-Head.gguf` | 1.5 GB | Extracted head-only GGUF (blk.64 + embeddings) |
| `models/EAGLE3-full-q8_0.gguf` | 1.77 GB | Full-vocab EAGLE-3 head (Q8_0) — loads & drafts |
| `models/EAGLE3-compressed-q8_0.gguf` | 596 MB | Compressed d2t EAGLE-3 head (Q8_0) |
| `models/Qwen3.8-27B-MTP-4bit/` | — | MLX MTP head (tokenizer + 227 MB safetensors) |
| `models/Ex0bit-EAGLE3-full/` | 3.3 GB | Source BF16 full head (downloaded) |
| `models/Ex0bit-EAGLE3-compressed/` | 1.1 GB | Source BF16 compressed head (downloaded) |
| `models/eagle3_qwen38_27b/` | — | EAGLE-3 scaffold + config |

### `docs/` (documentation)

| File | Content |
|---|---|
| `docs/baselines.md` | NPU + iGPU baseline measurements |
| `docs/mtp_sweep_results.json` | K-depth sweep raw data |
| `docs/benchmark_results.json` | Local benchmark matrix generated by `scripts/npu_benchmark.py`; not committed |
| `docs/npu_0.8b_bench_real.json` | Real NPU drafter benchmark |
| `docs/eagle3_npu_results.md` | Early EAGLE-3 findings + Track A/B/C |
| `docs/eagle3_spike_results.md` | Spike results + d2t fix + final comparison table |
| `docs/final_verdict.md` | Reconciled final verdict + forward-path assessment |
| `docs/HYBRID_NPU_PIPELINE.md` | Archived handoff-prototype notes |

---

## 4. Fork Modifications (ROCmFPX repo)

### 4.1 EAGLE-3 HF→GGUF converter (spike work)

`convert_hf_to_gguf.py` (+79 lines):
- New `EAGLE3Model` class (registered for `LlamaForCausalLMEagle3`), modeled on `DFlashModel`.
- Handles target-model tokenizer injection, `head_dim=128` (≠ n_embd/n_head), `target_layers=[1,31,60]`, `target_hidden_size=5120`.
- int64 (`d2t`) tensor preservation through the converter pipeline (3 targeted fixes).

`gguf-py/gguf/tensor_mapping.py` (+5 lines):
- `model.layers.{bid}.hidden_norm` → `ATTN_NORM_2`
- `d2t` → `D2T`

### 4.2 d2t negative-sentinel masking fix (C++ + shader)

| File | Change |
|---|---|
| `ggml/src/ggml-vulkan/vulkan-shaders/copy_to_quant.comp` | Skip scatter when i64 index sign bit set |
| `ggml/src/ggml-cpu/ops.cpp` | `if (i1 < 0) continue;` |
| `ggml/src/ggml-cuda/set-rows.cu` | `if (dst_row < 0) return;` (×4 kernels) |

Rebuilt via: `cmake --build build-strix-rocmfp4 --config Release -j 16`.

---

## 5. Key Findings (all empirical)

### 5.1 Performance matrix

| Architecture | Prefill | Decode | TTFT (short / long) |
|---|---|---|---|
| Standalone iGPU (no MTP) | 101.4 tok/s | 14.1 tok/s | ~1800 ms |
| **iGPU + MTP (K=4)** | 74.6 tok/s | **33.8 tok/s** | 510 / 1587 ms |
| Hybrid NPU+GPU | >370 tok/s | 33.8 tok/s | 721 / **870 ms** |
| EAGLE-3 full | — | 19.8 tok/s | — |
| EAGLE-3 compressed | — | 12.4 tok/s | — |

### 5.2 NPU drafter (measured)

- `qwen3.5-0.8b-FLM`: **42.9 tok/s**, **347 ms TTFT**, **~2 W**, 0.2 GB footprint.

### 5.3 EAGLE-3 verdict (closed)

- Full head: **90.6% acceptance (τ=2.88)** — cross-generation drift (Qwen 3.6→3.8) is **not** a problem.
- Compressed head: **7.4% acceptance** — its 32k draft vocab covers only 18.5k/248k of Qwen 3.8's tokens.
- Conclusion: **any separate drafter loses to embedded MTP** on this bandwidth-bound APU, because MTP shares the target's weights with zero auxiliary memory traffic.

### 5.4 Negative results (documented, important)

- KV cache quantization (`q8_0`): no-op for this hybrid linear-attention model.
- Split-device MTP head (CPU/GPU): 16.9–22.7 tok/s — loses to embedded 33.8.
- NPU as co-decoder: loses; NPU's 42.9 tok/s degrades to ~14 tok/s under shared-bus contention.

---

## 6. The Definitive Answer

**33.8 tok/s via embedded MTP (iGPU only) is the practical ceiling on Strix Halo.**

The NPU's real, proven value:
1. **1.8× faster first-token on long prompts** (870 ms vs 1587 ms) via the hybrid burst pipeline.
2. **2 W always-on intent routing** (`npu_router.py`).
3. It does **not** improve sustained decode speed.

The only path beyond 34 tok/s is a future parallel, target-aligned drafter (PARD-2 class) — not deployable today. See `docs/final_verdict.md`.
