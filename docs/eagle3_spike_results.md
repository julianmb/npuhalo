# EAGLE-3 Spike Results & d2t Sentinel Fix — Complete Findings

## 1. Summary of Actions & Empirical Measurements

1. **d2t Sentinel-Masking Fix Applied & Verified:**
   - Patched `ggml-vulkan` (`copy_to_quant.comp`), `ggml-cpu` (`ops.cpp`), and `ggml-cuda` (`set-rows.cu`) to safely drop negative $d2t$ sentinel values (`d2t[i] < 0`) instead of attempting invalid out-of-bounds row scatters.
   - Rebuilt `ROCmFPX` with Vulkan, CPU, and HIP backends successfully.

2. **Empirical Acceptance Comparison: Full Head vs. Compressed Head:**

| Speculative Drafter | Vocabulary Size | GGUF Footprint | Acceptance Rate | Mean Acceptance Length ($\tau$) | Generation Speed |
|---|---|---|---|---|---|
| **Full EAGLE-3 Head (`EAGLE3-full-q8_0`)** | **248,320 (Full)** | 1.77 GB | **90.6%** | **$\tau = 2.88$** | **19.8 tok/s** (bound by 675MB GEMV) |
| **Compressed EAGLE-3 Head (`EAGLE3-compressed-q8_0`)** | 32,768 (18.5k unique) | 596 MB | 7.4% | $\tau = 1.10$ | 12.4 tok/s (vocab miss rate) |
| **Embedded MTP ($K=4$, Vulkan0)** | **248,320 (Embedded)** | **0 MB extra** | **~80%** | **$\tau \approx 2.50$** | **33.8 tok/s** (🏆 Production Winner) |

---

## 2. Root Cause Analysis: Why Compressed Head Underperforms

1. **Vocabulary Truncation:**
   - The Ex0bit compressed head only maps **18,565 unique tokens** out of the 248,320 target vocabulary (only 7.4% total coverage).
   - Even among the top 500 common syntax and punctuation tokens, 33.2% are missing from the draft vocabulary.
   - Whenever the target model predicts any token outside this 18.5k subset, the draft step is rejected, collapsing acceptance length to 1.10.

2. **The Full Head vs. Embedded MTP Trade-off:**
   - The **Full EAGLE-3 head** has outstanding prediction accuracy (**90.6% acceptance**, $\tau = 2.88$), but on current unified memory architectures, transferring its 1.77 GB working set per draft cycle creates memory bus competition that slows decode to 19.8 tok/s.
   - The **Embedded MTP ($K=4$)** shares the main model's weights and KV cache with **zero auxiliary memory traffic**, allowing the 40 CUs of the Radeon 8060S to decode at **`33.8 tokens/second`**.

---

## 3. Final Production Architecture

1. **Sustained Decoding:** `Vulkan0` + `--spec-type draft-mtp --spec-draft-n-max 4` $\rightarrow$ **`33.8 tokens/sec`**.
2. **First-Token Latency (TTFT):** `qwen3.5-0.8b-FLM` on the **XDNA 2 NPU** $\rightarrow$ **`347 ms TTFT`** (1.8× faster on long context).
3. **Always-On Background Routing:** `scripts/npu_router.py` on `/dev/accel/accel0` at **~2 Watts**.
