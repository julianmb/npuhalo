# npuhalo

**Unlocking the power of the Halo NPU.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hardware: AMD Strix Halo](https://img.shields.io/badge/Hardware-AMD_Ryzen_AI_Max%2B_395-red)](https://www.amd.com)
[![Engine: ROCmFPX](https://img.shields.io/badge/Engine-ROCmFPX_(Vulkan_coopmat)-blue)](https://github.com/ggml-org/llama.cpp)
[![NPU: FastFlowLM](https://img.shields.io/badge/NPU_Engine-FastFlowLM_(XDNA2)-purple)](https://github.com/ROCm/FastFlowLM)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](pyproject.toml)
[![Tests](https://github.com/julianmb/npuhalo/actions/workflows/tests.yml/badge.svg)](https://github.com/julianmb/npuhalo/actions)

Your Ryzen AI Max+ 395 has a **48-tile XDNA 2 NPU** that sits idle while the iGPU runs your LLM. We benchmarked every plausible way to put it to work alongside a 35B-A3B MoE generator (`Ornith 1.5`, ROCmFP4, Vulkan). Some ideas worked. The most hyped one didn't. 

All numbers below were measured on real hardware (128 GB LPDDR5X, Linux 7.0 / Ubuntu 24.04). Tracked summaries and sanitized autopsy artifacts are available in [`verifier/results/`](verifier/results/) and [`docs/`](docs/); bulky raw run telemetry is intentionally excluded from Git.

> **Project status: alpha research software.** The routing and compression utilities are usable prototypes; live rollback remains experimental. Network services default to localhost, and generated-code evaluators are not security sandboxes.

[🚀 Quickstart](#-quickstart) • [📺 Live Demo](#-interactive-terminal-demo) • [📐 Architecture](#-3-tier-architecture) • [📊 Results](#-measured-results) • [📄 Decision Report](verifier/results/final-report.md) • [🗺️ Roadmap](docs/ROADMAP.md)

---

## ⚡ TL;DR

| Approach | Status | Measured Result | Production Verdict |
| :--- | :---: | :--- | :--- |
| **Tool-Output Context Compression** | 🟡 **Prototype** | **80–98% context reduction**, ~1.0s NPU latency | Promising on long outputs; fidelity needs broader validation |
| **Deterministic Parser Guard** | 🟢 **Ship** | <0.1ms, zero model compute | Deterministic schema validation + format hints; harness truncation was fixed separately |
| **NPU Query Router (Fast Lane)** | 🟡 **Feature** | **1,384 ms NPU-direct** vs 14,635 ms GPU-thinking | Fast, low-power (~2 W) classification & routing for short queries |
| **Live Stream Verification** | 🔵 **Research** | 4/4 catchable failures flagged, 0/22 clean false alarms, −8.2% throughput | Detection is measured; end-to-end rollback conversion remains unmeasured |
| **Speculative Decoding (NPU $\to$ GPU)** | 🔴 **Archived** | 2,640 ms vs 1,313 ms GPU-only (**2× slower**) | Active-parameter-light MoE targets decode faster than NPU drafters |

---

## 📺 Interactive Terminal Demo

Once the generator and verifier services are running, launch the terminal demo to stream tokens with color-coded verification badges:

```bash
npuhalo-demo
# From a source checkout: make demo
# Custom prompt: npuhalo-demo --prompt "Write a complete Quicksort in Python with unit tests."
```

````text
⚡ NPUHalo: Live Streaming Verification on AMD Strix Halo
Generator: http://127.0.0.1:8012/v1 (Radeon 8060S iGPU, ROCmFP4 MoE)
Verifier : http://127.0.0.1:8001/v1 (XDNA 2 NPU, FastFlowLM LFM2.5-tk)

Here is a clean implementation of Quicksort in Python:
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
[✓ NPU Checkpoint #1: CONTINUE | 885ms | code_fence_close]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```
[✓ NPU Checkpoint #2: CONTINUE | 910ms | blank_line_step]

📊 Run Summary:
 • Total Tokens Generated : 248 tokens
 • Generation Wall Time   : 3.36 s
 • Effective Throughput   : 73.8 tok/s
 • NPU Audits Completed   : 2 checkpoints
 • Mean NPU Checkpoint Lat: 897 ms (off-GPU, ~2 W power)
 • Final Stream Status    : PASSED
````

---

## 📐 3-Tier Architecture

The evaluation produced the following 3-tier research design for separating deterministic formatting checks from model-based anomaly detection:

```text
                               Agent Generation Step
                                         │
                                         ▼
            ┌────────────────────────────────────────┐
            │ Layer 1: Deterministic Parser Guard    │ ──(Malformed/Empty)──► Reject + Format Hint
            │          (Free, <0.1 ms latency)       │
            └────────────────────────────────────────┘
                                         │ (Valid tool call)
                                         ▼
            ┌────────────────────────────────────────┐
            │ Layer 2: NPU LFM2.5-tk Triage (:8001)  │ ──(CONTINUE)─────────► Stream Proceed (0 GPU tokens)
            │          (~0.9s latency, ~2 W power)   │
            └────────────────────────────────────────┘
                                         │ (SUSPECT)
                                         ▼
            ┌────────────────────────────────────────┐
            │ Layer 3: iGPU Qwen3.5-2B Judge (:8013) │ ──(ABORT)────────────► Candidate Rollback Decision
            │          (CONTINUE vs ABORT logprobs)  │                        (active conversion not yet measured)
            └────────────────────────────────────────┘
```

1. **Layer 1 (Parser Guard):** Instant deterministic schema validation for malformed tool calls. Four observed silent stalls were caused by a separate token-budget truncation bug and are not credited to this guard.
2. **Layer 2 (NPU Triage):** Asynchronous continuous verification on the 48-tile XDNA 2 NPU. Outpaces generator checkpoint boundaries (~0.9s vs ~3.4s window).
3. **Layer 3 (iGPU Escalation Judge):** The NPU 1.2B model acts as a conservative triage filter (votes `SUSPECT`, never raw `ABORT`). The iGPU judge separated one broken calibration trajectory ($r=0.000$) from one clean trajectory ($r=0.806$); a complete active rollback sweep is still required.

---

## 📊 Measured Results

* **Generator:** `Ornith 1.5 35B-A3B`, `Q4_0_ROCMFP4_STRIX_LEAN` quantization on Radeon 8060S iGPU — **72.04 tok/s decode**, **151.4 ms TTFT**. The unrelated 14.1 tok/s Qwen benchmark is not used as a same-model speedup baseline.
* **Verifier:** `LFM2.5-1.2B-Thinking` on XDNA 2 NPU via FastFlowLM — **~43 tok/s**, **~0.9s median checkpoint latency** (4× faster than generation checkpoint window).

| Evaluation Phase | Metric | Measured Result | Notes |
| :--- | :--- | :--- | :--- |
| **Set D Baseline (30 Agentic Tasks)** | Pass Rate | **80.0%** (24/30 single-seed; 78.6% across N=234 runs) | High capability bounds failure headroom |
| **Shadow Mode Autopsy** | Real Recall on Catchable Bugs | **4 / 4 (100.0%)** | Caught 100% of test errors, tracebacks & permission faults |
| **Shadow Mode Autopsy** | False Alarms on Clean Runs | **0 / 22 (0.0%)** | No clean trajectory was escalated in the observed sample |
| **Escalator Calibration** | Logprob Separation | **$r = 0.000$ (broken)** vs **$r = 0.806$ (clean)** | Clean threshold separation on iGPU `:8013` |
| **NPU Capacity & Concurrency** | Saturation Boundary | **3.6–3.8 concurrent streams** | Median latency: 1.15s @ 1 stream $\to$ 3.61s @ 4 streams |
| **NPU Context Compressor** | Context Size Reduction | **80.0% to 98.0%** | Prototype measured on tool outputs longer than 500 characters |

### Economic Breakeven (Live Verification)

$$\text{Modeled Net Savings} = (0.133 \text{ failure rate} \times 0.65 \text{ assumed token avoidance}) - 0.082 \text{ throughput cost} = \mathbf{+0.5\%}$$

The +0.5% value is a **scenario estimate**, not an active-mode measurement: the 65% avoided-waste term remains an assumption until abort-and-resample conversion is measured. Any operational value would come from preventing downstream side effects in long-running workflows.

---

## ❌ What Doesn't Work (and Why)

**NPU $\to$ GPU Speculative Decoding:** A 35B-A3B MoE activates ~3B params per token and decodes at 56–80 tok/s on the iGPU. The NPU draft model runs at 30–34 tok/s. Speculative decoding only wins when the target model is memory-bandwidth bound; an active-parameter-light MoE is compute-bound. Acceptance rates averaged **32%** on code and collapsed to **6–7%** on open-ended text.

* **Measured:** **2× slower** than GPU-only execution ($2,640\text{ ms}$ vs $1,313\text{ ms}$).
* **Full analysis:** [`docs/REPORT.md`](docs/REPORT.md) and [`docs/final_verdict.md`](docs/final_verdict.md); benchmark suite in [`scripts/npu_benchmark.py`](scripts/npu_benchmark.py).

---

## 🛠️ Quickstart

### 1. Requirements & NPU Setup
* **Hardware:** AMD Strix Halo (Ryzen AI Max+ 395, 128 GB UMA LPDDR5X-8000).
* **OS / Kernel:** Linux 6.11+ (tested on Linux 7.0 / Ubuntu 24.04) with `amdxdna` kernel driver.
* **Python:** CPython 3.12 or newer (the version used by CI and the measured workstation).
* **Boot Flag:** Boot with SVA enabled: `iommu=pt iommu.passthrough=0` (disabling IOMMU disables the NPU).
* **User Group:** `sudo usermod -aG render "$USER"` (log out and back in).

Verify hardware status with the repository triage tool:
```bash
make status
# or: python3 scripts/npu_status.py
```

### 2. Install

```bash
git clone https://github.com/julianmb/npuhalo.git
cd npuhalo
python3 -m pip install -e .

# Verify the packaged console entry point
npuhalo-demo --help
```

CI also builds a wheel, installs it without the source checkout, and smoke-tests the `npuhalo-demo` entry point. The wheel includes the calibrated verifier prompt used by the demo.

### 3. Launch Services

```bash
# 1. Primary Generator (Radeon 8060S iGPU, Vulkan cooperative matrices, Port 8012)
llama-server \
  -m /path/to/Ornith-1.5-35B-A3B-ROCmFP4.gguf \
  --device Vulkan0 --port 8012 --host 127.0.0.1 \
  -c 16384 -ngl 99 -fa 1 --threads 16 --no-context-shift -np 1

# 2. NPU Verifier & Compressor (FastFlowLM on XDNA 2 NPU, Port 8001)
flm serve lfm2.5-tk:1.2b --host 127.0.0.1 --port 8001

# 3. Escalation Judge (iGPU llama-server with logprobs, Port 8013)
llama-server \
  -m /path/to/Qwen3.5-2B-Q4_K_M.gguf \
  --device Vulkan0 --port 8013 --host 127.0.0.1 \
  -c 8192 -ngl 99 -fa 1 --threads 8
```

### 4. Run Evaluations & Benchmarks

```bash
# Run unit & regression test suite (25 tests)
make test

# Run 30-task agentic shadow autopsy (NPU verification logging)
make shadow

# Run NPU capacity & concurrency sweep
make sweep

# Run NPU tool-output context compression benchmark
make compress
```

---

## 🧩 FastFlowLM Logprobs Integration

Stock FastFlowLM returns `"logprobs": null` over HTTP, which blocks distribution-based verifier scoring on the NPU. Our measured setup used an **out-of-tree FastFlowLM modification** that adds:
* Full-vocabulary log-softmax from raw NPU logits (248K vocab)
* OpenAI-compatible `logprobs` and `top_logprobs` on `/v1/chat/completions` and `/v1/completions`
* GBNF grammar-constrained decoding ported from `llama.cpp`

The modification is published as a unified diff: [`patches/0003-flm-logprobs-and-grammars.patch`](patches/0003-flm-logprobs-and-grammars.patch) (see [`patches/README.md`](patches/README.md) for scope and apply instructions). The tracked Python-side validation tooling is [`npu_logit_adapter.py`](standalone-eval/scripts/npu_logit_adapter.py) and [`test_logprobs.py`](standalone-eval/scripts/test_logprobs.py). With the patched server, the verifier tier can run on the NPU while leaving the iGPU available for the primary generator.

---

## 🔍 Honest Caveats

1. **Recall sample size is small:** 4/4 catchable failures were flagged in shadow mode, but only 4 of 8 total failures produced observable evidence. The other 4 were harness truncation stalls fixed by increasing the token budget and reading `reasoning_content`; they are not credited to the parser guard or verifier.
2. **Abort precision requires prompt tuning:** Raw FastFlowLM text endpoints return `SUSPECT`, not `ABORT`. The 100% planted-error detection figure holds only when paired with calibrated prompts and the iGPU escalation judge.
3. **Overhead is −8.2% steady-state, not −3.5%:** An earlier internal preliminary report claimed 3.5%; live measurements on short tasks show **−8.2%** steady overhead and up to **−26%** on cold first tasks. We publish the reproduced number.
4. **NPU cold start is ~74s:** FastFlowLM pays a one-time graph compile/weight load on first invocation. Pre-warm the endpoint in production.
5. **Set D tasks topped out at ~80% baseline pass rate:** Ornith-1.5 is a strong generator; verification headroom is naturally bounded on simple tasks. Value claims should be understood as **insurance economics**, not magical pass-rate uplift.
6. **Active rollback is incomplete:** Shadow detection is measured, but whether abort-and-resample converts failures into successful runs is still an open experiment.
7. **Evaluation executors are not security sandboxes:** Agent and code-evaluation scripts may execute generated Python or shell commands with the current user's privileges. Run only trusted tasks inside a disposable container or VM; do not point them at sensitive workspaces.
8. **Router speed is measured; router accuracy is not:** The 1,384 ms fast-lane figure measures latency only. No A/B evaluation has compared answer quality of NPU-routed short queries against always-GPU generation.
9. **Compressor size reduction is measured; end-to-end time savings are not:** The 80–98% context reduction and ~1s NPU latency are measured, but no live-loop experiment has verified that the prefill tokens saved outweigh the compression latency on this hardware.
10. **TTFT handoff coherence is unscored:** The 347 ms first-token handoff ([archived prototype](docs/HYBRID_NPU_PIPELINE.md)) streams NPU text before the iGPU continues. Perceived-latency gains are measured; output coherence of the stitched text versus pure GPU generation has never been formally evaluated.

### Positioning

The measured evidence supports one framing: the NPU is a **low-power sidecar for small-model auxiliary work** (routing, compression, triage, first-token bursts) that leaves the unified memory bus untouched — not a second inference engine. Every attempt to make it accelerate big-model decode was measured and lost to memory-bus contention.

---

## 📂 Repository Layout

```text
npuhalo/
├── verifier/                    # 3-tier live streaming verification & agentic eval suite
│   ├── src/                     # Core runtime (agent_loop, stream_tap, verifier_client, escalator)
│   ├── scripts/                 # Benchmark runners (shadow autopsy, active escalation, sweep, compressor)
│   ├── data/                    # Evaluation sets (Set A math, Set B code, Set C errors, Set D agentic)
│   ├── prompts/                 # Calibrated verifier & judge prompts
│   └── results/                 # Sanitized autopsies, aggregate results, and final decision report
├── standalone-eval/             # Standalone pairwise LFM2.5 vs Qwen evaluation
├── examples/                    # Interactive visual demo (examples/demo_stream.py)
├── scripts/                     # Hardware diagnostics, NPU router, MTP/speculative-decoding benchmarks
├── tests/                       # Automated regression and unit tests (25 tests)
├── docs/                        # Reviewed reports, baselines, archived experiments, and public roadmap
│   └── research/                # Curated research index and evidence boundaries
├── patches/                     # Out-of-tree FastFlowLM / llama.cpp modifications (logprobs, GBNF)
├── Makefile                     # 1-command targets (make test, make demo, make shadow, make sweep)
├── pyproject.toml               # Python 3.12+ package, dependencies, prompt data, and CLI entry point
├── CONTRIBUTING.md              # Development workflow and benchmark contribution requirements
├── SECURITY.md                  # Threat model, safe-use guidance, and vulnerability reporting
└── LICENSE                      # Apache-2.0 License
```

---

## 💻 Hardware Reference

* **Processor:** AMD Ryzen AI Max+ 395 (16 Zen 5 cores, 32 threads)
* **iGPU:** AMD Radeon 8060S (40 CUs, RDNA 3.5, `gfx1151`)
* **NPU:** AMD XDNA 2 (`RyzenAI-npu5`, 48 AIE2p tiles @ 50 TOPS, `/dev/accel/accel0`)
* **Memory:** 128 GB LPDDR5X-8000 Unified Memory (~273 GB/s peak bandwidth)
* **OS & Kernel:** Linux 7.0 / Ubuntu 24.04 (`amdxdna` kernel module, SVA active)

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🤝 Acknowledgments

Built on [FastFlowLM](https://github.com/ROCm/FastFlowLM), [llama.cpp](https://github.com/ggml-org/llama.cpp), and the speculative-decoding test harnesses for AMD Strix Halo.
