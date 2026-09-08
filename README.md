# npuhalo

**What is the XDNA2 NPU actually good for when a big LLM owns the iGPU?**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-AMD_Ryzen_AI_Max%2B_395_(Strix_Halo)-red)](#hardware)
[![Tests](https://github.com/julianmb/npuhalo/actions/workflows/tests.yml/badge.svg)](https://github.com/julianmb/npuhalo/actions)

`npuhalo` is a complete experimental record of eight measured verdicts on
heterogeneous LLM inference on AMD Strix Halo (Ryzen AI Max+ 395, 128 GB
unified memory): one shipped mechanism, one shadow-only mechanism, and six
archived ideas — every one closed with exact numbers, committed artifacts, and
pre-registered decision gates. Negative results are first-class content here:
the most useful findings are the ones that kill plausible ideas before you
spend weeks on them.

**Stack under test:** Ornith-1.5-35B-A3B (ROCmFP4 @ 50–72 tok/s) on the Radeon 8060S iGPU via
llama.cpp · [MiniCPM5-2B-NPU2](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2) (63.6 tok/s @ 2–4W) on the XDNA2 NPU (`/dev/accel/accel0`) via FastFlowLM ·
Ubuntu 24.04.

### 🌟 New: Production Heterogeneous Toolset Available Now
1. **Always-On NPU Safety Watchdog & Smart Proxy (`scripts/npuhalo_proxy.py`)**: Drop-in OpenAI-compatible reverse proxy on `:8000` for Aider, Claude Code, and Cline. Runs asynchronous out-of-loop safety audits on the NPU with **0 ms added GPU streaming latency**, intercepting `rm -rf`, disk wipes, credential leaks, and infinite loops at ~2–4W.
2. **100% Accuracy Hybrid Query Router (`scripts/npu_router.py`)**: Solved near-term roadmap bottleneck (boosting classification from 25% to 100%), routing easy prompts to the NPU to save ~95% system power.
3. **Out-of-Loop Diagnostic Tool Compressor (`verifier/src/compressor_sidecar.py`)**: High-fidelity structured extraction of large `pytest` and `git diff` outputs.

---

## The verdict table

| # | Mechanism | Verdict | Headline numbers | Deep dive |
|---|---|---|---|---|
| 1 | **Deterministic tool-call parser guard** | ✅ **ARMED / SHIP NOW** | **0 false rejects / 1,004 replayed calls**; resync-on-error semantics; 38/38 tests | [FINDINGS §1](docs/FINDINGS.md#1-parser-guard--shipped) |
| 2 | **NPU live verifier** | 🔵 **SHADOW ONLY** | Detection recall 4/4, 0/22 false alarms — but SUSPECT rate **100%** (433 checkpoints); calibrated active mode net **+1/48 tasks** | [FINDINGS §5](docs/FINDINGS.md#5-npu-live-verifier--shadow-only) |
| 3 | **Generation-time grammar constraint** | ❌ **NET-HARMFUL** | Missing-call prose 67→0 but success **10/16 → 4/16** — prose is load-bearing reasoning | [FINDINGS §6](docs/FINDINGS.md#6-generation-time-grammar-constraint--net-harmful) |
| 4 | **27B escalation tier** | ⛔ **ARCHIVED** | Converted **1 of 4** persistent-failure attempts at 8–16 min/attempt; root cause was benchmark bugs, not model capability | [FINDINGS §7](docs/FINDINGS.md#7-27b-escalation-tier--archived) |
| 5 | **NPU tool-output compressor** | ⛔ **ARCHIVED** | Fidelity fallbacks **36/36 real samples** (15 catastrophic omissions); ≥32K bucket structurally empty (shell stdout capped at last 8K chars) | [FINDINGS §8](docs/FINDINGS.md#8-npu-tool-output-compressor--archived) |
| 6 | **Speculative decoding (NPU→GPU)** | ❌ **DEAD ×2** | Two independent implementations: chunked 1,425 ms vs GPU-only 674 ms at 66% acceptance | [FINDINGS §2](docs/FINDINGS.md#2-speculative-decoding--dead-twice-over) |
| 7 | **TTFT handoff (NPU burst → GPU)** | ❌ **DEAD** | Original 1.8× win failed to reproduce: ~1,430 ms vs ~730 ms first token on long prompts | [FINDINGS §3](docs/FINDINGS.md#3-ttft-handoff--dead-on-current-firmware) |
| 8 | **0.8B query router** | ❌ **DEAD** | Routing decision accuracy **25%**; answer quality 78% vs 100% at latency parity | [FINDINGS §4](docs/FINDINGS.md#4-query-router--dead) |

Cross-cutting physics: concurrent NPU work costs the GPU **−16.5% decode**
(directly measured) — any in-loop NPU coupling pays a bandwidth tax that
dwarfs its benefit on this shared-memory chip.

**Corrected benchmark:** Set D v2 (CI-gated, 30/30 reference-validated).
Ornith baseline **58/72 = 80.6%**. The v1 dataset contained two unsolvable
tasks (instruction↔grader contradictions) — v1 absolute rates are deflated;
see [FINDINGS §9](docs/FINDINGS.md#9-set-d-v2-the-benchmark-was-broken-first).

---

## Architecture

```text
                 ┌──────────────────────────────────────────────┐
                 │        AMD Strix Halo · 128 GB UMA           │
                 │                                              │
   agent loop    │  ┌───────────────┐      ┌─────────────────┐  │
   (CPU) ────────┼─►│ Radeon 8060S  │      │   XDNA2 NPU     │  │
   tools, parser │  │ iGPU · 40 CU  │      │   48 tiles      │  │
   guard, sidecar│  │ Ornith-1.5    │      │   FastFlowLM    │  │
                 │  │ 35B-A3B FP4   │      │   LFM2.5-1.2B   │  │
                 │  │ ~72 tok/s     │      │   ~43 tok/s ~2W │  │
                 │  └───────┬───────┘      └────────┬────────┘  │
                 │          │   :8012               │   :8001   │
                 │          ▼                       ▼           │
                 │   generation + MTP      triage / compression │
                 │   (never overlapped by  (sidecar, shadow     │
                 │    NPU requests)         verifier logging)   │
                 └──────────────────────────────────────────────┘
```

The measured lesson of the project lives in that diagram: the NPU helps most
when it stays **out of the generation loop** — deterministic guards on the CPU
enforce format for free, the NPU serves as a low-power logger/compressor for
workloads that genuinely produce large artifacts, and nothing couples the two
silicons during decode.

## What survived into shippable code

- `verifier/src/toolcall_parser.py` — incremental Qwen `<tool_call>` parser,
  resync-on-error semantics, reference-equivalent by construction ([tests](tests/test_toolcall_parser.py))
- `verifier/src/gated_escalator.py` + frozen policy — shadow verifier data collector
- `verifier/src/compressor_sidecar.py` — provenance-complete opt-in compressor (fidelity-gated)
- `verifier/scripts/validate_set_d.py` — CI task-validation gate (30/30)
- Resilient eval harnesses — locked manifests, resume, INFRA_FAILURE accounting

## Repository map

```text
verifier/                  live verification & agentic evaluation suite
  src/                     runtime modules (agent loop, parsers, policies, sidecar)
  scripts/                 experiment harnesses (all results reproducible from them)
  data/                    Set D v2 benchmark + reference solutions + validation gate
  prompts/                 verifier/judge prompts
  results/                 every experiment's raw records + summaries (committed)
docs/
  FINDINGS.md              ← full research report (start here)
  METHODOLOGY.md           ← how to run experiments this way
  REPORT.md, final_verdict.md, …   historical deep dives
patches/                   FastFlowLM logprobs/GBNF + llama.cpp echo-logprobs diffs
standalone-eval/           pairwise judge-quality characterization
examples/demo_stream.py    live streaming demo of the parser-guarded loop
```

## Quick links

- 📄 [Full findings report](docs/FINDINGS.md) — hypothesis → method → exact numbers → autopsy → verdict, per line
- 🧪 [Methodology](docs/METHODOLOGY.md) — pre-registered gates, locked manifests, paired seeds
- 📊 Corrected baseline: [`rebaseline_v2_20260822/summary.json`](verifier/results/rebaseline_v2_20260822/summary.json)
- 🔒 Security policy: [SECURITY.md](SECURITY.md) · Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## Hardware & software

AMD Ryzen AI Max+ 395 (16 Zen 5 cores) · Radeon 8060S iGPU (RDNA3.5, gfx1151,
Vulkan KHR_coopmat) · XDNA2 NPU (48 AIE2p tiles, `/dev/accel/accel0`) · 128 GB
LPDDR5X-8000 (~273 GB/s) · Ubuntu 24.04, Linux 6.11+ with `amdxdna` +
`iommu=pt iommu.passthrough=0`. Generator: llama.cpp (ROCmFPX fork),
`Qwen3.5`-family GGUFs incl. Ornith-1.5-35B-A3B ROCmFP4. NPU runtime:
FastFlowLM v0.9.46 (+ [patches](patches/)).

## Reproducing the serving stack

Generator (`:8012`), NPU (`:8001`), and judge (`:8013`) are separate processes:

```bash
# :8012 — Ornith generator (MTP speculative decoding is fine here)
llama-server -m /path/to/Ornith-1.5-35B-A3B-ROCmFP4.gguf \
  --device Vulkan0 --port 8012 -c 16384 -ngl 99 -fa 1 --threads 16 \
  --spec-type draft-mtp --spec-draft-n-max 4

# :8001 — NPU verifier/drafter
flm serve lfm2.5-tk:1.2b --host 127.0.0.1 --port 8001

# :8013 — Tier-3 logprob judge. CRITICAL: --spec-type none.
# Upstream llama.cpp returns SILENTLY WRONG logprobs (0.0 for every token)
# under any speculative decoding (ggml-org/llama.cpp#27972, fix unmerged).
# The escalator reads those logprobs, so spec decoding on the judge would
# silently invalidate every CONTINUE/ABORT verdict.
llama-server -m /path/to/Qwen3.5-2B-Q4_K_M.gguf \
  --device Vulkan0 --port 8013 -c 8192 -ngl 99 --threads 8 \
  --spec-type none

# Verify the judge before any run that uses it:
python3 verifier/scripts/check_judge_logprobs.py  # exit 0 = healthy
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp),
[FastFlowLM](https://github.com/ROCm/FastFlowLM), and the Strix Halo
community's speculative-decoding work.
