# AGENTS.md — npuhalo

## What this repo is
NPU + iGPU research on AMD Strix Halo (XDNA 2, /dev/accel/accel0):
github.com/julianmb/npuhalo. Live verification, compression and routing
experiments; findings feed the NPU sections of q38rocm/halofpx docs.

## Status / gotchas
1. Working tree often holds **active experiment scripts and results**
   (`verifier/results/*`, `run_active_recovery.py` etc.). Never `git add -A`;
   stage only the files you intentionally changed.
2. Established verdict (do not re-litigate without new evidence): the NPU
   does NOT improve sustained decode — embedded MTP wins. Its value is
   ~1.8x TTFT on long prompts and low-power intent routing.
3. NPU runtime comes via Lemonade's FastFlowLM backend (`flm:npu`), not XRT
   directly. XRT is built from q38rocm's `xdna-driver` submodule.
4. Draft models used here (Qwen3.5-0.8B/2B) live in `~/source/halofpx-research/o4a/models/draft/`.
