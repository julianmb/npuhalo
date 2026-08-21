# Historical NPU-to-iGPU Handoff Prototype

> **Status: archived experiment.** This document describes a handoff prototype,
> not validated speculative decoding and not a production deployment guide.
> See [`final_verdict.md`](final_verdict.md) for the measured decision.

## What It Did

The prototype in [`scripts/run_pipeline.py`](../scripts/run_pipeline.py) streamed
an initial burst from an NPU model, appended that text as an assistant
continuation, and asked the iGPU model to continue. It did **not** compare target
logits against draft tokens, compute an acceptance rate, or retract NPU tokens
already sent to the client. It is therefore a **handoff pipeline**, not a
speculative verifier.

Supporting experiments included:

- [`scripts/ipc/ring_buffer.py`](../scripts/ipc/ring_buffer.py): an experimental
  single-producer/single-consumer shared-memory queue.
- [`scripts/ipc/async_pipeline.py`](../scripts/ipc/async_pipeline.py): an
  asynchronous coordination prototype.
- [`scripts/export_eagle_head.py`](../scripts/export_eagle_head.py): an exporter
  scaffold; it does not contain trained EAGLE weights or an XDNA deployment.
- [`scripts/npu_benchmark.py`](../scripts/npu_benchmark.py): latency and
  contention benchmark utilities.

## Why It Was Archived

For the tested active-parameter-light MoE target, the iGPU generated tokens
faster than the NPU draft model. Cross-runtime orchestration added latency, and
acceptance rates were insufficient to recover that cost. The measured test was
approximately 2x slower than GPU-only execution.

Earlier drafts also assumed a projection head could reside entirely in NPU tile
SRAM or GPU cache. That assumption was not established and must not be used as
a deployment claim.

## Safe Use

These scripts are retained for reproducibility and further research. Services
bind to localhost by default. Do not expose them remotely without request
limits, authentication, and TLS. Treat generated-code execution as trusted-only
unless it runs inside a disposable container or VM.
