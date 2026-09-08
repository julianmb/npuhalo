# Public Research Roadmap

`npuhalo` explores useful sidecar workloads for the AMD XDNA 2 NPU while a
larger generator occupies the Strix Halo iGPU. Measured evidence supports
continued work on conservative anomaly triage and compression of very large
tool outputs; routing and TTFT handoff are currently blocked by measured
deficits (see the README's Honest Caveats).

## Near Term

1. Complete the active rollback sweep across multiple seeds (in progress).
2. Improve router classification accuracy from the measured 25% baseline — candidates: larger classifier model, fine-tuned intent head, or deterministic rules plus NPU fallback.
3. Reduce compressor extraction latency (~3–4 s) via non-thinking decoding or grammar-constrained output, lowering the ~32K-char breakeven point.
4. Publish sanitized aggregate telemetry and a machine-readable result schema.
5. Measure verifier behavior on real agent traffic rather than synthetic-only tasks.
6. Upstream the published FastFlowLM logprob patch ([`patches/`](../patches/)) to the FastFlowLM repository.

## Engineering

1. Replace trusted-process evaluation with a documented container sandbox.
2. Add readiness checks, bounded requests, and authentication guidance for network services.
3. Move importable modules into an explicit `npuhalo` package namespace.
4. Version model, runtime, driver, prompt, and dataset hashes with every benchmark.
5. Document or work around the ROCmFPX `llama-server` `seq_rm` abort observed under rapid varying-length prefill loads.

## Research Boundaries

- NPU-to-iGPU speculative decoding is archived for the tested model/runtime pair because it was slower than GPU-only generation.
- The NPU-to-GPU TTFT handoff was re-measured in 2026-08 and loses to direct GPU generation on both short and long prompts; it remains archived.
- Query routing is blocked pending a classifier that beats the measured 25% decision accuracy.
- Shadow-mode detection is measured; successful abort-and-resample conversion is not.
- The NPU verifier is a triage filter in the observed runs and should not be treated as an independent abort authority.
