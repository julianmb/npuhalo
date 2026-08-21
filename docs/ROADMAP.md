# Public Research Roadmap

`npuhalo` explores useful sidecar workloads for the AMD XDNA 2 NPU while a
larger generator occupies the Strix Halo iGPU. Current evidence supports
continued work on routing, compression, and conservative anomaly triage.

## Near Term

1. Complete the interrupted active rollback sweep across multiple seeds.
2. Publish sanitized aggregate telemetry and a machine-readable result schema.
3. Validate context-compression fidelity on larger, independently labeled tool outputs.
4. Measure verifier behavior on real agent traffic rather than synthetic-only tasks.
5. Upstream the published FastFlowLM logprob patch ([`patches/`](../patches/)) to the FastFlowLM repository.

## Engineering

1. Replace trusted-process evaluation with a documented container sandbox.
2. Add readiness checks, bounded requests, and authentication guidance for network services.
3. Move importable modules into an explicit `npuhalo` package namespace.
4. Version model, runtime, driver, prompt, and dataset hashes with every benchmark.

## Research Boundaries

- NPU-to-iGPU speculative decoding is archived for the tested model/runtime pair because it was slower than GPU-only generation.
- Shadow-mode detection is measured; successful abort-and-resample conversion is not.
- The NPU verifier is a triage filter in the observed runs and should not be treated as an independent abort authority.
