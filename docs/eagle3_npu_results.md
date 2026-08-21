# EAGLE-3 and NPU Drafting: Archived Experiment Summary

> **Status: superseded.** This file preserves a short index to the EAGLE-3
> investigation without retaining stale recommendations. The reconciled result
> is in [`final_verdict.md`](final_verdict.md).

## Outcome

EAGLE-3 conversion and acceptance were tested after the original implementation
spike. A custom vocabulary recovered acceptance, but separate-drafter weight
streaming and dispatch costs still kept decode below embedded MTP on the tested
Strix Halo configuration.

| Configuration | Measured decode | Outcome |
|---|---:|---|
| Embedded MTP | 33.8–38.2 tok/s | Best measured configuration |
| Full EAGLE-3 head | 19.8 tok/s | Weight-streaming bound |
| Custom compressed EAGLE-3 Q4 head | 24.9 tok/s | Acceptance recovered; still slower |

The earlier assumption that the full projection/output path could live entirely
in NPU tile SRAM or GPU MALL was incorrect. The required tensors exceed those
on-chip capacities and must access shared memory.

## Related Evidence

- [`eagle3_spike_results.md`](eagle3_spike_results.md): conversion spike and measured variants.
- [`EAGLE3_QWEN38_ANALYSIS.md`](EAGLE3_QWEN38_ANALYSIS.md): byte-budget and acceptance analysis.
- [`final_verdict.md`](final_verdict.md): final contention measurements and decision.
- [`../scripts/npu_benchmark.py`](../scripts/npu_benchmark.py): benchmark utilities.

Local model artifacts are excluded from Git and are not distributed by this
repository.
