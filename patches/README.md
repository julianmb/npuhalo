# Patches

Out-of-tree modifications used for the measured results in this repository.
They are provided as unified diffs against the upstream sources at the time of
measurement; apply with `git apply <patch>` inside a checkout of the target
repository and expect to resolve offsets against newer revisions.

## `0002-llama-cpp-echo-prompt-logprobs.patch`

**Target:** llama.cpp (ROCmFPX fork lineage, `tools/server/`)

Stock llama-server rejects `echo=true`. This patch re-enables `echo` so that
prompt-token logprobs can be returned over the OpenAI-compatible API. It was
used to score prompt-prefix probability mass during escalator calibration.

## `0003-flm-logprobs-and-grammars.patch`

**Target:** FastFlowLM (`src/common/`, `src/include/`, `src/server/rest_handler.cpp`)

Stock FastFlowLM returns `"logprobs": null` over HTTP, which blocks
distribution-based verifier scoring on the NPU. This patch adds:

* Full-vocabulary log-softmax capture from raw NPU logits (248K vocab)
* OpenAI-compatible `logprobs` and `top_logprobs` on `/v1/chat/completions`
  and `/v1/completions`
* GBNF grammar-constrained decoding ported from llama.cpp
  (`src/common/grammar/`)

With this patch applied, the entire verifier tier runs on the NPU while the
iGPU stays dedicated to the primary generator.

## Reference tooling

The Python-side counterpart used to validate these endpoints lives in
[`../standalone-eval/scripts/npu_logit_adapter.py`](../standalone-eval/scripts/npu_logit_adapter.py)
and [`../standalone-eval/scripts/test_logprobs.py`](../standalone-eval/scripts/test_logprobs.py).
