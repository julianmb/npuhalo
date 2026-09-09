# MiniCPM5-2B Port for AMD XDNA 2 NPU (FastFlowLM)

This directory contains the conversion scripts and verification tools used to port **[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)** to the AMD XDNA 2 NPU on FastFlowLM.

Ready-to-use quantized weights and compiled AIE firmware are available on Hugging Face:
👉 **[julianmb/MiniCPM5-2B-NPU2](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)**


> [!NOTE]
> **Upstream FastFlowLM Status (42-Layer Runlist Timeout):**
> * **Weights & Kernels**: Fully converted and verified with GQA $16:2 \to 16:8$ replication and unit RMSNorm injection.
> * **Prefill**: Passes cleanly on XDNA 2 (`chunk 1/1 with 38 tokens`).
> * **Decode**: FastFlowLM's `qwen3` engine (`libqwen3_npu.so`) batches all layer forward passes into a single monolithic `xrt::runlist`. While 24-layer (`Qwen3.5-0.8B`), 28-layer (`Qwen3-1.7B`), and 36-layer (`Qwen3-4B`) models decode cleanly, queuing 42 layers in one batch trips `ERT_CMD_STATE_TIMEOUT` during decode.
> * **Triage & Reproducer**: Tracked upstream in [ROCm/FastFlowLM#712](https://github.com/ROCm/FastFlowLM/issues/712). A standalone reproduction harness is available in the dedicated repo: [julianmb/minicpm5-xdna2](https://github.com/julianmb/minicpm5-xdna2).


---

### Conversion Pipeline
1. `expand_kv_heads.py`: Replicates dimension 0 of `k_proj` and `v_proj` $4\times$ (from 2 heads to 8 heads). Under GQA, this preserves exact mathematical equivalence while converting the attention ratio from $16:2$ ($8:1$) to $16:8$ ($2:1$), matching the native `_gen_mha_seq_d128_q2` AIE kernel.
2. Quantization to GGUF Q4_0 (`convert_hf_to_gguf.py` + `llama-quantize`).
3. Packing to AMD AIE layout (`ROCm/FLM_Q4NX_Converter`).
4. `inject_qk_norm.py`: Injects synthetic unit RMSNorm weights ($\gamma = 1.0$, $[128]$ BF16) for `q_norm` and `k_norm` across all 42 layers, routing safely through `libqwen3_npu.so`.
5. `test_quality.py`: Evaluates mathematical reasoning, code bug fixing, structured JSON extraction, and instruction following over FastFlowLM HTTP REST server.
