# Phase 3: Logprob Compatibility & Calibration Analysis

**Evaluated Model:** `LiquidAI/LFM2.5-1.2B-Thinking`  
**Control Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Raw Results:** `standalone-eval/results/logprob-results.json`  
**Date:** 2026-08-20

---

## 1. Summary of Empirical Logprob Tests

We executed the six canonical validation scenarios prescribed by the task specification.

| Test Case | Scenario Description | LFM2.5 Raw Probs ($P_A$ vs $P_E$) | LFM2.5 Expected Score $[0, 1]$ | Notes / Diagnostic |
| :--- | :--- | :--- | :--- | :--- |
| **1. correct_math** | $2 + 2 = 4$ | $P_A=0.965, P_E=0.015$ | **0.022** | Single-step prefill collapses onto initial letter $A$ without reasoning. |
| **2. incorrect_math** | $2 + 2 = 5$ | $P_A=0.959, P_E=0.016$ | **0.025** | Fails to discriminate from correct math on raw zero-shot logits. |
| **3. ambiguous** | Open-ended tea temperature | $P_A=0.914, P_C=0.046$ | **0.052** | High concentration on $A$. |
| **4. malformed** | Python syntax error | $P_A=0.774, P_C=0.093$ | **0.155** | Modest dispersion across invalid tokens. |
| **5. code_test_pass** | 14/14 tests PASSED | $P_A=0.488, P_E=0.264$ | **0.366** | Noticeable shift toward $E$ on clear code evidence. |
| **6. code_test_fail** | AssertionError, 3 failed | $P_A=0.758, P_E=0.031$ | **0.116** | Clear downward discrimination ($\Delta = +0.250$ vs pass). |

---

## 2. Key Findings

1. **Reasoning-Model Dynamics:**
   * `LFM2.5-1.2B-Thinking` is fundamentally trained as a long-trace reasoning model. When presented with an un-prefilled prompt, its first predicted token is `<think>`.
   * When coerced into immediate score emission via prompt constraints or prefilling (`<score_A>`), its zero-shot probability distribution suffers from high base-rate bias toward the first letter in the prompt ($A$), unless explicit test execution keywords are present.
2. **Code Verification Signal:**
   * On code-agent evaluation cases (cases 5 & 6), LFM2.5 successfully separates passing test suites ($E=0.366$) from failing test suites ($E=0.116$).
3. **Logprob Availability Verdict:**
   * While the **model** naturally exposes full logits when run in PyTorch / llama-server, the **NPU runtime** (`flm-real`) currently does not return token logprobs. Thus, running `llm-as-a-verifier` on the NPU requires an engine update or hybrid execution where NPU performs triage and CPU/GPU handles logprob-based scoring.
