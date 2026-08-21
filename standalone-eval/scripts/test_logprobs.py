#!/usr/bin/env python3
"""
Phase 3: Token Logprob Compatibility Test Suite.
Tests LFM2.5-1.2B-Thinking and Qwen models on single-token score extraction,
verifying logprob availability, distribution normalization, and expected score computation.
"""

import math
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
GRANULARITY_5 = {
    "A": 1.0,  # clearly failed / incorrect
    "B": 2.0,
    "C": 3.0,  # uncertain
    "D": 4.0,
    "E": 5.0,  # clearly correct / succeeded
}

GRANULARITY_20 = {
    chr(65 + i): float(20 - i) for i in range(20)  # A=20 (best), ..., T=1 (worst)
}

TEST_CASES = [
    {
        "id": "correct_math",
        "description": "Clearly correct candidate",
        "prompt": "Evaluate whether the following mathematical derivation is correct:\nTask: Calculate 2 + 2\nCandidate Solution: 2 + 2 = 4.\nRate correctness from A (definitely incorrect) to E (definitely correct). Output exactly one letter: A, B, C, D, or E."
    },
    {
        "id": "incorrect_math",
        "description": "Clearly incorrect candidate",
        "prompt": "Evaluate whether the following mathematical derivation is correct:\nTask: Calculate 2 + 2\nCandidate Solution: 2 + 2 = 5.\nRate correctness from A (definitely incorrect) to E (definitely correct). Output exactly one letter: A, B, C, D, or E."
    },
    {
        "id": "ambiguous",
        "description": "Deliberately ambiguous candidate",
        "prompt": "Evaluate whether the candidate answered the question:\nQuestion: What is the optimal temperature for tea?\nCandidate Solution: Some people like hot tea, while others prefer iced tea depending on personal taste.\nRate correctness from A (definitely incorrect) to E (definitely correct). Output exactly one letter: A, B, C, D, or E."
    },
    {
        "id": "malformed",
        "description": "Malformed candidate",
        "prompt": "Evaluate the following program fix:\nCandidate Solution: ```python\ndef foo(:\n   return !!!\n```\nRate validity from A (definitely incorrect) to E (definitely correct). Output exactly one letter: A, B, C, D, or E."
    },
    {
        "id": "code_test_pass",
        "description": "Code patch with passing test evidence",
        "prompt": "Evaluate the following code patch and test output:\nPatch: Fixed off-by-one error in binary search by changing `high = mid` to `high = mid - 1`.\nTest Output: 14/14 tests PASSED (100% test suite green).\nRate candidate quality from A (definitely incorrect) to E (definitely correct). Output exactly one letter: A, B, C, D, or E."
    },
    {
        "id": "code_test_fail",
        "description": "Code patch with explicit test failure evidence",
        "prompt": "Evaluate the following code patch and test output:\nPatch: Modified regex in parser.\nTest Output: AssertionError: expected 'int' but got 'NoneType'. 3 failed, 11 passed.\nRate candidate quality from A (definitely incorrect) to E (definitely correct). Output exactly one letter: A, B, C, D, or E."
    }
]

def run_logprob_tests():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()

    results = []

    # Map target letter tokens
    letters = ["A", "B", "C", "D", "E"]
    letter_token_ids = {}
    for l in letters:
        # Check raw and leading space
        t_raw = tokenizer.encode(l, add_special_tokens=False)
        t_sp = tokenizer.encode(" " + l, add_special_tokens=False)
        letter_token_ids[l] = {
            "raw_id": t_raw[0] if len(t_raw) == 1 else None,
            "sp_id": t_sp[0] if len(t_sp) == 1 else None
        }

    print("\nTarget Letter Token IDs:")
    for l, ids in letter_token_ids.items():
        print(f"  {l}: raw={ids['raw_id']}, with_space={ids['sp_id']}")

    for tc in TEST_CASES:
        print(f"\nRunning test case: {tc['id']} ({tc['description']})")
        # Format chat prompt
        messages = [{"role": "user", "content": tc["prompt"]}]
        try:
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_str = f"User: {tc['prompt']}\nAssistant: Rating: "

        inputs = tokenizer(prompt_str, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            logprobs = torch.log_softmax(next_token_logits, dim=-1)

        # Get top-20 tokens
        topk_vals, topk_indices = torch.topk(logprobs, 20)
        topk_list = []
        for val, idx in zip(topk_vals.tolist(), topk_indices.tolist()):
            tok_str = tokenizer.decode([idx])
            topk_list.append({"id": idx, "token": tok_str, "logprob": val})

        # Calculate probabilities across A, B, C, D, E
        letter_probs = {}
        for l in letters:
            raw_id = letter_token_ids[l]["raw_id"]
            sp_id = letter_token_ids[l]["sp_id"]
            lp_raw = logprobs[raw_id].item() if raw_id is not None else -100.0
            lp_sp = logprobs[sp_id].item() if sp_id is not None else -100.0
            
            # Combine or take max
            p = max(math.exp(lp_raw), math.exp(lp_sp))
            letter_probs[l] = p

        total_p = sum(letter_probs.values())
        norm_probs = {l: p / total_p if total_p > 0 else 0.0 for l, p in letter_probs.items()}

        # Compute continuous expected score on 1-5 scale normalized to [0, 1]
        # E(score) = sum(val * p)
        exp_score_5 = sum(GRANULARITY_5[l] * norm_probs[l] for l in letters)
        norm_score_01 = (exp_score_5 - 1.0) / 4.0  # (score - min) / (max - min)

        best_token = topk_list[0]["token"]

        case_res = {
            "id": tc["id"],
            "description": tc["description"],
            "top_token": best_token,
            "top_20_logprobs": topk_list,
            "raw_letter_probs": letter_probs,
            "normalized_letter_probs": norm_probs,
            "expected_score_1_to_5": exp_score_5,
            "normalized_score_0_to_1": norm_score_01
        }
        results.append(case_res)

        print(f"  Top Token: {repr(best_token)}")
        print("  Norm Probs: " + ", ".join(f"{l}: {norm_probs[l]:.4f}" for l in letters))
        print(f"  Expected Score (1-5): {exp_score_5:.3f} | Normalized [0,1]: {norm_score_01:.3f}")

    out_json = "standalone-eval/results/logprob-results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw machine-readable results to {out_json}")

if __name__ == "__main__":
    run_logprob_tests()
