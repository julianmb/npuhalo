#!/usr/bin/env python3
"""
FastFlowLM / NPU Final-Step Logit Extraction Adapter.

Exposes a lightweight shim that queries the NPU runtime for the final-step
unnormalized logits, extracts the 20 A-T label token logits (or indices),
computes normalized logprobs locally, and formats an OpenAI-compatible response.
"""

import math
from typing import Dict, List, Tuple

# Standard A-T token vocabulary IDs for LFM2.5 and Qwen2.5
LETTERS = [chr(65 + i) for i in range(20)] # A through T (A=20 pts, T=1 pt)

def compute_at_logprobs_from_logits(
    logits: List[float],
    token_id_map: Dict[str, int]
) -> Tuple[Dict[str, float], float]:
    """
    Given full vocabulary logits at the final step, extract the 20 A-T tokens,
    compute softmax over only the valid score vocabulary, and calculate
    the expected reward score on [0.0, 1.0].
    
    Returns:
        (letter_logprobs, normalized_expected_score)
    """
    # 1. Extract raw logits z_i for each letter A-T
    letter_logits = {}
    for letter in LETTERS:
        tok_id = token_id_map.get(letter)
        if tok_id is not None and tok_id < len(logits):
            letter_logits[letter] = logits[tok_id]
        else:
            letter_logits[letter] = -1e9

    # 2. Stable softmax over the 20 tokens
    max_logit = max(letter_logits.values())
    exp_logits = {k: math.exp(v - max_logit) for k, v in letter_logits.items()}
    sum_exp = sum(exp_logits.values())
    
    probs = {k: v / sum_exp for k, v in exp_logits.items()}
    logprobs = {k: math.log(max(p, 1e-12)) for k, p in probs.items()}

    # 3. Expected score: A=20, B=19, ..., T=1
    expected_pts = sum((20 - i) * probs[c] for i, c in enumerate(LETTERS))
    normalized_score = (expected_pts - 1.0) / 19.0 # Scale [0.0, 1.0]

    return logprobs, normalized_score

def format_openai_logprob_response(
    chosen_letter: str,
    letter_logprobs: Dict[str, float]
) -> dict:
    """
    Construct an OpenAI-compatible choices logprob payload that drop-in replaces
    the upstream llm-as-a-verifier logprob expectation parser.
    """
    top_logprobs_list = [
        {"token": k, "logprob": v, "bytes": [ord(c) for c in k]}
        for k, v in sorted(letter_logprobs.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return {
        "content": [
            {
                "token": chosen_letter,
                "logprob": letter_logprobs.get(chosen_letter, -0.01),
                "top_logprobs": top_logprobs_list
            }
        ]
    }

if __name__ == "__main__":
    # Unit test / verification demonstration
    print("NPU Logit Adapter Specification Initialized.")
    mock_token_map = {chr(65+i): 100+i for i in range(20)}
    mock_logits = [0.0] * 200
    mock_logits[100] = 5.2 # Letter A
    mock_logits[104] = 3.1 # Letter E
    
    logprobs, score = compute_at_logprobs_from_logits(mock_logits, mock_token_map)
    print(f"Calculated Score: {score:.4f}")
    payload = format_openai_logprob_response("A", logprobs)
    print("Sample OpenAI formatted logprob payload:", payload["content"][0]["top_logprobs"][:3])
