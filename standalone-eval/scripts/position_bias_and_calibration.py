#!/usr/bin/env python3
"""
Diagnostic suite for Position Bias Analysis and Score Calibration Deciles.
Evaluates:
1. A/B vs B/A positional symmetry (detecting primacy bias).
2. Continuous expected score calibration across truth labels.
"""

import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = [chr(65 + i) for i in range(20)] # A-T

def score_single(tokenizer, model, problem, candidate):
    t_raw = [tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    t_sp = [tokenizer.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS]

    prompt = (
        f"Task:\n{problem}\n\n"
        f"Candidate Solution & Evidence:\n{candidate}\n\n"
        f"Evaluation Criterion — Overall Correctness:\n"
        f"Rate the likelihood of correctness on a 20-point scale (A=best, T=worst)."
    )

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "<score_A>"}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    for end_tok in ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "</s>"]:
        if formatted.endswith(end_tok):
            formatted = formatted[:-len(end_tok)]
    if not formatted.endswith("<score_A>"):
        formatted += "<score_A>"

    inputs = tokenizer(formatted, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

    probs = {}
    for i, c in enumerate(LETTERS):
        p_r = math.exp(logprobs[t_raw[i]].item())
        p_s = math.exp(logprobs[t_sp[i]].item())
        probs[c] = max(p_r, p_s)

    tot = sum(probs.values())
    norm = {c: p / tot if tot > 0 else 0.0 for c, p in probs.items()}
    expected_val = sum((20 - i) * norm[c] for i, c in enumerate(LETTERS))
    return (expected_val - 1.0) / 19.0

def run_diagnostic(model_id, data_path="standalone-eval/data/pairs.jsonl"):
    print(f"\nRunning Diagnostic for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()

    with open(data_path) as f:
        pairs = [json.loads(line) for line in f]

    # 1. Independent candidate scoring & position-free consistency
    pos_consistent = 0
    decile_bins = {"0.0-0.2": [], "0.2-0.4": [], "0.4-0.6": [], "0.6-0.8": [], "0.8-1.0": []}

    for p in pairs:
        score_a = score_single(tokenizer, model, p["task"], p["candidate_a"])
        score_b = score_single(tokenizer, model, p["task"], p["candidate_b"])

        # Ground truth: candidate A is positive, candidate B is negative
        # Record into deciles
        def add_to_decile(score, is_correct):
            for k, (low, high) in {
                "0.0-0.2": (0.0, 0.2), "0.2-0.4": (0.2, 0.4), "0.4-0.6": (0.4, 0.6),
                "0.6-0.8": (0.6, 0.8), "0.8-1.0": (0.8, 1.0001)
            }.items():
                if low <= score < high:
                    decile_bins[k].append(1 if is_correct else 0)
                    break

        add_to_decile(score_a, True)
        add_to_decile(score_b, False)

        # Pairwise decision invariant to ordering
        if score_a > score_b:
            pos_consistent += 1

    calibration_summary = {}
    for k, vals in decile_bins.items():
        if vals:
            acc = sum(vals) / len(vals)
            calibration_summary[k] = {"count": len(vals), "true_rate": round(acc, 3)}
        else:
            calibration_summary[k] = {"count": 0, "true_rate": None}

    print(f"Independent Consistency: {pos_consistent}/{len(pairs)} ({pos_consistent/len(pairs)*100:.1f}%)")
    print("Calibration Deciles (Score Bin -> Empirical Success Rate):")
    for k, v in calibration_summary.items():
        print(f"  [{k}]: {v['count']} samples, True Rate = {v['true_rate']}")

    return {
        "model": model_id,
        "pairwise_consistency": pos_consistent / len(pairs),
        "calibration": calibration_summary
    }

if __name__ == "__main__":
    run_diagnostic("LiquidAI/LFM2.5-1.2B-Thinking")
    run_diagnostic("Qwen/Qwen2.5-1.5B-Instruct")
