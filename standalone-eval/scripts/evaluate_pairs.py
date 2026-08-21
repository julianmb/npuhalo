#!/usr/bin/env python3
"""
Phase 5: Pairwise Trajectory Benchmark Engine.
Evaluates LFM2.5-1.2B-Thinking and baseline models on pairwise candidate selection,
comparing K-repeats, holistic vs decomposed criteria, and continuous expected scores.
"""

import argparse
import json
import math
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DECOMPOSED_CRITERIA = [
    {
        "name": "Specification Compliance",
        "description": "Evaluate whether the candidate directly addressed the user task requirement."
    },
    {
        "name": "Execution & Test Evidence",
        "description": "Evaluate verified test passing, error rates, and absence of runtime exceptions."
    },
    {
        "name": "Patch Correctness & Scope",
        "description": "Evaluate whether the code patch is minimal, correct, and free of scope creep."
    },
    {
        "name": "Contradiction & Failure Signals",
        "description": "Identify any contradictions between claimed success and underlying traceback/failure evidence."
    }
]

HOLISTIC_CRITERION = {
    "name": "Overall Trajectory Correctness",
    "description": "Rate how likely the candidate trajectory solved the task correctly based on all available code and execution evidence."
}

def load_verifier_model(model_id):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def compute_expected_score(tokenizer, model, problem, candidate, criterion_name, criterion_desc):
    letters = [chr(65 + i) for i in range(20)] # A-T (A=20 ... T=1)
    t_raw = [tokenizer.encode(c, add_special_tokens=False)[0] for c in letters]
    t_sp = [tokenizer.encode(" " + c, add_special_tokens=False)[0] for c in letters]

    prompt = (
        f"Task:\n{problem}\n\n"
        f"Candidate Solution & Evidence:\n{candidate}\n\n"
        f"Evaluation Criterion — {criterion_name}:\n{criterion_desc}\n\n"
        "Rate the likelihood of correctness on a 20-point scale (A=best, T=worst)."
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
    input_tokens_count = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
    latency = time.time() - t0

    probs = {}
    for i, c in enumerate(letters):
        p_raw = math.exp(logprobs[t_raw[i]].item())
        p_sp = math.exp(logprobs[t_sp[i]].item())
        probs[c] = max(p_raw, p_sp)

    tot = sum(probs.values())
    norm = {c: p / tot if tot > 0 else 0.0 for c, p in probs.items()}
    # Expected value on [1, 20] scale, normalized to [0, 1]
    expected_val = sum((20 - i) * norm[c] for i, c in enumerate(letters))
    norm_score = (expected_val - 1.0) / 19.0

    return norm_score, latency, input_tokens_count

def run_benchmark(model_id, data_path="standalone-eval/data/pairs.jsonl", repeats_list=[1, 3, 5]):
    tokenizer, model = load_verifier_model(model_id)

    with open(data_path) as f:
        pairs = [json.loads(line) for line in f]

    all_results = {}

    for k in repeats_list:
        for mode in ["holistic", "decomposed"]:
            setting_key = f"K={k}_{mode}"
            print("\n==========================================")
            print(f"Running Benchmark: {model_id} | {setting_key}")
            print("==========================================")

            correct = 0
            ties = 0
            total_lat = 0.0
            total_tokens = 0
            margins = []
            pair_records = []

            for idx, p in enumerate(pairs):
                # Run evaluation for candidate A and B
                scores_a = []
                scores_b = []

                criteria_to_run = DECOMPOSED_CRITERIA if mode == "decomposed" else [HOLISTIC_CRITERION]

                for rep in range(k):
                    for crit in criteria_to_run:
                        sa, lat_a, tok_a = compute_expected_score(
                            tokenizer, model, p["task"], p["candidate_a"], crit["name"], crit["description"]
                        )
                        sb, lat_b, tok_b = compute_expected_score(
                            tokenizer, model, p["task"], p["candidate_b"], crit["name"], crit["description"]
                        )
                        scores_a.append(sa)
                        scores_b.append(sb)
                        total_lat += (lat_a + lat_b)
                        total_tokens += (tok_a + tok_b)

                mean_a = sum(scores_a) / len(scores_a)
                mean_b = sum(scores_b) / len(scores_b)
                diff = mean_a - mean_b
                margin = abs(diff)
                margins.append(margin)

                # Tolerance for tie: 0.005
                if diff > 0.005:
                    chosen = "A"
                elif diff < -0.005:
                    chosen = "B"
                else:
                    chosen = "TIE"

                is_correct = (chosen == p["preferred"])
                if is_correct:
                    correct += 1
                elif chosen == "TIE":
                    ties += 1

                pair_records.append({
                    "id": p["id"],
                    "preferred": p["preferred"],
                    "chosen": chosen,
                    "score_a": round(mean_a, 4),
                    "score_b": round(mean_b, 4),
                    "margin": round(margin, 4),
                    "is_correct": is_correct
                })

            acc = correct / len(pairs)
            tie_rate = ties / len(pairs)
            avg_margin = sum(margins) / len(margins)
            avg_lat_per_call = total_lat / (len(pairs) * k * len(criteria_to_run) * 2)
            tok_s = total_tokens / total_lat if total_lat > 0 else 0.0

            print(f"Result for {setting_key}:")
            print(f"  Accuracy: {correct}/{len(pairs)} ({acc*100:.1f}%)")
            print(f"  Tie Rate: {ties}/{len(pairs)} ({tie_rate*100:.1f}%)")
            print(f"  Mean Confidence Margin: {avg_margin:.4f}")
            print(f"  Latency/eval: {avg_lat_per_call*1000:.1f} ms | Throughput: {tok_s:.1f} tok/s")

            all_results[setting_key] = {
                "accuracy": acc,
                "correct_count": correct,
                "total_pairs": len(pairs),
                "tie_rate": tie_rate,
                "mean_margin": avg_margin,
                "avg_call_latency_ms": avg_lat_per_call * 1000,
                "tokens_per_second": tok_s,
                "records": pair_records
            }

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Thinking")
    parser.add_argument("--output", default="standalone-eval/results/benchmark.json")
    args = parser.parse_args()

    results = run_benchmark(args.model)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {args.output}")
