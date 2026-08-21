#!/usr/bin/env python3
"""
Full comparative benchmark evaluating:
- LiquidAI/LFM2.5-1.2B-Thinking
- Qwen/Qwen2.5-1.5B-Instruct
- Hard-rule baseline
- Random baseline
"""

import json
import math
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_PATH = "standalone-eval/data/pairs.jsonl"
OUT_JSON = "standalone-eval/results/benchmark.json"

CRITERIA = {
    "holistic": [
        {
            "name": "Overall Trajectory Correctness",
            "desc": "Rate how likely the candidate trajectory solved the task correctly based on all code and test evidence."
        }
    ],
    "decomposed": [
        {
            "name": "Specification Compliance",
            "desc": "Evaluate whether the candidate directly addressed the user task requirement."
        },
        {
            "name": "Execution & Test Evidence",
            "desc": "Evaluate verified test passing, error rates, and absence of runtime exceptions."
        },
        {
            "name": "Patch Correctness & Scope",
            "desc": "Evaluate whether the code patch is minimal, correct, and free of scope creep."
        },
        {
            "name": "Contradiction & Failure Signals",
            "desc": "Identify any contradictions between claimed success and underlying traceback/failure evidence."
        }
    ]
}

def load_model_and_tokenizer(model_id):
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

def eval_candidate(tokenizer, model, problem, candidate, criterion_name, criterion_desc):
    letters = [chr(65 + i) for i in range(20)] # A-T
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
    tok_count = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
    lat = time.time() - t0

    probs = {}
    for i, c in enumerate(letters):
        p_r = math.exp(logprobs[t_raw[i]].item())
        p_s = math.exp(logprobs[t_sp[i]].item())
        probs[c] = max(p_r, p_s)

    tot = sum(probs.values())
    norm = {c: p / tot if tot > 0 else 0.0 for c, p in probs.items()}
    expected_val = sum((20 - i) * norm[c] for i, c in enumerate(letters))
    norm_score = (expected_val - 1.0) / 19.0

    return norm_score, lat, tok_count

def run_model_eval(model_id, pairs):
    tokenizer, model = load_model_and_tokenizer(model_id)
    results = {}

    for mode in ["holistic", "decomposed"]:
        crit_list = CRITERIA[mode]
        correct = 0
        ties = 0
        total_lat = 0.0
        total_tokens = 0
        margins = []
        records = []

        for p in pairs:
            scores_a = []
            scores_b = []
            for crit in crit_list:
                sa, la, ta = eval_candidate(tokenizer, model, p["task"], p["candidate_a"], crit["name"], crit["desc"])
                sb, lb, tb = eval_candidate(tokenizer, model, p["task"], p["candidate_b"], crit["name"], crit["desc"])
                scores_a.append(sa)
                scores_b.append(sb)
                total_lat += (la + lb)
                total_tokens += (ta + tb)

            mean_a = sum(scores_a) / len(scores_a)
            mean_b = sum(scores_b) / len(scores_b)
            diff = mean_a - mean_b
            margin = abs(diff)
            margins.append(margin)

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

            records.append({
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
        eval_count = len(pairs) * len(crit_list) * 2
        avg_lat_ms = (total_lat / eval_count) * 1000
        tok_s = total_tokens / total_lat if total_lat > 0 else 0.0

        print(f"[{model_id}] {mode.upper()}: Acc={acc*100:.1f}%, Ties={tie_rate*100:.1f}%, Margin={avg_margin:.4f}, Latency={avg_lat_ms:.1f}ms, Tok/s={tok_s:.1f}")

        results[mode] = {
            "accuracy": acc,
            "correct_count": correct,
            "total_pairs": len(pairs),
            "tie_rate": tie_rate,
            "mean_margin": avg_margin,
            "avg_latency_ms": avg_lat_ms,
            "tokens_per_second": tok_s,
            "records": records
        }

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results

def main():
    with open(DATA_PATH) as f:
        pairs = [json.loads(line) for line in f]

    print(f"Loaded {len(pairs)} benchmark pairs.")

    benchmark_data = {
        "metadata": {
            "num_pairs": len(pairs),
            "date": "2026-08-20",
            "scale": "A-T (20-point granularity)",
            "host": "AMD Strix Halo (Ryzen AI Max+ 395)"
        },
        "models": {}
    }

    # 1. Evaluate LFM2.5
    benchmark_data["models"]["LiquidAI/LFM2.5-1.2B-Thinking"] = run_model_eval(
        "LiquidAI/LFM2.5-1.2B-Thinking", pairs
    )

    # 2. Evaluate Qwen2.5-1.5B
    benchmark_data["models"]["Qwen/Qwen2.5-1.5B-Instruct"] = run_model_eval(
        "Qwen/Qwen2.5-1.5B-Instruct", pairs
    )

    # 3. Hard-rule baseline
    from baseline_hardrules import evaluate_hardrules
    hard_res = evaluate_hardrules(DATA_PATH)
    benchmark_data["baselines"] = {
        "hard_rules": hard_res,
        "random": {"accuracy": 0.50, "tie_rate": 0.0}
    }

    with open(OUT_JSON, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nAll benchmark results successfully written to {OUT_JSON}")

if __name__ == "__main__":
    main()
