#!/usr/bin/env python3
"""
Ornith-1.5 (35B Agent) + LFM2.5-1.2B (Verifier / Judge) Pipeline.

1. Generator: Ornith-1.5-35B generates candidate solutions (rollouts) for coding tasks via OpenAI API or local engine.
2. Judge: LiquidAI/LFM2.5-1.2B-Thinking acts as the verifier/judge scoring each candidate.
3. Reranker: Computes continuous expected reward E[S] over A-T (20-point scale) and selects best candidate.
"""

import json
import math
import time
import torch
import urllib.request
import urllib.error
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = [chr(65 + i) for i in range(20)] # A (20 pts) to T (1 pt)

BENCHMARK_TASKS = [
    {
        "id": "task_01",
        "instruction": "Write a python function `is_valid_parentheses(s: str) -> bool` that checks if brackets '()', '{}', '[]' are closed properly and in order.",
        "test_code": """
assert is_valid_parentheses("()[]{}") == True
assert is_valid_parentheses("([)]") == False
assert is_valid_parentheses("{[]}") == True
assert is_valid_parentheses("]") == False
assert is_valid_parentheses("") == True
"""
    },
    {
        "id": "task_02",
        "instruction": "Write a python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping intervals.",
        "test_code": """
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([]) == []
"""
    },
    {
        "id": "task_03",
        "instruction": "Write a python function `find_peak_element(nums: list[int]) -> int` that returns index of any peak element where nums[i] > nums[i+1] and nums[i] > nums[i-1] in O(log n) time.",
        "test_code": """
assert find_peak_element([1,2,3,1]) == 2
assert find_peak_element([1,2,1,3,5,6,4]) in [1, 5]
"""
    }
]

class OrnithGenerator:
    """Client for querying Ornith-1.5-35B via OpenAI API or fallback rollouts."""
    def __init__(self, endpoint_url="http://127.0.0.1:8012/v1/chat/completions", model_name="Ornith-1.5-35B"):
        self.endpoint_url = endpoint_url
        self.model_name = model_name

    def generate_candidate(self, task_instruction: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are Ornith-1.5, an expert coding agent. Provide only the python code implementation."},
                {"role": "user", "content": task_instruction}
            ],
            "temperature": temperature,
            "max_tokens": 512
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res["choices"][0]["message"]["content"]
        except Exception:
            return None

class LFMVerifierJudge:
    def __init__(self, model_id="LiquidAI/LFM2.5-1.2B-Thinking"):
        print(f"[*] Initializing LFM2.5 Verifier Judge ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.model.eval()
        self.t_raw = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
        self.t_sp = [self.tokenizer.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS]
        print("[+] LFM2.5 Judge is ready.")

    def score_candidate(self, task_instruction: str, candidate_code: str) -> Dict[str, Any]:
        prompt = (
            f"Task:\n{task_instruction}\n\n"
            f"Candidate Code Solution:\n{candidate_code}\n\n"
            "Evaluation Criterion — Code Correctness & Robustness:\n"
            "Rate how likely the candidate code solves the task correctly on a 20-point scale (A=best, T=worst)."
        )

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<score_A>"}
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
        for end_tok in ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "</s>"]:
            if formatted.endswith(end_tok):
                formatted = formatted[:-len(end_tok)]
        if not formatted.endswith("<score_A>"):
            formatted += "<score_A>"

        inputs = self.tokenizer(formatted, return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits[0, -1, :]
            logprobs = torch.log_softmax(logits, dim=-1)
        latency = time.time() - t0

        probs = {}
        for i, c in enumerate(LETTERS):
            p_r = math.exp(logprobs[self.t_raw[i]].item())
            p_s = math.exp(logprobs[self.t_sp[i]].item())
            probs[c] = max(p_r, p_s)

        tot = sum(probs.values())
        norm = {c: p / tot if tot > 0 else 0.0 for c, p in probs.items()}
        expected_val = sum((20 - i) * norm[c] for i, c in enumerate(LETTERS))
        norm_score = (expected_val - 1.0) / 19.0

        top_letter = max(norm.items(), key=lambda x: x[1])[0]

        return {
            "score": norm_score,
            "expected_pts": expected_val,
            "top_letter": top_letter,
            "top_letter_prob": norm[top_letter],
            "latency_ms": latency * 1000
        }

def run_pipeline(task_index: int = 0, num_candidates: int = 4):
    task = BENCHMARK_TASKS[task_index % len(BENCHMARK_TASKS)]
    print("\n=================================================================")
    print(" Ornith-1.5 (35B Agent) -> LiquidAI LFM2.5 (Verifier Judge)")
    print(f" Task: {task['instruction']}")
    print("=================================================================")

    generator = OrnithGenerator()
    judge = LFMVerifierJudge()

    # Generate or assemble candidate rollouts
    candidates = []
    print(f"\n[+] Sampling {num_candidates} rollouts from Ornith-1.5...")
    for i in range(num_candidates):
        live_code = generator.generate_candidate(task["instruction"], temperature=0.6 + i*0.2)
        if live_code:
            candidates.append({"id": f"rollout_{i+1}", "code": live_code, "type": "live_generation"})
        else:
            # High-fidelity mock rollouts covering varying solution qualities
            if i == 0:
                code = "def is_valid_parentheses(s: str) -> bool:\n    stack = []\n    m = {')':'(', '}':'{', ']':'['}\n    for c in s:\n        if c in m:\n            if not stack or stack.pop() != m[c]: return False\n        else: stack.append(c)\n    return not stack"
            elif i == 1:
                code = "def is_valid_parentheses(s: str) -> bool:\n    while '()' in s or '{}' in s or '[]' in s:\n        s = s.replace('()', '').replace('{}', '').replace('[]', '')\n    return len(s) == 0"
            elif i == 2:
                code = "def is_valid_parentheses(s: str) -> bool:\n    # Greedy counter (fails on order)\n    return s.count('(') == s.count(')') and s.count('[') == s.count(']')"
            else:
                code = "def is_valid_parentheses(s: str) -> bool:\n    # Incomplete\n    return True if len(s) % 2 == 0 else False"
            candidates.append({"id": f"rollout_{i+1}", "code": code, "type": "sample_rollout"})

    # Evaluate each rollout with LFM2.5 Judge
    print("\n[+] LFM2.5 Verifier is evaluating candidate rollouts:")
    scored = []
    for cand in candidates:
        eval_res = judge.score_candidate(task["instruction"], cand["code"])
        record = {**cand, **eval_res}
        scored.append(record)
        print(f"\n  Candidate [{cand['id']}]:")
        print(f"    Expected Score: {eval_res['score']:.4f} ({eval_res['expected_pts']:.1f}/20 pts)")
        print(f"    Top Predicted Token: {eval_res['top_letter']} (P={eval_res['top_letter_prob']:.3f})")
        print(f"    Verification Latency: {eval_res['latency_ms']:.1f} ms")

    # Sort candidates by LFM2.5 Judge score
    scored.sort(key=lambda x: x["score"], reverse=True)
    winner = scored[0]

    print("\n=================================================================")
    print(f" [*] VERIFIER JUDGE SELECTION: Candidate [{winner['id']}]")
    print(f" Expected Score: {winner['score']:.4f} / 1.0000")
    print(f" Score Margin over runner-up: {winner['score'] - scored[1]['score']:.4f}")
    print(" Code:")
    print("-----------------------------------------------------------------")
    print(winner["code"])
    print("=================================================================")

if __name__ == "__main__":
    run_pipeline()
