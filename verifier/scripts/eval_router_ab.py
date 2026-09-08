#!/usr/bin/env python3
"""
eval_router_ab.py — Router decision accuracy + NPU-direct vs GPU answer quality A/B.

Closes README caveat #8 ("Router speed is measured; router accuracy is not").

Part A: does the router send short-answerable queries to the NPU and reasoning
        queries to the GPU? (routing decision accuracy vs ground truth)
Part B: for short-answerable queries, compare answer correctness and latency of
        NPU-direct generation versus full GPU generation.

Grading is objective (normalized/numeric match against gold answers) on a small
hand-labeled set; results carry small-n caveats.

Usage: python3 verifier/scripts/eval_router_ab.py [--out results/router_ab.json]
"""

import argparse
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

GEN_URL = os.environ.get("ROUTER_AB_GEN", "http://127.0.0.1:8012/v1/chat/completions")
NPU_URL = os.environ.get("ROUTER_AB_NPU", "http://127.0.0.1:8001/v1/chat/completions")
NPU_MODEL = os.environ.get("ROUTER_AB_NPU_MODEL", "qwen3.5:0.8b")

ROUTER_PROMPT = (
    "Classify the following user request. Reply with a single JSON object with keys:\n"
    '  "intent": one of chat|code|math|classification|translation|toolcall\n'
    '  "length": short|medium|long\n'
    '  "route": npu|gpu\n'
    "Use route=npu for simple facts, greetings, or single-word answers; "
    "route=gpu for anything requiring reasoning, code, math, or long output.\n"
    "Request:\n"
)

# (query, gold_answer, should_route_to_npu)
DATASET = [
    ("What is the capital of Australia?", "canberra", True),
    ("What is 17 * 23?", "391", True),
    ("How many continents are there?", "7", True),
    ("What is the chemical symbol for gold?", "au", True),
    ("Who wrote Romeo and Juliet?", "shakespeare", True),
    ("What is 15% of 200?", "30", True),
    ("What year did World War II end?", "1945", True),
    ("How many legs does a spider have?", "8", True),
    ("What is the largest planet in our solar system?", "jupiter", True),
    ("What is the freezing point of water in Celsius?", "0", True),
    ("Say hello in French.", "bonjour", True),
    ("What is 12 squared?", "144", True),
    ("How many minutes are in 2 hours?", "120", True),
    ("What color do you get when you mix blue and yellow?", "green", True),
    ("What is the square root of 81?", "9", True),
    ("Hi, how are you today?", "", True),
    ("Thanks for your help!", "", True),
    ("What is the plural of mouse?", "mice", True),
    ("Which planet is known as the Red Planet?", "mars", True),
    ("How many sides does a hexagon have?", "6", True),
    ("Write a Python function to reverse a linked list.", "", False),
    ("Explain the tradeoffs of event sourcing architecture.", "", False),
    ("Debug this traceback: RecursionError maximum recursion depth exceeded in a quicksort.", "", False),
    ("Translate this legal clause into plain English and explain the implications.", "", False),
    ("Design a rate limiter for a distributed API gateway.", "", False),
    ("Why does my Rust borrow checker reject this lifetime pattern?", "", False),
    ("Refactor this class hierarchy using the strategy pattern.", "", False),
    ("Compare CAP theorem consistency models for a multi-region database.", "", False),
]


def http_chat(url: str, payload: dict, timeout: int = 90) -> tuple[str, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    dt = (time.time() - t0) * 1000
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    return text, dt


def classify_route(query: str) -> str:
    text, _ = http_chat(NPU_URL, {
        "model": "lfm2.5-tk:1.2b",
        "messages": [{"role": "user", "content": ROUTER_PROMPT + query}],
        "max_tokens": 96,
        "temperature": 0.0,
    })
    m = re.search(r'"route"\s*:\s*"?(npu|gpu)', text)
    return m.group(1) if m else "gpu"


def grade(answer: str, gold: str) -> bool:
    if not gold:
        return None  # ungradeable chitchat
    a = re.sub(r"[^a-z0-9. ]", "", answer.lower())
    g = re.sub(r"[^a-z0-9. ]", "", gold.lower())
    if re.fullmatch(r"-?\d+(\.\d+)?", gold):
        # numeric gold: match as a standalone number token anywhere in the answer
        pattern = rf"(?<![\d.]){re.escape(gold)}(?![\d])(?!\.\d)"
        return bool(re.search(pattern, answer))
    return g in a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(REPO, "verifier", "results", "router_ab.json"))
    args = parser.parse_args()

    # Warm endpoints
    http_chat(NPU_URL, {"model": NPU_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    http_chat(GEN_URL, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0})

    # ---- Part A: routing decision accuracy ----
    route_correct = 0
    route_rows = []
    for query, _, should_npu in DATASET:
        got = classify_route(query)
        ok = (got == "npu") == should_npu
        route_correct += ok
        route_rows.append({"query": query, "routed": got, "expected": "npu" if should_npu else "gpu", "correct": ok})
        print(f"[route] {'OK ' if ok else 'MISS'} {got:>3} <- {query[:60]}", flush=True)
    routing_accuracy = route_correct / len(DATASET)

    # ---- Part B: answer quality A/B on NPU-routed (short-answerable) set ----
    ab_rows = []
    npu_correct = gpu_correct = 0
    npu_gradable = gpu_gradable = 0
    npu_lat, gpu_lat = [], []
    for query, gold, should_npu in DATASET:
        if not should_npu:
            continue
        ans_npu, lat_npu = http_chat(NPU_URL, {
            "model": NPU_MODEL, "messages": [{"role": "user", "content": query}],
            "max_tokens": 64, "temperature": 0.0,
        })
        ans_gpu, lat_gpu = http_chat(GEN_URL, {
            "messages": [{"role": "user", "content": query}], "max_tokens": 1024, "temperature": 0,
        })
        g_npu, g_gpu = grade(ans_npu, gold), grade(ans_gpu, gold)
        if g_npu is not None:
            npu_gradable += 1
            npu_correct += g_npu
        if g_gpu is not None:
            gpu_gradable += 1
            gpu_correct += g_gpu
        npu_lat.append(lat_npu)
        gpu_lat.append(lat_gpu)
        ab_rows.append({"query": query, "gold": gold, "npu_answer": ans_npu[:200], "gpu_answer": ans_gpu[:200],
                        "npu_correct": g_npu, "gpu_correct": g_gpu,
                        "npu_latency_ms": round(lat_npu), "gpu_latency_ms": round(lat_gpu)})
        print(f"[ab] npu={'OK' if g_npu else 'MISS'}({lat_npu:.0f}ms) gpu={'OK' if g_gpu else 'MISS'}({lat_gpu:.0f}ms) {query[:50]}", flush=True)

    result = {
        "routing_decision_accuracy": round(routing_accuracy, 3),
        "routing_correct": route_correct,
        "routing_total": len(DATASET),
        "answer_accuracy": {
            "npu_direct": round(npu_correct / max(1, npu_gradable), 3),
            "gpu_full": round(gpu_correct / max(1, gpu_gradable), 3),
            "gradable_n": npu_gradable,
        },
        "mean_latency_ms": {
            "npu_direct": round(sum(npu_lat) / len(npu_lat)),
            "gpu_full": round(sum(gpu_lat) / len(gpu_lat)),
        },
        "caveats": [
            "Small hand-labeled set (n=%d short, %d long); objective grading only." % (
                sum(1 for d in DATASET if d[2]), sum(1 for d in DATASET if not d[2])),
            "GPU path pays Ornith thinking-token latency even on trivial queries.",
        ],
        "rows_routing": route_rows,
        "rows_ab": ab_rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("rows")}, indent=2))


if __name__ == "__main__":
    main()
