#!/usr/bin/env python3
"""
benchmark_router.py — Evaluates the Hybrid NPU Query Router against ground-truth labels.
Tests Near-Term Roadmap Item #2: Improving router accuracy from the 25% baseline.
"""

import asyncio
import json
from npu_router import HybridNPURouter

TEST_DATASET = [
    # Simple Factual / Conversational (Should route to NPU)
    {"prompt": "Hello! How are you doing today?", "expected": "npu", "category": "Greeting"},
    {"prompt": "What is the capital of France?", "expected": "npu", "category": "Fact"},
    {"prompt": "What is the capital of Germany?", "expected": "npu", "category": "Fact"},
    {"prompt": "Who wrote Romeo and Juliet?", "expected": "npu", "category": "Fact"},
    {"prompt": "What is 15 * 24?", "expected": "npu", "category": "Arithmetic"},
    {"prompt": "What is 100 / 4?", "expected": "npu", "category": "Arithmetic"},
    {"prompt": "What is 7 + 8?", "expected": "npu", "category": "Arithmetic"},
    {"prompt": "Good morning assistant!", "expected": "npu", "category": "Greeting"},
    {"prompt": "Translate 'hello world' to Spanish", "expected": "npu", "category": "Translation"},
    {"prompt": "What is the capital of Australia?", "expected": "npu", "category": "Fact"},

    # Complex / Coding / Agentic (Must route to GPU)
    {"prompt": "Write a python function to reverse a linked list.", "expected": "gpu", "category": "Coding"},
    {"prompt": "Implement a binary search tree in C++ with deletion.", "expected": "gpu", "category": "Coding"},
    {"prompt": "Explain in detail the mathematical derivation of backpropagation.", "expected": "gpu", "category": "Reasoning"},
    {"prompt": "Debug this code: def add(a, b): return a - b", "expected": "gpu", "category": "Coding"},
    {"prompt": "Write a docker compose file for PostgreSQL and Redis.", "expected": "gpu", "category": "DevOps"},
    {"prompt": "Refactor this SQL query using window functions: SELECT * FROM sales", "expected": "gpu", "category": "Database"},
    {"prompt": "Write a bash script to monitor disk space and email alerts.", "expected": "gpu", "category": "Scripting"},
    {"prompt": "Compare and contrast Paxos and Raft consensus algorithms.", "expected": "gpu", "category": "Architecture"},
    {"prompt": "Prove that the square root of 2 is irrational.", "expected": "gpu", "category": "Math Proof"},
    {"prompt": "Create an API endpoint in FastAPI that accepts user registration.", "expected": "gpu", "category": "Coding"},
]

async def main():
    router = HybridNPURouter()
    await router.start()

    print("=" * 80)
    print("HYBRID NPU INTENT ROUTER BENCHMARK (ROADMAP ITEM #2)")
    print("=" * 80)

    correct = 0
    total = len(TEST_DATASET)
    latencies = []
    by_category = {}

    for i, item in enumerate(TEST_DATASET):
        prompt = item["prompt"]
        expected = item["expected"]
        cat = item["category"]

        res = await router.classify(prompt)
        route = res.get("route")
        lat = res.get("latency_ms", 0.0)
        method = res.get("method", "")
        latencies.append(lat)

        is_correct = (route == expected)
        if is_correct:
            correct += 1

        by_category.setdefault(cat, {"correct": 0, "total": 0})
        by_category[cat]["total"] += 1
        if is_correct:
            by_category[cat]["correct"] += 1

        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"[{i+1:2d}/{total}] {status} | Lat: {lat:6.1f}ms | Route: {route:<3} (Exp: {expected:<3}) | Method: {method:<20} | '{prompt[:45]}'")

    await router.close()

    acc = (correct / total) * 100
    avg_lat = sum(latencies) / len(latencies)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    print(f"Overall Decision Accuracy : {correct}/{total} ({acc:.1f}%) [Prior Baseline: 25.0%]")
    print(f"Average Routing Overhead   : {avg_lat:.2f} ms")
    print(f"Zero-Latency Rule Matches : {sum(1 for l in latencies if l < 5.0)} / {total}")
    print("-" * 80)
    print("Breakdown by Category:")
    for cat, d in by_category.items():
        c_acc = (d["correct"] / d["total"]) * 100
        print(f"  • {cat:<15}: {d['correct']}/{d['total']} ({c_acc:.0f}%)")
    print("=" * 80 + "\n")

    # Save results
    out_file = "/home/user/source/npuhalo/docs/router_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "accuracy_pct": acc,
            "baseline_pct": 25.0,
            "avg_latency_ms": avg_lat,
            "correct": correct,
            "total": total,
            "by_category": by_category,
        }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
