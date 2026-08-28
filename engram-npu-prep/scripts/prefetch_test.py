#!/usr/bin/env python3
"""
prefetch_test.py — Asynchronous N-Gram Lookup & GPU Layer Compute Overlap Simulation

Simulates and benchmarks the pipeline overlap:
  - GPU computes Layer 0..L-1 dense transformer forward passes
  - Asynchronous worker (NPU/CPU DMA) precomputes N-gram hash & gathers table embedding e_t
  - Measures exposed latency vs hidden latency across varying GPU layer compute times and table lookup latencies
"""

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch


def simulate_overlap_step(
    gpu_layer_compute_ms: float,
    lookup_latency_ms: float,
    engram_layer_idx: int = 1,
    num_layers: int = 30,
) -> Dict:
    """
    Simulates one token step where:
    - GPU computes Layer 0 (takes gpu_layer_compute_ms)
    - At start of Layer 0, asynchronous lookup is dispatched for Layer engram_layer_idx
    - GPU reaches Layer engram_layer_idx after (engram_layer_idx * gpu_layer_compute_ms)
    - If lookup_latency_ms <= available_window, lookup is 100% hidden (0 ms exposed latency)
    - If lookup_latency_ms > available_window, exposed latency = lookup_latency_ms - available_window
    """
    available_window_ms = engram_layer_idx * gpu_layer_compute_ms
    
    if lookup_latency_ms <= available_window_ms:
        exposed_latency_ms = 0.0
        hidden_latency_ms = lookup_latency_ms
        hidden_pct = 100.0
    else:
        exposed_latency_ms = lookup_latency_ms - available_window_ms
        hidden_latency_ms = available_window_ms
        hidden_pct = (hidden_latency_ms / lookup_latency_ms) * 100.0

    total_step_time_ms = (num_layers * gpu_layer_compute_ms) + exposed_latency_ms
    unoverlapped_time_ms = (num_layers * gpu_layer_compute_ms) + lookup_latency_ms
    time_saved_ms = unoverlapped_time_ms - total_step_time_ms

    return {
        "gpu_layer_compute_ms": round(gpu_layer_compute_ms, 3),
        "lookup_latency_ms": round(lookup_latency_ms, 3),
        "engram_layer_idx": engram_layer_idx,
        "available_buffer_window_ms": round(available_window_ms, 3),
        "hidden_latency_ms": round(hidden_latency_ms, 3),
        "exposed_latency_ms": round(exposed_latency_ms, 3),
        "hidden_pct": round(hidden_pct, 1),
        "total_step_time_ms": round(total_step_time_ms, 3),
        "time_saved_ms": round(time_saved_ms, 3),
    }


def benchmark_real_threads(num_iters: int = 100, embed_dim: int = 512, head_dim: int = 64, num_heads: int = 16):
    """Measures real Python/C++ threading synchronization and tensor handoff overhead."""
    import concurrent.futures

    # Synthetic lookup function
    table = torch.randn(100_000, head_dim, dtype=torch.float16)
    indices = torch.randint(0, 100_000, (num_heads,), dtype=torch.long)

    def async_gather():
        t0 = time.perf_counter()
        res = table[indices]
        return res, (time.perf_counter() - t0) * 1000

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    sync_overheads = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        # Dispatch async lookup
        future = executor.submit(async_gather)
        # Simulate GPU Layer 0 work
        time.sleep(0.001)  # 1.0 ms
        # Synchronize
        res, lookup_ms = future.result()
        t_total = (time.perf_counter() - t0) * 1000
        sync_overheads.append(t_total)

    executor.shutdown()
    return float(np.mean(sync_overheads)), float(np.median(sync_overheads))


def main():
    parser = argparse.ArgumentParser(description="Prefetch & Overlap Benchmark Simulation")
    parser.add_argument("--out-dir", default="engram-npu-prep/results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'='*70}")
    print("⚡ Pipeline Overlap & Prefetch Simulation for Engram / Qwen3.8-Flash-Next")
    print(f"{'='*70}")

    # Test cases:
    # 1. Strix Halo iGPU (74 tok/s decode = 13.5 ms per token total across 30 layers = 0.45 ms/layer)
    # 2. 4x RTX 3090 cluster (120 tok/s decode = 8.3 ms per token across 61 layers = 0.136 ms/layer)
    # 3. Lookup latencies: 0.05 ms (Fast RAM gather), 0.50 ms (PCIe/AXI DMA), 2.0 ms (PCIe burst)
    
    scenarios = [
        {"platform": "Strix Halo iGPU (74 tok/s, 30L)", "gpu_layer_ms": 0.45, "layers": 30, "layer_idx": 1, "lookup_ms": 0.05, "desc": "Host RAM CPU Gather (1.4 µs)"},
        {"platform": "Strix Halo iGPU (74 tok/s, 30L)", "gpu_layer_ms": 0.45, "layers": 30, "layer_idx": 1, "lookup_ms": 0.35, "desc": "NPU AXI DMA Gather (350 µs)"},
        {"platform": "Strix Halo iGPU (74 tok/s, 30L)", "gpu_layer_ms": 0.45, "layers": 30, "layer_idx": 15, "lookup_ms": 0.35, "desc": "NPU AXI Gather at Layer 15 (6.75 ms buffer)"},
        {"platform": "4x RTX 3090 Cluster (120 tok/s, 61L)", "gpu_layer_ms": 0.136, "layers": 61, "layer_idx": 1, "lookup_ms": 0.08, "desc": "PCIe Gen4 H2D Gather Transfer (80 µs)"},
        {"platform": "4x RTX 3090 Cluster (120 tok/s, 61L)", "gpu_layer_ms": 0.136, "layers": 61, "layer_idx": 1, "lookup_ms": 0.30, "desc": "High contention PCIe lookup (300 µs)"},
        {"platform": "4x RTX 3090 Cluster (120 tok/s, 61L)", "gpu_layer_ms": 0.136, "layers": 61, "layer_idx": 3, "lookup_ms": 0.30, "desc": "Engram at Layer 3 with PCIe (408 µs buffer)"},
    ]

    results = []
    for s in scenarios:
        res = simulate_overlap_step(
            gpu_layer_compute_ms=s["gpu_layer_ms"],
            lookup_latency_ms=s["lookup_ms"],
            engram_layer_idx=s["layer_idx"],
            num_layers=s["layers"],
        )
        res["platform"] = s["platform"]
        res["description"] = s["desc"]
        results.append(res)
        print(f"\n[*] {s['platform']} | {s['desc']}:")
        print(f" • GPU Layer Time : {res['gpu_layer_compute_ms']} ms | Buffer Window: {res['available_buffer_window_ms']} ms")
        print(f" • Lookup Latency : {res['lookup_latency_ms']} ms | Hidden: {res['hidden_pct']}% ({res['hidden_latency_ms']} ms)")
        print(f" • Exposed Latency: {res['exposed_latency_ms']} ms | Total Step: {res['total_step_time_ms']} ms (Saved: {res['time_saved_ms']} ms)")

    mean_thread_ms, med_thread_ms = benchmark_real_threads()
    print(f"\n[*] Measured Thread Synchronization & Dispatch Overhead: {mean_thread_ms:.3f} ms")

    out_data = {
        "benchmark": "prefetch_and_pipeline_overlap",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thread_dispatch_overhead_ms": round(mean_thread_ms, 3),
        "scenarios": results,
    }

    out_file = os.path.join(args.out_dir, "prefetch_overlap_results.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[+] Saved results to {out_file}")


if __name__ == "__main__":
    main()
