#!/usr/bin/env python3
"""
Inject q_norm and k_norm unit tensors into model.q4nx for MiniCPM5-2B.
libqwen3_npu.so expects model.layers.{i}.self_attn.q_norm.weight and k_norm.weight (shape [128], BF16).

NOTE (correctness): all-ones scale tensors satisfy the weight loader but are NOT
a mathematical identity. RMSNorm(x, gamma=1) still divides by RMS(x), so if the
Qwen3 engine applies QK RMSNorm, attention scores differ from the original
MiniCPM5 architecture (which has no QK normalization). Output equivalence of
this port is therefore unvalidated; treat any generation as unverified.
"""

import sys
import torch
from safetensors.torch import load_file, save_file

def main():
    q4nx_path = "/home/user/.config/flm/models/MiniCPM5-2B-NPU2/model.q4nx"
    if len(sys.argv) > 1:
        q4nx_path = sys.argv[1]
        
    print(f"Loading {q4nx_path}...")
    tensors = load_file(q4nx_path)
    
    num_layers = 42
    print(f"Injecting q_norm and k_norm for {num_layers} layers...")
    for i in range(num_layers):
        q_norm_key = f"model.layers.{i}.self_attn.q_norm.weight"
        k_norm_key = f"model.layers.{i}.self_attn.k_norm.weight"
        tensors[q_norm_key] = torch.ones(128, dtype=torch.bfloat16)
        tensors[k_norm_key] = torch.ones(128, dtype=torch.bfloat16)
        
    print(f"Total tensors after injection: {len(tensors)}")
    print(f"Saving to {q4nx_path}...")
    save_file(tensors, q4nx_path)
    print("Injection complete!")

if __name__ == "__main__":
    main()
