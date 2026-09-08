#!/usr/bin/env python3
"""
expand_kv_heads.py
Replicates MiniCPM5-2B's 2 KV heads into 8 KV heads (4x replication per head).
This transforms the GQA ratio from 16:2 (8:1, unsupported by libmha.so)
to 16:8 (2:1, natively supported by libmha.so's _gen_mha_seq_d128_q2 kernel).
"""

import os
import json
import shutil
from safetensors.torch import load_file, save_file

def adapt_model(src_dir: str, dst_dir: str, target_kv_heads: int = 8):
    os.makedirs(dst_dir, exist_ok=True)

    # 1. Update config.json
    config_path = os.path.join(src_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    orig_kv_heads = config.get("num_key_value_heads", 2)
    orig_q_heads = config.get("num_attention_heads", 16)
    head_dim = config.get("head_dim", 128)
    num_layers = config.get("num_hidden_layers", 42)

    print(f"[INFO] Loaded config: H_q={orig_q_heads}, H_kv={orig_kv_heads}, d_head={head_dim}, layers={num_layers}")
    assert target_kv_heads % orig_kv_heads == 0, f"Target {target_kv_heads} must be divisible by original {orig_kv_heads}"
    rep_factor = target_kv_heads // orig_kv_heads
    print(f"[INFO] Expanding KV heads from {orig_kv_heads} to {target_kv_heads} (replication factor {rep_factor}x)")

    config["num_key_value_heads"] = target_kv_heads
    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved modified config.json to {dst_dir}")

    # 2. Copy auxiliary files
    aux_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja"
    ]
    for af in aux_files:
        src_file = os.path.join(src_dir, af)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(dst_dir, af))
            print(f"[INFO] Copied {af}")

    # 3. Load Safetensors
    st_files = [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]
    print(f"[INFO] Found safetensor files: {st_files}")

    for st_name in st_files:
        src_st = os.path.join(src_dir, st_name)
        dst_st = os.path.join(dst_dir, st_name)
        print(f"[INFO] Processing {src_st}...")

        state_dict = load_file(src_st)
        new_state_dict = {}

        for k, v in state_dict.items():
            if "self_attn.k_proj.weight" in k or "self_attn.v_proj.weight" in k:
                hidden_size = v.shape[-1]
                assert v.shape[0] == orig_kv_heads * head_dim, f"Unexpected shape for {k}: {v.shape}"
                v_reshaped = v.view(orig_kv_heads, head_dim, hidden_size)
                v_expanded = v_reshaped.repeat_interleave(rep_factor, dim=0)
                v_out = v_expanded.view(target_kv_heads * head_dim, hidden_size)
                new_state_dict[k] = v_out.contiguous()
            elif "self_attn.k_proj.bias" in k or "self_attn.v_proj.bias" in k:
                assert v.shape[0] == orig_kv_heads * head_dim
                v_reshaped = v.view(orig_kv_heads, head_dim)
                v_expanded = v_reshaped.repeat_interleave(rep_factor, dim=0)
                v_out = v_expanded.view(target_kv_heads * head_dim)
                new_state_dict[k] = v_out.contiguous()
            else:
                new_state_dict[k] = v

        print(f"[INFO] Saving adapted weights to {dst_st}...")
        save_file(new_state_dict, dst_st, metadata={"format": "pt"})
        print(f"[INFO] Successfully saved {dst_st}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/home/user/source/npuhalo/minicpm5-npu/hf_raw")
    parser.add_argument("--dst", default="/home/user/source/npuhalo/minicpm5-npu/hf_adapted")
    parser.add_argument("--target-kv-heads", type=int, default=8)
    args = parser.parse_args()
    adapt_model(args.src, args.dst, args.target_kv_heads)
