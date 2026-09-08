#!/usr/bin/env python3
"""
extract_mtp_gguf.py — Extract standalone MTP head-only GGUF from Qwen 3.8 27B GGUF.
Copies all metadata fields and the 15 blk.64.* nextn tensors + embeddings/output.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.environ.get("GGUF_PY_DIR", "gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGUFValueType

ROOT = Path(__file__).resolve().parent.parent
SRC_GGUF = os.environ.get(
    "QWEN38_GGUF",
    str(ROOT / "models" / "Qwen3.8-27B-ROCmFP4-FAST.gguf"),
)
DST_GGUF = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3.8-27B-MTP-Head.gguf")

def extract_mtp_head():
    print(f"=== Extracting MTP Head from {SRC_GGUF} ===")
    r = GGUFReader(SRC_GGUF)
    
    keep_names = set()
    for t in r.tensors:
        if t.name.startswith("blk.64.") or t.name in ["token_embd.weight", "output.weight", "output_norm.weight"]:
            keep_names.add(t.name)
            
    print(f" • Found {len(keep_names)} tensors to keep (15 blk.64 + globals)")
    
    w = GGUFWriter(DST_GGUF, arch="qwen35", use_temp_file=False)
    
    # Copy metadata fields directly
    for field in r.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        try:
            # Reconstruct field in writer
            val_type = field.types[0]
            if val_type == GGUFValueType.STRING:
                val = str(field.parts[field.data[0]].tobytes().decode("utf-8", errors="replace")) if field.data else ""
                w.add_string(field.name, val)
            elif val_type == GGUFValueType.UINT32:
                w.add_uint32(field.name, int(field.parts[field.data[0]][0]))
            elif val_type == GGUFValueType.INT32:
                w.add_int32(field.name, int(field.parts[field.data[0]][0]))
            elif val_type == GGUFValueType.FLOAT32:
                w.add_float32(field.name, float(field.parts[field.data[0]][0]))
            elif val_type == GGUFValueType.BOOL:
                w.add_bool(field.name, bool(field.parts[field.data[0]][0]))
            elif val_type == GGUFValueType.UINT64:
                w.add_uint64(field.name, int(field.parts[field.data[0]][0]))
            elif val_type == GGUFValueType.INT64:
                w.add_int64(field.name, int(field.parts[field.data[0]][0]))
            elif val_type == GGUFValueType.ARRAY:
                # Copy array
                sub_type = field.types[1]
                if sub_type == GGUFValueType.STRING:
                    strs = [part.tobytes().decode("utf-8", errors="replace") for part in field.parts]
                    w.add_array(field.name, strs)
                elif sub_type in [GGUFValueType.INT32, GGUFValueType.UINT32]:
                    arr = [int(x) for x in field.parts[0]]
                    w.add_array(field.name, arr)
                elif sub_type in [GGUFValueType.FLOAT32]:
                    arr = [float(x) for x in field.parts[0]]
                    w.add_array(field.name, arr)
        except Exception:
            # print(f"Skipped field {field.name}: {e}")
            pass

    # Ensure nextn metadata is set
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_uint32("qwen35.block_count", 65)

    # Add tensors
    for t in r.tensors:
        if t.name in keep_names:
            w.add_tensor(t.name, t.data, raw_shape=t.shape, raw_dtype=t.tensor_type)
            print(f"   + Tensor: {t.name:<35} | Shape: {str(t.shape):<20} | Type: {t.tensor_type} ({t.n_bytes:,} bytes)")

    print(f" • Writing output to {DST_GGUF}...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    
    print(f" • MTP Head GGUF extracted successfully: {os.path.getsize(DST_GGUF):,} bytes (~{os.path.getsize(DST_GGUF)/(1024*1024):.1f} MB)")

if __name__ == "__main__":
    extract_mtp_head()
