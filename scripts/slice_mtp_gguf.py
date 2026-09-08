#!/usr/bin/env python3
"""
slice_mtp_gguf.py — Binary slice of MTP head-only GGUF from Qwen3.8-27B GGUF.
Preserves raw quant bytes verbatim (no re-encoding). Uses gguf-py GGUFReader for
authoritative parsing, then rewrites header/KV/info/data for the kept tensors.
"""

import struct
import os
import sys

SRC_GGUF = os.environ.get("QWEN38_GGUF", "models/Qwen3.8-27B-ROCmFP4-FAST.gguf")
DST_GGUF = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-MTP-Head.gguf")

def main():
    sys.path.insert(0, os.environ.get("GGUF_PY_DIR", "gguf-py"))
    from gguf import GGUFReader

    print(f"=== Slicing MTP head from {os.path.basename(SRC_GGUF)} ===")
    r = GGUFReader(SRC_GGUF)

    # select keep tensors (file order)
    keep = [t for t in r.tensors if t.name.startswith("blk.64.") or
            t.name in ("token_embd.weight", "output.weight", "output_norm.weight")]
    print(f" • keeping {len(keep)} tensors")
    for t in keep:
        print(f"    {t.name:<35} type={t.tensor_type} bytes={t.n_bytes:,}")

    alignment = r.alignment
    kv_count = len([f for f in r.fields if not f.startswith("GGUF.")])
    tensor_info_start = r.tensors[0].field.offset
    version = 3

    with open(SRC_GGUF, "rb") as fsrc:
        # verify magic
        magic = fsrc.read(4)
        assert magic == b"GGUF"
        # kv bytes region: [24, tensor_info_start)
        fsrc.seek(24)
        kv_bytes = fsrc.read(tensor_info_start - 24)

        # read raw data for each kept tensor (absolute offsets)
        data = {}
        for t in keep:
            fsrc.seek(t.data_offset)
            data[t.name] = fsrc.read(t.n_bytes)

    # write new GGUF
    with open(DST_GGUF, "wb") as out:
        out.write(magic)
        out.write(struct.pack("<I", version))
        out.write(struct.pack("<Q", len(keep)))
        out.write(struct.pack("<Q", kv_count))
        out.write(kv_bytes)

        # compute offsets relative to data region start
        cur = 0
        for t in keep:
            name_b = t.name.encode("utf-8")
            out.write(struct.pack("<Q", len(name_b)))
            out.write(name_b)
            # reconstruct file-order dims from field.parts[3] (raw uint64 array)
            raw_dims = t.field.parts[3]
            out.write(struct.pack("<I", len(raw_dims)))
            out.write(raw_dims.tobytes())
            out.write(struct.pack("<I", int(t.tensor_type)))
            out.write(struct.pack("<Q", cur))
            cur += t.n_bytes
            cur += (alignment - (cur % alignment)) % alignment

        # align data region start to `alignment` (reader requirement)
        pad_start = (alignment - (out.tell() % alignment)) % alignment
        if pad_start:
            out.write(b"\x00" * pad_start)

        # write data
        for t in keep:
            out.write(data[t.name])
            pad = (alignment - (t.n_bytes % alignment)) % alignment
            if pad:
                out.write(b"\x00" * pad)

    sz = os.path.getsize(DST_GGUF)
    print(f" • wrote {DST_GGUF}: {sz:,} bytes ({sz/1024/1024:.1f} MB)")

    # verify
    r2 = GGUFReader(DST_GGUF)
    ok = True
    for t in r2.tensors:
        if t.data_offset + t.n_bytes > sz:
            print(f"  OVERFLOW {t.name}")
            ok = False
    print(f" • verification: {'OK' if ok else 'FAILED'} ({len(r2.tensors)} tensors read back)")

if __name__ == "__main__":
    main()
