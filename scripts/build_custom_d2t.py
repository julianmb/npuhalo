#!/usr/bin/env python3
"""
build_custom_d2t.py — Build a frequency-optimized compressed EAGLE-3 head.

The stock Ex0bit compressed head failed because its 32k draft vocab only covers
7.4% of Qwen 3.8's realized tokens. This script builds our own compressed head:
  1. Tokenize a representative corpus (code + prose) with the Qwen3.5 tokenizer.
  2. Pick the top-32768 token IDs by frequency (+ forced special tokens).
  3. Slice the FULL head's lm_head rows to those tokens -> [32768, 5120].
  4. Emit d2t = selected target IDs (int64).
  5. Write a new safetensors (other tensors copied from the full head) + config.
Output: models/EAGLE3-custom/compressed/ ready for the fork converter.
"""

import json
import glob
import os
import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

FULL_ST = os.environ.get(
    "EAGLE3_FULL_ST",
    os.path.join(ROOT, "models", "Ex0bit-EAGLE3-full", "full", "model.safetensors"),
)
TOK_JSON = os.environ.get(
    "QWEN38_TOKENIZER_JSON",
    os.path.join(ROOT, "models", "Qwen3.8-27B-MTP-4bit", "tokenizer.json"),
)
OUT_DIR = os.path.join(ROOT, "models", "EAGLE3-custom")
DRAFT_VOCAB = 32768

# 1. Corpus: code + prose + docs (large, diverse — Zipf needs scale)
corpus_files = []
BUDGET = 15_000_000
total = 0
for root in os.environ.get("CORPUS_ROOTS", ROOT).split(os.pathsep):
    for ext in ("*.py", "*.md", "*.cpp", "*.h", "*.c", "*.sh", "*.json"):
        for f in glob.glob(f"{root}/**/{ext}", recursive=True):
            try:
                sz = os.path.getsize(f)
            except Exception:
                continue
            if sz > 2_000_000:
                continue
            corpus_files.append(f)
            total += sz
            if total > BUDGET:
                break
        if total > BUDGET:
            break
    if total > BUDGET:
        break
text = ""
for f in corpus_files:
    try:
        text += open(f, encoding="utf-8", errors="replace").read() + "\n"
    except Exception:
        pass
# add generic prose (chat-style Q&A representative of generation)
text += (
    "Explain how speculative decoding accelerates large language model inference. "
    "The difference between a list and a tuple in Python is that lists are mutable. "
    "Unified memory architecture allows the CPU and GPU to share the same memory pool. "
) * 50
print(f"corpus: {len(text):,} chars from {len(corpus_files)} files")

# 2. Token frequencies
tok = Tokenizer.from_file(TOK_JSON)
counts = Counter()
CHUNK = 100000
for i in range(0, len(text), CHUNK):
    for t in tok.encode(text[i:i+CHUNK]).ids:
        counts[t] += 1

# force-include special tokens (BOS/EOS/PAD from config)
specials = [248044, 248046, 248055]
top = [tid for tid, _ in counts.most_common(DRAFT_VOCAB)]
for s in specials:
    if s not in top and len(top) < DRAFT_VOCAB:
        top.append(s)
top = top[:DRAFT_VOCAB]
d2t = np.array(sorted(set(top)), dtype=np.int64)  # sorted: cheap locality
print(f"selected {len(d2t)} draft tokens; coverage of corpus tokens: "
      f"{sum(counts[t] for t in d2t)/max(1,sum(counts.values()))*100:.1f}%")

# 3. Slice the FULL head's lm_head
from safetensors import safe_open
with safe_open(FULL_ST, framework="pt") as f:
    lm = f.get_tensor("lm_head.weight")            # [248320, 5120] bf16
    tensors = {}
    for k in f.keys():
        if k != "lm_head.weight":
            tensors[k] = f.get_tensor(k)
lm_sel = lm[torch.from_numpy(d2t).long(), :]        # [32768, 5120]
tensors["lm_head.weight"] = lm_sel.contiguous()
tensors["d2t"] = torch.from_numpy(d2t)

import os
os.makedirs(OUT_DIR, exist_ok=True)
save_file(tensors, f"{OUT_DIR}/model.safetensors")
print(f"wrote {OUT_DIR}/model.safetensors ({os.path.getsize(OUT_DIR+'/model.safetensors')/1e6:.0f} MB)")

# 4. Copy + patch config (draft_vocab_size already 32768)
cfg = json.load(open(os.path.join(ROOT, "models", "Ex0bit-EAGLE3-compressed", "compressed", "config.json")))
json.dump(cfg, open(f"{OUT_DIR}/config.json", "w"), indent=4)
print("config written")
