#!/usr/bin/env python3
"""
eval_compressor_fidelity.py — Held-out fidelity evaluation of the compressor
sidecar over REAL recorded tool outputs.

Corpus: stratified sample of actual tool_result payloads from recorded agent
runs (shell output, test/grade output, source reads). All sizes are labeled;
the pre-registered production threshold is >=32K chars, which the current
harness cannot produce (shell cap 8K) — sizes here are sub-threshold and
results are reported per-size-bucket accordingly.

Scoring per sample (deterministic, no subjective judgment):
  - required-fact retention (paths / nonzero exit codes / traceback heads /
    failing test names extracted from the ORIGINAL, checked verbatim in the
    compressed text)
  - catastrophic omission flag (critical class lost entirely)
Autopsies saved per sample.
"""

import json
import glob
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from compressor_sidecar import CompressorSidecar, classify_output  # noqa: E402

RUN_DIRS = [
    os.path.join(REPO, "verifier", "results", "rebaseline_v2_20260822", "runs"),
    os.path.join(REPO, "verifier", "results", "active_recovery_calibrated_20260821_1637", "runs"),
    os.path.join(REPO, "verifier", "results", "active_recovery_20260821_1040", "runs"),
]
MIN_CHARS = 500          # prior prototype floor
TARGET_SAMPLES = 36
SEED = 11


def content_kind(text):
    if re.search(r"Traceback \(most recent call last\)|AssertionError|FAILED|=+ test", text):
        return "test_or_traceback"
    if re.search(r"(Collecting |Downloading |Installing |pip )", text):
        return "package_install_log"
    if re.search(r"def |class |import ", text) and len(text) < 4000:
        return "source_inspection"
    if re.search(r"^d[-rwx]{9}|^-rwx|^total ", text, re.M):
        return "file_listing"
    return "generic_shell_output"


import re  # noqa: E402


def build_corpus():
    pool = defaultdict(list)
    for d in RUN_DIRS:
        for f in glob.glob(d + "/*.json"):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            tid, cfg, seed = r.get("task_id"), r.get("config"), r.get("seed")
            for s in (r.get("steps") or []):
                tr = s.get("tool_result") or {}
                out = ((tr.get("stdout") or "") + "\n" + (tr.get("stderr") or "")).strip()
                if len(out) < MIN_CHARS:
                    continue
                kind = content_kind(out)
                pool[kind].append({"task": tid, "config": cfg, "seed": seed,
                                   "tool": tr.get("name") or s.get("tool_call", {}).get("name"),
                                   "text": out})
    rng = random.Random(SEED)
    corpus = []
    kinds = sorted(pool)
    # round-robin strata until TARGET_SAMPLES
    i = 0
    while len(corpus) < TARGET_SAMPLES and any(pool[k] for k in kinds):
        k = kinds[i % len(kinds)]
        if pool[k]:
            corpus.append((k, pool[k].pop(rng.randrange(len(pool[k])))))
        i += 1
    return corpus


def main():
    corpus = build_corpus()
    print(f"corpus: {len(corpus)} samples across strata: "
          f"{ {k: sum(1 for kk,_ in corpus if kk==k) for k in set(k for k,_ in corpus)} }")
    sc = CompressorSidecar(threshold_chars=MIN_CHARS)  # eval mode: compress everything
    results = []
    for idx, (kind, sample) in enumerate(corpus):
        text = sample["text"]
        cls = classify_output(text)
        res = sc.process(sample["tool"] or "shell", text)
        entry = {
            "idx": idx, "stratum": kind, "task": sample["task"], "seed": sample["seed"],
            "chars": len(text), "eligible_class": cls["eligible"], "class_reason": cls["reason"],
            "source": res.source, "fallback_reason": res.fallback_reason,
            "compression_latency_ms": res.compression_latency_ms,
            "compressed_chars": res.compressed_chars,
            "ratio_pct": round(100 * res.compressed_chars / max(1, len(text)), 1),
            "facts_required": {k: v["required"] for k, v in res.retained_facts.items()},
            "facts_retained": {k: v["retained"] for k, v in res.retained_facts.items()},
            "missing_facts": {k: v["missing"] for k, v in res.retained_facts.items() if v["missing"]},
            "catastrophic_omission": any(
                v["required"] and not v["retained"]
                for v in res.retained_facts.values()),
            "inserted_head": res.inserted_text[:120],
        }
        results.append(entry)
        print(f"[{kind:>20} {entry['chars']:>5} ch] src={res.source:<10} "
              f"ratio={entry['ratio_pct']:>5}% lat={entry['compression_latency_ms']:>6}ms "
              f"cat_omit={entry['catastrophic_omission']}", flush=True)

    n = len(results)
    compressed = [r for r in results if r["source"] == "compressed"]
    cats = [r for r in results if r["catastrophic_omission"]]
    summary = {
        "samples": n,
        "strata_counts": {k: sum(1 for kk, _ in corpus if kk == k)
                          for k in set(k for k, _ in corpus)},
        "compressed_ok": len(compressed),
        "fallbacks": [ {"idx": r["idx"], "task": r["task"],
                        "reason": r["fallback_reason"]} for r in results if r["source"] == "original"],
        "catastrophic_omissions": len(cats),
        "mean_ratio_pct_when_compressed": round(
            sum(r["ratio_pct"] for r in compressed) / max(1, len(compressed)), 1),
        "note": "ALL SAMPLES ARE SUB-THRESHOLD (<16K); production threshold is >=32K "
                "which this harness cannot generate (shell cap 8,000 chars)",
        "results": results,
    }
    out = os.path.join(REPO, "verifier", "results", "compressor_production_20260822",
                       "fidelity_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsamples={n} compressed={len(compressed)} fallbacks={n-len(compressed)} "
          f"catastrophic={len(cats)}")
    print(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
