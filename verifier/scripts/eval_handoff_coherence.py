#!/usr/bin/env python3
"""
eval_handoff_coherence.py — TTFT handoff: latency gain vs output coherence.

Closes README caveat #10 ("TTFT handoff coherence is unscored").

For each open-ended prompt:
  Path A (pure GPU): stream from the iGPU generator; record TTFT and full text.
  Path B (handoff):  stream an NPU burst (~24 tokens) for instant first tokens,
                     then continue with the GPU using the burst as an assistant
                     prefix (the archived run_pipeline.py behavior); record
                     effective TTFT, total time, and stitched text.

Scoring (disclosed limitations): a Qwen3.5-2B judge on :8013 rates both answers
blind (randomized order, anonymized labels) on a 1-10 coherence/quality rubric,
plus a deterministic boundary check (does the stitch land mid-sentence?).
A 2B judge is a weak instrument; results are indicative, not definitive.

Usage: python3 verifier/scripts/eval_handoff_coherence.py [--out ...]
"""

import argparse
import json
import os
import random
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

GEN_URL = os.environ.get("HC_GEN", "http://127.0.0.1:8012/v1/chat/completions")
NPU_URL = os.environ.get("HC_NPU", "http://127.0.0.1:8001/v1/chat/completions")
NPU_MODEL = os.environ.get("HC_NPU_MODEL", "qwen3.5:0.8b")
JUDGE_URL = os.environ.get("HC_JUDGE", "http://127.0.0.1:8013/v1/chat/completions")

BURST_TOKENS = 24
PROMPTS = [
    "Explain how HTTPS keeps data private in transit.",
    "Write a short paragraph about why coastal cities flood more often.",
    "Describe the difference between RAM and disk storage.",
    "Summarize how a refrigerator works.",
    "Explain what a database index does and when it hurts performance.",
    "Describe the water cycle in plain language.",
    "Explain why ice floats on water.",
    "Write a brief explanation of compound interest for a teenager.",
]

RUBRIC = """You are grading AI assistant answers for COHERENCE and QUALITY.
Rate the answer below from 1 to 10 where:
  1-3 = broken, contradictory, or mid-sentence garbage
  4-6 = understandable but awkward, repetitive, or shallow
  7-10 = coherent, correct, well-written
Reply with ONLY the number.

ANSWER:
{answer}

SCORE:"""


def stream_chat(url: str, payload: dict) -> tuple[str, float]:
    """Stream a chat completion; return (full_text, ttft_ms).

    Thinking models emit reasoning via delta.reasoning_content before
    delta.content. TTFT counts the first token of either (what a user
    perceives); the returned text prefers visible content over reasoning.
    """
    body = json.dumps({**payload, "stream": True}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    content, reasoning = [], []
    t0, ttft = time.time(), None
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
            except Exception:
                continue
            piece = delta.get("content") or ""
            think = delta.get("reasoning_content") or ""
            if piece or think:
                if ttft is None:
                    ttft = (time.time() - t0) * 1000
            if piece:
                content.append(piece)
            elif think:
                reasoning.append(think)
    text = "".join(content).strip() or "".join(reasoning).strip()
    return text, ttft


def chat_sync(url: str, payload: dict, timeout: int = 120) -> str:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


LONG_PREFIX = (
    "You are given extensive background notes from a codebase review session. "
    "Read them and answer the final question.\n\n" + (
        "Session note: the team profiled the service under load and found that the "
        "connection pool saturates at 200 concurrent requests, latency spikes correlate "
        "with garbage collection pauses, and the cache hit ratio drops below 60% during "
        "deploy windows because keys are versioned by build hash. "
    ) * 22
)

# (prompt, use_long_prefix)
CASES = [(p, False) for p in PROMPTS] + [
    ("Given all the session notes above, what is the most likely root cause of the latency spikes, and which two fixes would you try first?", True),
    ("Given all the session notes above, draft a short incident-summary paragraph for the cache hit ratio problem.", True),
    ("Given all the session notes above, propose a rollout plan for the connection pool change with a rollback criterion.", True),
    ("Given all the session notes above, explain whether key versioning by build hash is a good idea and what to do instead.", True),
]


def boundary_broken(text_at_stitch: str) -> bool:
    """True if the burst ends mid-word/mid-sentence."""
    tail = text_at_stitch.rstrip()
    return bool(tail) and not re.search(r"[.!?:\n]$", tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(REPO, "verifier", "results", "handoff_coherence.json"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Warm endpoints
    stream_chat(NPU_URL, {"model": NPU_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    stream_chat(GEN_URL, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    chat_sync(JUDGE_URL, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})

    rows = []
    for prompt, use_long in CASES:
        user_content = (LONG_PREFIX + "\n\nQuestion: " + prompt) if use_long else prompt
        # Path A: pure GPU
        ans_gpu, ttft_gpu = stream_chat(GEN_URL, {
            "messages": [{"role": "user", "content": user_content}], "max_tokens": 768, "temperature": 0.3,
        })
        # Path B: NPU burst then GPU continuation
        burst, ttft_npu = stream_chat(NPU_URL, {
            "model": NPU_MODEL, "messages": [{"role": "user", "content": user_content}],
            "max_tokens": BURST_TOKENS, "temperature": 0.2,
        })
        cont, ttft_cont = stream_chat(GEN_URL, {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": burst},
                {"role": "user", "content": "Continue your answer seamlessly from exactly where you left off."},
            ],
            "max_tokens": 768, "temperature": 0.3,
        })
        stitched = burst + cont

        # Blind judge scoring (randomized order)
        order = [("A", ans_gpu), ("B", stitched)]
        rng.shuffle(order)
        scores = {}
        for label, answer in order:
            verdict = chat_sync(JUDGE_URL, {
                "messages": [{"role": "user", "content": RUBRIC.format(answer=answer[:3000])}],
                "max_tokens": 16, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            m = re.search(r"\b(10|[1-9])\b", verdict)
            scores[label] = int(m.group(1)) if m else None

        row = {
            "prompt": prompt[:80],
            "long_prefix": use_long,
            "ttft_gpu_ms": round(ttft_gpu),
            "ttft_handoff_ms": round(ttft_npu),
            "burst_chars": len(burst),
            "boundary_mid_sentence": boundary_broken(burst),
            "judge_score_pure_gpu": scores.get("A"),
            "judge_score_handoff": scores.get("B"),
            "answer_gpu_head": ans_gpu[:150],
            "answer_handoff_head": stitched[:150],
        }
        rows.append(row)
        print(f"[handoff{' long' if use_long else ''}] ttft {row['ttft_gpu_ms']}->{row['ttft_handoff_ms']}ms | "
              f"judge gpu={scores.get('A')} handoff={scores.get('B')} | "
              f"mid-sentence={row['boundary_mid_sentence']} | {prompt[:45]}", flush=True)

    def agg(key, pred):
        vals = [r[key] for r in rows if pred(r) and r[key] is not None]
        return round(sum(vals) / len(vals)) if vals else None

    valid = [r for r in rows if r["judge_score_pure_gpu"] and r["judge_score_handoff"]]
    result = {
        "mean_ttft_gpu_ms_short": agg("ttft_gpu_ms", lambda r: not r["long_prefix"]),
        "mean_ttft_handoff_ms_short": agg("ttft_handoff_ms", lambda r: not r["long_prefix"]),
        "mean_ttft_gpu_ms_long": agg("ttft_gpu_ms", lambda r: r["long_prefix"]),
        "mean_ttft_handoff_ms_long": agg("ttft_handoff_ms", lambda r: r["long_prefix"]),
        "boundary_mid_sentence_rate": round(
            sum(r["boundary_mid_sentence"] for r in rows) / len(rows), 2),
        "mean_judge_pure_gpu": round(sum(r["judge_score_pure_gpu"] for r in valid) / len(valid), 1) if valid else None,
        "mean_judge_handoff": round(sum(r["judge_score_handoff"] for r in valid) / len(valid), 1) if valid else None,
        "judge_preference": {
            "gpu": sum(1 for r in valid if r["judge_score_pure_gpu"] > r["judge_score_handoff"]),
            "tie": sum(1 for r in valid if r["judge_score_pure_gpu"] == r["judge_score_handoff"]),
            "handoff": sum(1 for r in valid if r["judge_score_handoff"] > r["judge_score_pure_gpu"]),
        },
        "caveats": [
            "Judge is Qwen3.5-2B — a weak instrument; treat scores as indicative only.",
            f"Burst fixed at {BURST_TOKENS} tokens; stitching prompt asks the GPU to continue seamlessly.",
            "Open-ended prompts only; no factual-correctness grading.",
        ],
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
