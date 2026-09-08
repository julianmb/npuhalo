#!/usr/bin/env python3
"""
check_judge_logprobs.py — Startup self-test for the Tier-3 logprob judge.

Background: upstream llama.cpp returns SILENTLY WRONG logprobs (0.0 for every
token, empty top_logprobs) whenever ANY speculative decoding (--spec-type) is
enabled on the server (ggml-org/llama.cpp#27972, fix unmerged). The Tier-3
escalator reads first-token top_logprobs, so a judge launched with spec
decoding would produce meaningless CONTINUE/ABORT verdicts.

This script sends an escalator-shaped request and FAILS loudly unless the
endpoint returns genuine, non-degenerate logprobs.

Usage:
  python3 verifier/scripts/check_judge_logprobs.py [--endpoint URL]
  # default endpoint: http://127.0.0.1:8013

Exit codes: 0 = judge healthy, 1 = unhealthy (message explains why).
"""

import argparse
import json
import math
import sys
import urllib.request
import urllib.error

DEFAULT_ENDPOINT = "http://127.0.0.1:8013"

# Fixed probe: a question with an unambiguous single-token continuation,
# so a healthy judge MUST return a peaked (non-uniform) distribution.
PROBE_MESSAGES = [
    {"role": "system", "content": "You are a strict QA judge. /no_think"},
    {"role": "user", "content": "Reply with exactly one word: CONTINUE"},
]


def fail(msg: str) -> int:
    print(f"JUDGE CHECK FAILED: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help="Judge base URL, e.g. http://127.0.0.1:8013")
    args = ap.parse_args()
    base = args.endpoint.rstrip("/")

    # 1. Health check
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
            if r.status != 200:
                return fail(f"/health returned HTTP {r.status}")
    except Exception as e:
        return fail(f"/health unreachable at {base}: {e}")

    # 2. Escalator-shaped logprobs request (mirrors verifier/src/escalator.py)
    payload = {
        "messages": PROBE_MESSAGES,
        "max_tokens": 4,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    try:
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return fail(f"/v1/chat/completions HTTP {e.code}: {e.read()[:200]!r}. "
                    f"Does this server build support logprobs?")
    except Exception as e:
        return fail(f"request failed: {e}")

    # 3. Structural assertions
    ch = (body.get("choices") or [{}])[0]
    lp_list = ((ch.get("logprobs") or {}).get("content") or [])
    if not lp_list:
        return fail("choices[0].logprobs.content is empty — server returned no logprobs. "
                    "If speculative decoding is enabled, disable it (--spec-type none).")
    first = lp_list[0] or {}
    top = first.get("top_logprobs") or []
    if not top:
        return fail("first token has empty top_logprobs — logprobs are present but hollow. "
                    "Suspect speculative-decoding placeholder values; relaunch judge with --spec-type none.")

    # 4. Degeneracy assertions (the spec-decoding bug signature)
    lps = []
    for t in top:
        try:
            lps.append(float(t["logprob"]))
        except (KeyError, TypeError, ValueError):
            return fail(f"top_logprobs entry missing numeric logprob: {t!r}")
    if all(lp == 0.0 for lp in lps):
        return fail("all top_logprobs are exactly 0.0 — this is the known "
                    "speculative-decoding placeholder signature. "
                    "Relaunch judge with --spec-type none.")
    if max(lps) - min(lps) < 1e-6:
        return fail(f"top_logprobs have zero spread ({lps[:3]}...) — distribution is degenerate.")
    # A peaked distribution is expected on this probe; top-1 should dominate
    probs = [math.exp(lp) for lp in lps]
    if max(probs) / (sum(probs) or 1.0) < 0.05:
        return fail("top-1 probability mass implausibly diffuse for a peaked probe.")

    top_tok = top[0].get("token", "?")
    print(f"JUDGE CHECK PASSED: {len(top)} top_logprobs, "
          f"top-1 {top_tok!r} logprob={lps[0]:.3f}, spread={max(lps)-min(lps):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
