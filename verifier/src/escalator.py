#!/usr/bin/env python3
"""
escalator.py — Qwen3.5-2B logprob escalation tier (iGPU llama-server, :8013).

The NPU LFM endpoint only returns text verdicts (and leans SUSPECT, never a
clean ABORT). For SUSPECT steps we escalate to the iGPU judge, which returns
CONTINUE / ABORT. We read the first-token logprobs (top_logprobs=20) and
aggregate probability mass over the "CONT"/"AB" token prefixes to get a
continuous confidence. Low confidence -> treated as ABORT.

Returns: r in [0,1] = P(continue); jeopardy when r is low (<= threshold or
bottom quartile of recent scores).
"""

import json
import math
import urllib.request
from typing import Any, Dict, List

JUDGE_PROMPT = """You are a strict QA reviewer watching a software agent work.

TASK:
{task}

RECENT TOOL OUTPUTS:
{trajectory}

Based ONLY on this evidence, should the agent CONTINUE its current approach,
or is it on a broken path that should be ABORTed (and retried)?
Respond with a single word: CONTINUE or ABORT."""


class QwenEscalator:
    def __init__(self, endpoint: str = "http://127.0.0.1:8013/v1",
                 threshold: float = 0.50):
        self.chat_url = endpoint.rstrip("/")
        if not self.chat_url.endswith("/chat/completions"):
            self.chat_url += "/chat/completions"
        self.threshold = threshold
        self.scores: List[float] = []

    @staticmethod
    def _prefix_mass(dist: Dict[str, float], prefixes: List[str]) -> float:
        total = 0.0
        for tok, p in dist.items():
            t = tok.strip().upper()
            if any(t.startswith(p) for p in prefixes):
                total += p
        return total

    def score_step(self, task: Dict[str, Any], steps: List[Any]) -> Dict[str, Any]:
        parts = []
        for s in steps[-3:]:
            tc = s.tool_call if hasattr(s, "tool_call") else s.get("tool_call")
            tr = s.tool_result if hasattr(s, "tool_result") else s.get("tool_result")
            name = (tc or {}).get("name") if isinstance(tc, dict) else None
            out = (tr or {}).get("stderr") or (tr or {}).get("stdout") or ""
            parts.append(f"[{name}] exit={(tr or {}).get('exit_code')}\n{out[:400]}")
        traj = "\n".join(parts)

        payload = {
            "messages": [
                {"role": "system", "content": "You are a strict QA judge. /no_think"},
                {"role": "user", "content": JUDGE_PROMPT.format(
                    task=task["instruction"], trajectory=traj)},
            ],
            "max_tokens": 2,
            "temperature": 1.0,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.chat_url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        ch = (body.get("choices") or [{}])[0]
        content = (ch.get("message") or {}).get("content") or ""
        lp = ((ch.get("logprobs") or {}).get("content") or [None])[0]
        dist: Dict[str, float] = {}
        if lp:
            for t in lp.get("top_logprobs", []):
                dist[t.get("token", "")] = math.exp(t["logprob"])

        p_cont = self._prefix_mass(dist, ["CONT", "PROCEED", "GO"])
        p_abort = self._prefix_mass(dist, ["AB", "STOP", "HALT"])
        total = p_cont + p_abort
        r = p_cont / total if total > 1e-9 else 0.5
        # hard override: if the model literally said ABORT, treat as abort
        said_abort = content.strip().upper().startswith("AB")
        if said_abort:
            r = 0.0
        self.scores.append(r)
        return {
            "r": r,
            "p_continue": p_cont,
            "p_abort": p_abort,
            "content": content.strip(),
        }

    def is_jeopardy(self, score_record: Dict[str, Any]) -> bool:
        r = score_record["r"]
        if r <= self.threshold:
            return True
        if len(self.scores) >= 4:
            q1 = sorted(self.scores)[len(self.scores) // 4]
            if r < q1:
                return True
        return False
