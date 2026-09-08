#!/usr/bin/env python3
"""
Phase 1: Async Stream Tap Middleware & Checkpoint Detector.

Intercepts token stream from Ornith-1.5 without blocking client delivery.
Triggers checkpoint verification on structural boundaries or 250-token fallback.
Cancels generation immediately when ABORT verdict is rendered.
"""

import asyncio
import json
import time
import urllib.request
from typing import AsyncGenerator, Dict, Any, Optional, List, Callable
try:
    from .settings import load_settings
except ImportError:  # Support direct imports used by source-tree benchmark scripts.
    from settings import load_settings

_SETTINGS = load_settings()

class StreamTapProxy:
    def __init__(
        self,
        verifier: Any,
        generator_url: str = None,
        token_limit: int = None
    ):
        self.verifier = verifier
        self.generator_url = generator_url or self._endpoint_to_chat_completions(
            _SETTINGS["ORNITH_ENDPOINT"]
        )
        self.token_limit = token_limit or int(_SETTINGS["CHECKPOINT_TOKEN_LIMIT"])

    @staticmethod
    def _endpoint_to_chat_completions(endpoint: str) -> str:
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/v1/chat/completions"):
            return endpoint
        return endpoint + "/chat/completions"

    async def stream_and_verify(
        self,
        task_prompt: str,
        prompt_template: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        on_verdict_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams tokens from generator, yielding to consumer immediately,
        while asynchronously checking checkpoints in the background.
        """
        payload = {
            "messages": [{"role": "user", "content": task_prompt}],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.generator_url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=60))

        accumulated_text = ""
        current_chunk_tokens = 0
        checkpoint_idx = 0
        total_tokens = 0
        abort_signal = False
        abort_details = None

        pending_tasks: List[asyncio.Task] = []

        try:
            while not abort_signal:
                line = await loop.run_in_executor(None, resp.readline)
                if not line:
                    # Stream ended: drain any in-flight verifications so that
                    # late verdicts are observed and reported.
                    if pending_tasks:
                        await asyncio.wait(pending_tasks)
                        for t in pending_tasks:
                            pending_tasks.remove(t)
                            res = t.result()
                            if on_verdict_cb:
                                on_verdict_cb(res)
                            if res["verdict"] == "ABORT":
                                abort_signal = True
                                abort_details = res
                                yield {
                                    "type": "abort",
                                    "checkpoint_index": res["checkpoint_index"],
                                    "tokens_generated": total_tokens,
                                    "trigger": res["trigger"],
                                    "evidence": res.get("evidence", ""),
                                    "latency_ms": res["latency_ms"]
                                }
                    break

                t_token_start = time.time()
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                    continue

                try:
                    chunk = json.loads(line_str[6:])
                    delta = chunk["choices"][0]["delta"]
                    token_text = delta.get("content") or delta.get("reasoning_content") or ""
                except Exception:
                    token_text = ""

                if not token_text:
                    continue

                total_tokens += 1
                current_chunk_tokens += 1
                accumulated_text += token_text

                tap_overhead_ms = (time.time() - t_token_start) * 1000

                # Yield immediately to client (zero wait)
                yield {
                    "type": "token",
                    "token": token_text,
                    "accumulated": accumulated_text,
                    "token_index": total_tokens,
                    "tap_overhead_ms": tap_overhead_ms
                }

                # Check for checkpoint triggers:
                # 1. Code fence close (```)
                # 2. Tool call close (</tool_call>)
                # 3. Blank line boundary (\n\n)
                # 4. Token count fallback >= token_limit
                is_trigger = False
                trigger_reason = ""

                if "```" in token_text and accumulated_text.count("```") % 2 == 0:
                    is_trigger = True
                    trigger_reason = "code_fence_close"
                elif "</tool_call>" in token_text or "</tool>" in token_text:
                    is_trigger = True
                    trigger_reason = "tool_call_close"
                elif token_text == "\n" and accumulated_text.endswith("\n\n"):
                    is_trigger = True
                    trigger_reason = "blank_line_step"
                elif current_chunk_tokens >= self.token_limit:
                    is_trigger = True
                    trigger_reason = "token_limit_fallback"

                if is_trigger:
                    checkpoint_idx += 1
                    chk_text = accumulated_text
                    current_chunk_tokens = 0

                    # Spawn non-blocking background verification
                    task = asyncio.create_task(
                        self._async_verify_step(
                            checkpoint_idx,
                            task_prompt,
                            chk_text,
                            prompt_template,
                            trigger_reason,
                            total_tokens
                        )
                    )
                    pending_tasks.append(task)

                # Check if any completed verifier task returned ABORT
                done_tasks = [t for t in pending_tasks if t.done()]
                for t in done_tasks:
                    pending_tasks.remove(t)
                    res = t.result()
                    if on_verdict_cb:
                        on_verdict_cb(res)

                    if res["verdict"] == "ABORT":
                        abort_signal = True
                        abort_details = res
                        yield {
                            "type": "abort",
                            "checkpoint_index": res["checkpoint_index"],
                            "tokens_generated": total_tokens,
                            "trigger": res["trigger"],
                            "evidence": res.get("evidence", ""),
                            "latency_ms": res["latency_ms"]
                        }
                        break

        finally:
            resp.close()
            # Clean up pending verifications
            for t in pending_tasks:
                if not t.done():
                    t.cancel()

        yield {
            "type": "finish",
            "total_tokens": total_tokens,
            "accumulated_text": accumulated_text,
            "aborted": abort_signal,
            "abort_details": abort_details
        }

    async def _async_verify_step(
        self,
        checkpoint_idx: int,
        task_prompt: str,
        trajectory: str,
        prompt_template: str,
        trigger: str,
        token_count: int
    ) -> Dict[str, Any]:
        """Runs LFM verification in executor thread without blocking event loop."""
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            lambda: self.verifier.evaluate_checkpoint(
                task_prompt=task_prompt,
                trajectory=trajectory,
                prompt_template=prompt_template,
                k_samples=3,
                temperature=0.7
            )
        )
        res["checkpoint_index"] = checkpoint_idx
        res["trigger"] = trigger
        res["token_count"] = token_count
        return res
