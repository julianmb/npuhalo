#!/usr/bin/env python3
"""
npuhalo_proxy.py — Always-On NPU Guardrail & Smart Reverse Proxy for AMD Strix Halo
Exposes an OpenAI-compatible /v1/chat/completions server on port 8000.

Features:
  1. Zero-Latency Streaming: Passes GPU tokens (Radeon 8060S :8012) directly to client.
  2. Asynchronous NPU Watchdog: XDNA 2 NPU (:8001) audits tool calls & commands at ~2-4W out-of-loop.
  3. Security Interception: Detects destructive commands, secret leaks, and infinite loops.
  4. Optional Pre-Route Gating: Routes trivial queries directly to the NPU to save GPU power.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import argparse
from typing import Optional

import aiohttp
from aiohttp import web

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "verifier", "src"))
from watchdog_analyzer import WatchdogAnalyzer
from npu_router import HybridNPURouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("npuhalo_proxy")

DEFAULT_PROXY_PORT = 8000
DEFAULT_GPU_URL = "http://127.0.0.1:8012"
DEFAULT_NPU_URL = "http://127.0.0.1:8001"
DEFAULT_GPU_MODEL = "Ornith-1.5-35B-A3B-ROCmFP4.gguf"
DEFAULT_NPU_MODEL = "minicpm5:2b"

class NPUHaloProxy:
    def __init__(
        self,
        gpu_url: str = DEFAULT_GPU_URL,
        npu_url: str = DEFAULT_NPU_URL,
        gpu_model: str = DEFAULT_GPU_MODEL,
        npu_model: str = DEFAULT_NPU_MODEL,
        guard_mode: str = "audit",  # "audit" (alert/telemetry only) | "block" (reject destructive steps)
        enable_router: bool = False,
    ):
        self.gpu_url = gpu_url.rstrip("/")
        self.npu_url = npu_url.rstrip("/")
        self.gpu_model = gpu_model
        self.npu_model = npu_model
        self.guard_mode = guard_mode
        self.enable_router = enable_router
        
        self.watchdog = WatchdogAnalyzer(
            npu_url=f"{self.npu_url}/v1/chat/completions",
            npu_model=self.npu_model,
            enable_npu_eval=True,
        )
        self.router = HybridNPURouter(url=self.npu_url, model=self.npu_model) if self.enable_router else None
        self.session: Optional[aiohttp.ClientSession] = None
        self._interceptions_count = 0

    async def start(self):
        connector = aiohttp.TCPConnector(limit=100, force_close=False)
        self.session = aiohttp.ClientSession(connector=connector)
        if self.router:
            await self.router.start()
        logger.info(f"NPUHalo Proxy initialized | GPU={self.gpu_url} | NPU={self.npu_url} | GuardMode={self.guard_mode}")

    async def stop(self):
        if self.session:
            await self.session.close()
        if self.router:
            await self.router.close()

    async def handle_models(self, request: web.Request) -> web.Response:
        """OpenAI-compatible /v1/models endpoint."""
        models = [
            {"id": self.gpu_model, "object": "model", "owned_by": "amd-rdna3.5-igpu"},
            {"id": f"{self.npu_model}-watchdog", "object": "model", "owned_by": "amd-xdna2-npu"},
            {"id": "npuhalo-hybrid", "object": "model", "owned_by": "npuhalo-coprocessor"},
        ]
        return web.json_response({"object": "list", "data": models})

    async def handle_watchdog_status(self, request: web.Request) -> web.Response:
        """Watchdog telemetry and health status endpoint."""
        stats = self.watchdog.get_stats()
        stats["interceptions_blocked"] = self._interceptions_count
        stats["guard_mode"] = self.guard_mode
        stats["npu_target"] = f"{self.npu_url} ({self.npu_model})"
        stats["gpu_target"] = f"{self.gpu_url} ({self.gpu_model})"
        return web.json_response(stats)

    async def handle_watchdog_clear(self, request: web.Request) -> web.Response:
        """Reset telemetry and state history."""
        self.watchdog._history.clear()
        self.watchdog._verdict_history.clear()
        self._interceptions_count = 0
        return web.json_response({"status": "cleared"})

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Proxy /v1/chat/completions with zero streaming delay and async NPU audit."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        stream = body.get("stream", False)
        messages = body.get("messages", [])
        session_id = request.headers.get("X-Session-ID", str(uuid.uuid4())[:8])
        user_prompt = messages[-1].get("content", "") if messages else ""

        # Pre-Route Check: If enabled and query is simple, answer on NPU directly
        if self.enable_router and self.router:
            dec = await self.router.classify(user_prompt)
            if dec.get("route") == "npu":
                logger.info(f"Router: Fast-laning trivial query to NPU @ 2W (reason: {dec.get('reason')}): '{user_prompt[:50]}'")
                return await self._forward_npu_direct(request, body)

        # Standard GPU Path: Forward request to primary GPU server (:8012)
        gpu_endpoint = f"{self.gpu_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        if stream:
            return await self._handle_streaming_completion(request, gpu_endpoint, body, headers, session_id, user_prompt)
        else:
            return await self._handle_non_streaming_completion(request, gpu_endpoint, body, headers, session_id, user_prompt)

    def _is_trivial_query(self, text: str) -> bool:
        """Fast heuristic for routing simple queries to NPU."""
        text_lower = text.lower().strip()
        if len(text.split()) <= 8 and not any(kw in text_lower for kw in ["code", "def ", "class ", "bash", "run ", "test"]):
            if any(text_lower.startswith(q) for q in ["what is", "who is", "who wrote", "calculate", "hello", "hi", "help"]):
                return True
        return False

    async def _forward_npu_direct(self, request: web.Request, body: dict) -> web.Response:
        """Direct dispatch to NPU (saving GPU power)."""
        payload = {**body, "model": self.npu_model}
        async with self.session.post(f"{self.npu_url}/v1/chat/completions", json=payload) as resp:
            data = await resp.json()
            return web.json_response(data, status=resp.status, headers={"X-NPUHalo-Route": "npu-direct"})

    async def _handle_streaming_completion(
        self,
        request: web.Request,
        target_url: str,
        body: dict,
        headers: dict,
        session_id: str,
        user_prompt: str
    ) -> web.StreamResponse:
        """Stream chunks directly to client while asynchronously buffering for NPU audit."""
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)

        accumulated_chunks = []
        t0 = time.perf_counter()

        try:
            async with self.session.post(target_url, json=body, headers=headers) as upstream_resp:
                if upstream_resp.status != 200:
                    err_txt = await upstream_resp.text()
                    await response.write(f"data: {json.dumps({'error': err_txt})}\n\n".encode("utf-8"))
                    await response.write(b"data: [DONE]\n\n")
                    return response

                # Pass-through streaming to client with ZERO delay
                async for chunk in upstream_resp.content.iter_any():
                    if chunk:
                        await response.write(chunk)
                        accumulated_chunks.append(chunk)

            # Reconstruct full response text in background
            full_raw = b"".join(accumulated_chunks).decode("utf-8", errors="replace")
            generated_text = self._extract_text_from_sse(full_raw)
            stream_ms = (time.perf_counter() - t0) * 1000

            # Schedule Out-of-Loop NPU Audit Task (0ms delay to client stream)
            asyncio.create_task(self._async_audit_background(session_id, user_prompt, generated_text, stream_ms))

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            await response.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode("utf-8"))

        await response.write_eof()
        return response

    async def _handle_non_streaming_completion(
        self,
        request: web.Request,
        target_url: str,
        body: dict,
        headers: dict,
        session_id: str,
        user_prompt: str
    ) -> web.Response:
        """Handle non-streaming completion with guardrail interception."""
        async with self.session.post(target_url, json=body, headers=headers) as upstream_resp:
            data = await upstream_resp.json()
            if upstream_resp.status != 200:
                return web.json_response(data, status=upstream_resp.status)

            generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Run Tier 1 + Tier 2 instant checks
            verdict = await self.watchdog.audit_step(session_id, user_prompt, generated_text, session=self.session)
            
            resp_headers = {
                "X-NPU-Watchdog-Risk": verdict.risk_level,
                "X-NPU-Watchdog-Category": verdict.category,
            }

            if verdict.is_anomaly and self.guard_mode == "block" and verdict.risk_level in ("HIGH", "CRITICAL"):
                self._interceptions_count += 1
                logger.warning(f"INTERCEPTED dangerous agent action: {verdict.reason}")
                data["choices"][0]["message"]["content"] = (
                    f"[NPUHALO WATCHDOG INTERCEPTION]\n"
                    f"Action blocked by NPU safety guardrail: {verdict.reason}.\n"
                    f"Please reformulate a safe command."
                )
                resp_headers["X-NPU-Watchdog-Action"] = "BLOCKED"

            return web.json_response(data, headers=resp_headers)

    def _extract_text_from_sse(self, sse_data: str) -> str:
        """Extract combined text content from SSE chunks."""
        text_pieces = []
        for line in sse_data.splitlines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    payload = json.loads(line[6:])
                    delta = payload.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        text_pieces.append(content)
                except Exception:
                    continue
        return "".join(text_pieces)

    async def _async_audit_background(self, session_id: str, prompt: str, text: str, stream_ms: float):
        """Asynchronously executes on NPU without delaying the client."""
        t0 = time.perf_counter()
        verdict = await self.watchdog.audit_step(session_id, prompt, text, session=self.session)
        audit_ms = (time.perf_counter() - t0) * 1000

        if verdict.is_anomaly:
            logger.warning(
                f"[WATCHDOG ALERT] Risk={verdict.risk_level} | Cat={verdict.category} | "
                f"Reason: {verdict.reason} | Audit Latency: {audit_ms:.1f}ms"
            )
        else:
            logger.debug(f"[WATCHDOG OK] Step verified safe in {audit_ms:.1f}ms")

def create_app(proxy: NPUHaloProxy) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/models", proxy.handle_models)
    app.router.add_post("/v1/chat/completions", proxy.handle_chat_completions)
    app.router.add_get("/v1/watchdog/status", proxy.handle_watchdog_status)
    app.router.add_post("/v1/watchdog/clear", proxy.handle_watchdog_clear)
    return app

def main():
    parser = argparse.ArgumentParser(description="NPUHalo Smart Guardrail Reverse Proxy")
    parser.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--gpu-url", default=DEFAULT_GPU_URL, help="Primary GPU llama-server URL")
    parser.add_argument("--npu-url", default=DEFAULT_NPU_URL, help="NPU FastFlowLM URL")
    parser.add_argument("--guard-mode", choices=["audit", "block"], default="audit", help="Guard mode: audit or block")
    parser.add_argument("--enable-router", action="store_true", help="Enable pre-route dispatching of trivial queries to NPU")
    args = parser.parse_args()

    proxy = NPUHaloProxy(
        gpu_url=args.gpu_url,
        npu_url=args.npu_url,
        guard_mode=args.guard_mode,
        enable_router=args.enable_router,
    )

    app = create_app(proxy)
    
    async def on_startup(app):
        await proxy.start()

    async def on_cleanup(app):
        await proxy.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    print("\n========================================================")
    print("  NPUHALO SMART GUARDRAIL PROXY FOR STRIX HALO")
    print(f"  Listening on : http://{args.host}:{args.port}/v1")
    print(f"  GPU Server   : {args.gpu_url}")
    print(f"  NPU Coproc   : {args.npu_url} (MiniCPM5-2B @ ~2-4W)")
    print(f"  Guard Mode   : {args.guard_mode.upper()}")
    print(f"  Router Gating: {'ENABLED' if args.enable_router else 'DISABLED'}")
    print("========================================================\n")

    web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
