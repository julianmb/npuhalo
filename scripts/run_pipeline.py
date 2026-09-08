#!/usr/bin/env python3
"""
run_pipeline.py — Experimental Strix Halo NPU-to-iGPU Handoff Pipeline

Real hybrid architecture:
  1. NPU (qwen3.5-0.8b-FLM @ 42 tok/s, 347ms TTFT, ~2W) streams the FIRST tokens
     of the answer to the client instantly — giving sub-350ms perceived latency.
  2. The pipeline feeds the NPU text as an assistant continuation into the
     iGPU model. This is a handoff experiment: it does not perform speculative
     token acceptance or retract NPU tokens already streamed to the client.

Exposes an OpenAI-compatible endpoint on --port (default: 11435).
"""

import os
import json
import time
import signal
import asyncio
import argparse
import subprocess
from pathlib import Path
from typing import Optional
import aiohttp
from aiohttp import web

DEFAULT_GPU_MODEL = os.environ.get("QWEN38_GGUF", "models/Qwen3.8-27B-ROCmFP4-FAST.gguf")
DEFAULT_NPU_MODEL = "qwen3.5-0.8b-FLM"
DEFAULT_LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "build-strix-rocmfp4/bin/llama-server")
GPU_SERVER_PORT = 8012
NPU_SERVER_URL = "http://127.0.0.1:13305"   # lemonade front
PIPELINE_PORT = 11435
NPU_BURST_TOKENS = 24   # how many tokens the NPU drafts before GPU handoff


class StrixHaloHybridPipeline:
    def __init__(
        self,
        gpu_model_path: str = DEFAULT_GPU_MODEL,
        npu_model_name: str = DEFAULT_NPU_MODEL,
        llama_bin: str = DEFAULT_LLAMA_SERVER_BIN,
        port: int = PIPELINE_PORT,
        gpu_port: int = GPU_SERVER_PORT,
        npu_url: str = NPU_SERVER_URL,
        device: str = "Vulkan0",
        draft_n: int = 4,
        npu_burst_tokens: int = NPU_BURST_TOKENS,
        host: str = "127.0.0.1",
    ):
        self.gpu_model_path = gpu_model_path
        self.npu_model_name = npu_model_name
        self.llama_bin = llama_bin
        self.port = port
        self.gpu_port = gpu_port
        self.npu_url = npu_url
        self.device = device
        self.draft_n = draft_n
        self.npu_burst_tokens = npu_burst_tokens
        self.host = host
        self.gpu_process: Optional[subprocess.Popen] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.npu_available = False
        self.gpu_available = False
        self.app = web.Application(client_max_size=1024 * 1024)
        self.setup_routes()

    def setup_routes(self):
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_get("/v1/models", self.handle_models)
        self.app.router.add_post("/v1/chat/completions", self.handle_chat_completions)

    async def handle_health(self, request: web.Request) -> web.Response:
        status = "ready" if self.gpu_available else "not_ready"
        return web.json_response({
            "status": status,
            "pipeline": "Strix-Halo-NPU-iGPU-Handoff",
            "npu_node": "/dev/accel/accel0",
            "npu_available": self.npu_available,
            "gpu_device": self.device,
            "target_model": Path(self.gpu_model_path).name,
            "draft_model": self.npu_model_name,
            "npu_burst_tokens": self.npu_burst_tokens,
        }, status=200 if self.gpu_available else 503)

    async def handle_models(self, request: web.Request) -> web.Response:
        return web.json_response({
            "object": "list",
            "data": [{
                "id": "qwen3.8-27b-hybrid",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "strix-halo-pipeline",
            }]
        })

    def start_gpu_server(self):
        if not (os.path.exists(self.gpu_model_path) and os.path.exists(self.llama_bin)):
            print("[Pipeline] Error: missing model or llama-server binary")
            return False
        cmd = [
            self.llama_bin, "-m", self.gpu_model_path,
            "--port", str(self.gpu_port), "--host", "127.0.0.1",
            "--device", self.device,
            "--spec-type", "draft-mtp", "--spec-draft-n-max", str(self.draft_n),
            "-ngl", "99", "-fa", "1", "-c", "32768", "-b", "2048", "-ub", "2048",
            "--no-mmap", "--reasoning", "off",
        ]
        env = os.environ.copy()
        env["ROCBLAS_USE_HIPBLASLT"] = "1"
        print(f"[Pipeline] starting iGPU server ({self.device}, K={self.draft_n}) :{self.gpu_port}")
        self.gpu_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE, env=env,
                                            preexec_fn=os.setsid)
        return True

    async def wait_for_gpu_server(self, max_retries=60):
        url = f"http://127.0.0.1:{self.gpu_port}/health"
        for i in range(max_retries):
            try:
                async with self.http_session.get(url) as r:
                    if r.status == 200:
                        print(f"[Pipeline] iGPU ready ({i}s)")
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
        print("[Pipeline] WARN: iGPU server not ready")
        return False

    async def probe_npu(self):
        try:
            async with self.http_session.get(f"{self.npu_url}/api/v1/models") as r:
                if r.status == 200:
                    self.npu_available = True
                    print("[Pipeline] NPU drafter available")
                    return True
        except Exception as e:
            print(f"[Pipeline] NPU not available: {e}")
        self.npu_available = False
        return False

    async def npu_stream(self, messages, max_tokens):
        """Yield NPU draft tokens incrementally as they arrive."""
        payload = {
            "model": self.npu_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": True,
        }
        async with self.http_session.post(f"{self.npu_url}/v1/chat/completions", json=payload) as r:
            if r.status != 200:
                return
            async for line in r.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    pass

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        stream = body.get("stream", False)
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)

        if not isinstance(messages, list) or not messages:
            return web.json_response({"error": "messages must be a non-empty list"}, status=400)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            return web.json_response({"error": "max_tokens must be between 1 and 4096"}, status=400)

        if not stream:
            # non-stream: just forward to GPU
            gpu_payload = {"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "stream": False}
            async with self.http_session.post(f"http://127.0.0.1:{self.gpu_port}/v1/chat/completions",
                                              json=gpu_payload) as r:
                return web.json_response(await r.json())

        response = web.StreamResponse(status=200, reason="OK", headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        })
        await response.prepare(request)

        def sse(delta_text):
            chunk = {"id": f"chatcmpl-{int(time.time()*1000)}", "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"content": delta_text}}]}
            return f"data: {json.dumps(chunk)}\n\n".encode()

        try:
            # === Phase 1: NPU burst (stream incrementally for instant TTFT) ===
            npu_draft = ""
            t0 = time.perf_counter()
            if self.npu_available:
                async for delta in self.npu_stream(messages, self.npu_burst_tokens):
                    npu_draft += delta
                    await response.write(sse(delta))
            npu_ms = (time.perf_counter() - t0) * 1000

            # === Phase 2: GPU continuation (authoritative 27B answer) ===
            cont_messages = list(messages) + [{"role": "assistant", "content": npu_draft}] if npu_draft else list(messages)
            gpu_payload = {"messages": cont_messages, "max_tokens": max_tokens,
                           "temperature": temperature, "stream": True}
            async with self.http_session.post(f"http://127.0.0.1:{self.gpu_port}/v1/chat/completions",
                                              json=gpu_payload) as r:
                async for line in r.content:
                    await response.write(line)

            await response.write(b"data: [DONE]\n\n")
            print(f"[Pipeline] turn done: npu_burst={len(npu_draft.split())} tokens in {npu_ms:.0f}ms")
        except (ConnectionResetError, aiohttp.ClientConnectionError) as e:
            print(f"[Pipeline] client disconnected during stream: {e}")
        except Exception as e:
            print(f"[Pipeline] stream error: {e}")
            try:
                await response.write(sse(f"\n[error: {e}]"))
                await response.write(b"data: [DONE]\n\n")
            except Exception:
                pass
        return response

    async def start(self):
        self.http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
        if not self.start_gpu_server():
            raise RuntimeError("Cannot start pipeline without the configured GPU server")
        gpu_task = asyncio.create_task(self.wait_for_gpu_server())
        npu_task = asyncio.create_task(self.probe_npu())
        self.gpu_available, _ = await asyncio.gather(gpu_task, npu_task)
        if not self.gpu_available:
            raise RuntimeError("GPU server did not become ready")

        runner = web.AppRunner(self.app)
        await runner.setup()
        await web.TCPSite(runner, self.host, self.port).start()
        print(f"[Pipeline] handoff pipeline on http://{self.host}:{self.port} "
              f"(NPU={'on' if self.npu_available else 'off'})")

    def stop(self):
        if self.gpu_process:
            try:
                os.killpg(os.getpgid(self.gpu_process.pid), signal.SIGTERM)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-model", default=DEFAULT_GPU_MODEL)
    p.add_argument("--npu-model", default=DEFAULT_NPU_MODEL)
    p.add_argument("--port", type=int, default=PIPELINE_PORT)
    p.add_argument("--host", default="127.0.0.1", help="Listen address; use a reverse proxy for remote access")
    p.add_argument("--gpu-port", type=int, default=GPU_SERVER_PORT)
    p.add_argument("--device", default="Vulkan0")
    p.add_argument("--draft-n", type=int, default=4)
    p.add_argument("--npu-burst-tokens", type=int, default=NPU_BURST_TOKENS)
    args = p.parse_args()

    pipe = StrixHaloHybridPipeline(gpu_model_path=args.gpu_model, npu_model_name=args.npu_model,
                                   port=args.port, gpu_port=args.gpu_port, device=args.device,
                                    draft_n=args.draft_n, npu_burst_tokens=args.npu_burst_tokens,
                                    host=args.host)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(pipe.start())
        loop.run_forever()
    except KeyboardInterrupt:
        print("\n[Pipeline] shutting down")
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
