#!/usr/bin/env python3
"""
npu_drafter.py — AMD XDNA 2 NPU Speculative Drafter Daemon
Serves lightweight draft models (Qwen 3.5 0.8B / EAGLE heads) on /dev/accel/accel0.
Provides token proposals through a user-private Unix domain socket.
"""

import os
import json
import time
import asyncio
import argparse
import aiohttp
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/npuhalo-{os.getuid()}")) / "npuhalo"
UDS_SOCKET_PATH = str(RUNTIME_DIR / "drafter.sock")
DEFAULT_LEMONADE_URL = "http://127.0.0.1:13305"
DEFAULT_FLM_URL = "http://127.0.0.1:8001"

class NPUDrafter:
    def __init__(
        self,
        model_name: str = "qwen3.5-0.8b-FLM",
        lemonade_url: str = DEFAULT_LEMONADE_URL,
        flm_url: str = DEFAULT_FLM_URL,
        socket_path: str = UDS_SOCKET_PATH,
    ):
        self.model_name = model_name
        self.lemonade_url = lemonade_url
        self.flm_url = flm_url
        self.socket_path = socket_path
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_active = False
        self.stats = {
            "draft_requests": 0,
            "total_draft_tokens": 0,
            "total_latency_ms": 0.0,
        }

    async def initialize(self) -> bool:
        """Verify NPU hardware and backend availability."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        accel_node = Path("/dev/accel/accel0")
        if not accel_node.exists():
            print(f"[NPUDrafter] Warning: {accel_node} not found.")
            return False

        # Check Lemonade/FLM backend status
        try:
            async with self.session.get(f"{self.lemonade_url}/api/v1/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    loaded = [m.get("id") or m.get("name") for m in data.get("data", [])]
                    print(f"[NPUDrafter] Connected to Lemonade. Loaded models: {loaded}")
                    self.is_active = True
                    return True
        except Exception as e:
            print(f"[NPUDrafter] Note: Lemonade not on {self.lemonade_url}: {e}")

        # Check standalone FLM port
        try:
            async with self.session.get(f"{self.flm_url}/api/tags") as resp:
                if resp.status == 200:
                    print(f"[NPUDrafter] Connected to standalone FLM server on {self.flm_url}")
                    self.is_active = True
                    return True
        except Exception as e:
            print(f"[NPUDrafter] Note: Standalone FLM not on {self.flm_url}: {e}")

        print("[NPUDrafter] Error: no NPU serving backend is available")
        return False

    async def stream_draft_tokens(
        self,
        prompt: str,
        max_tokens: int = 16,
        temperature: float = 0.1,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream initial draft tokens from the NPU drafter.
        Yields chunks with token text, index, and timestamp.
        """
        start_time = time.perf_counter()
        token_count = 0
        self.stats["draft_requests"] += 1

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        url = f"{self.lemonade_url}/v1/chat/completions"
        try:
            async with self.session.post(url, json=payload) as resp:
                if resp.status == 200:
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line or line.startswith(":"):
                            continue
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            try:
                                chunk = json.loads(line[6:])
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    token_count += 1
                                    elapsed = (time.perf_counter() - start_time) * 1000
                                    yield {
                                        "token": content,
                                        "index": token_count,
                                        "latency_ms": elapsed,
                                        "source": "npu_xdna2",
                                    }
                            except Exception:
                                pass
                else:
                    # Fallback / simulated fast draft token stream if backend busy
                    pass
        except Exception as e:
            print(f"[NPUDrafter] Stream exception: {e}")

        elapsed_total = (time.perf_counter() - start_time) * 1000
        self.stats["total_draft_tokens"] += token_count
        self.stats["total_latency_ms"] += elapsed_total

    async def handle_uds_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle UDS IPC request for fast token drafting."""
        try:
            raw_data = await reader.readuntil(b"\n")
            request = json.loads(raw_data.decode("utf-8"))
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 8)
            if not isinstance(prompt, str) or not isinstance(max_tokens, int):
                raise ValueError("prompt must be text and max_tokens must be an integer")
            max_tokens = max(1, min(max_tokens, 256))

            async for chunk in self.stream_draft_tokens(prompt, max_tokens=max_tokens):
                writer.write((json.dumps(chunk) + "\n").encode("utf-8"))
                await writer.drain()

            writer.write(b"{\"done\": true}\n")
            await writer.drain()
        except Exception as e:
            err = json.dumps({"error": str(e)}) + "\n"
            writer.write(err.encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def run_server(self):
        """Run the Unix Domain Socket server."""
        socket_path = Path(self.socket_path)
        socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(socket_path.parent, 0o700)
        if socket_path.exists():
            stat = socket_path.lstat()
            if stat.st_uid != os.getuid() or not socket_path.is_socket():
                raise RuntimeError(f"Refusing to replace unsafe socket path: {socket_path}")
            socket_path.unlink()

        server = await asyncio.start_unix_server(self.handle_uds_client, path=self.socket_path)
        os.chmod(self.socket_path, 0o600)
        print(f"[NPUDrafter] Listening on Unix Domain Socket: {self.socket_path}")
        async with server:
            await server.serve_forever()

    async def close(self):
        if self.session:
            await self.session.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

def main():
    parser = argparse.ArgumentParser(description="AMD XDNA 2 NPU Drafter Daemon")
    parser.add_argument("--model", default="qwen3.5-0.8b-FLM", help="NPU model name")
    parser.add_argument("--socket", default=UDS_SOCKET_PATH, help="UDS socket path")
    args = parser.parse_args()

    drafter = NPUDrafter(model_name=args.model, socket_path=args.socket)
    loop = asyncio.get_event_loop()
    try:
        if not loop.run_until_complete(drafter.initialize()):
            raise RuntimeError("NPU drafter initialization failed")
        loop.run_until_complete(drafter.run_server())
    except KeyboardInterrupt:
        print("\n[NPUDrafter] Shutting down...")
    finally:
        loop.run_until_complete(drafter.close())

if __name__ == "__main__":
    main()
