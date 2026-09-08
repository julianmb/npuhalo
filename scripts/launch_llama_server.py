#!/usr/bin/env python3
"""Launch and stop a detached localhost llama-server for benchmark runs."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

LLAMA_ROOT = os.environ.get("ROCmFPX_LLAMA_BUILD", "build-strix-rocmfp4")
LLAMA = os.environ.get("LLAMA_SERVER_BIN", os.path.join(LLAMA_ROOT, "bin", "llama-server"))
MODEL = os.environ.get("QWEN38_GGUF")
RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/npuhalo-{os.getuid()}")) / "npuhalo"


def runtime_path(port: int, suffix: str) -> Path:
    RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(RUNTIME_DIR, 0o700)
    return RUNTIME_DIR / f"llama_bench_{port}.{suffix}"


def common_cmd(port: int) -> list[str]:
    if not MODEL:
        raise RuntimeError("Set QWEN38_GGUF to the benchmark model path")
    return [
        LLAMA, "-m", MODEL, "--device", "Vulkan0", "--no-mmap",
        "-ngl", "99", "-fa", "1", "-c", "8192", "-b", "2048", "-ub", "2048",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
        "--spec-draft-p-min", "0.0", "--reasoning", "off",
        "--port", str(port), "--host", "127.0.0.1",
    ]


def stop(port: int) -> None:
    pid_path = runtime_path(port, "pid")
    try:
        pid = int(pid_path.read_text().strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if "llama-server" not in cmdline or f"--port {port}" not in cmdline:
            raise RuntimeError(f"PID {pid} does not match the expected llama-server command")
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        pid_path.unlink(missing_ok=True)
        print(f"sent SIGTERM to llama-server process group {pid}")
    except Exception as exc:
        print(f"stop: {exc}")


def launch(port: int, cache: bool) -> subprocess.Popen:
    cmd = common_cmd(port)
    cmd += ["--cache-prompt", "--cache-reuse", "256", "-ctxcp", "16"] if cache else [
        "--no-cache-prompt", "--cache-reuse", "0",
    ]
    log_path = runtime_path(port, "log")
    log = log_path.open("wb")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path = runtime_path(port, "pid")
    fd = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as pid_file:
        pid_file.write(str(proc.pid))
    print(f"daemonized, pid={proc.pid}, log={log_path}", flush=True)
    return proc


def wait_ready(port: int, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        stop(args.port)
        return
    if not (args.cache ^ args.no_cache):
        sys.exit("specify exactly one of --cache / --no-cache")
    launch(args.port, args.cache)
    print(f"server {'ready' if wait_ready(args.port) else 'NOT READY after timeout'}")


if __name__ == "__main__":
    main()
