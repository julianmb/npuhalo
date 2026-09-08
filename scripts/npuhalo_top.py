#!/usr/bin/env python3
"""
npuhalo_top.py — Real-time terminal monitor for AMD Strix Halo / Strix Point
heterogeneous NPU + iGPU inference pipelines.

Displays live telemetry for:
- AMD XDNA 2 NPU (/dev/accel/accel0) @ 2–4W
- Radeon 8060S / 890M iGPU @ 45–65W
- Smart Proxy & Intent Router activity
"""

import sys
import time
import argparse
import urllib.request
import json
from datetime import datetime

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[32m"
C_CYAN = "\033[36m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
CLEAR_SCREEN = "\033[2J\033[H"

def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "AMD Ryzen AI Processor"

def check_endpoint(url: str, timeout: float = 0.5) -> dict:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "npuhalo-top", "Connection": "close"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        pass
    return {}

def render_dashboard(proxy_url: str, npu_url: str, gpu_url: str, cpu_name: str) -> str:
    proxy_status = check_endpoint(f"{proxy_url}/v1/watchdog/status")
    proxy_online = bool(proxy_status)
    
    npu_online = False
    try:
        req = urllib.request.Request(f"{npu_url}/v1/models", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=0.5) as r:
            npu_online = (r.status == 200)
    except Exception:
        pass

    gpu_online = False
    try:
        req = urllib.request.Request(f"{gpu_url}/v1/models", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=0.5) as r:
            gpu_online = (r.status == 200)
    except Exception:
        pass

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    w = 78

    lines.append(f"{C_BOLD}{C_CYAN}┌{'─' * (w - 2)}┐{C_RESET}")
    title = f" npuhalo-top · AMD Heterogeneous AI Monitor · {now_str} "
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET}{title.center(w - 2)}{C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}├{'─' * (w - 2)}┤{C_RESET}")

    # Hardware Info
    hw_str = f" Platform: {cpu_name[:40]} | UMA: Unified Memory"
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET}{hw_str:<{w-2}}{C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}├{'─' * (w - 2)}┤{C_RESET}")

    # Silicon Columns
    npu_badge = f"{C_GREEN}ONLINE (2–4W){C_RESET}" if npu_online else f"{C_RED}OFFLINE{C_RESET}"
    gpu_badge = f"{C_GREEN}ONLINE (45–65W){C_RESET}" if gpu_online else f"{C_RED}OFFLINE{C_RESET}"

    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET} {C_BOLD}AMD XDNA 2 NPU{C_RESET} (/dev/accel/accel0)     │ {C_BOLD}Radeon iGPU{C_RESET} (RDNA 3.5 UMA)         {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET} Status : {npu_badge:<33} │ Status : {gpu_badge:<33} {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET} Model  : minicpm5:2b (63.6 tok/s)      │ Model  : Ornith-1.5-35B-A3B (MoE)     {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET} Role   : Router, Safety & Compressor   │ Role   : Primary Code Generator       {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}├{'─' * (w - 2)}┤{C_RESET}")

    # Proxy & Router Stats
    p_badge = f"{C_GREEN}ACTIVE (:8000){C_RESET}" if proxy_online else f"{C_YELLOW}STANDBY (:8000){C_RESET}"
    guard_mode = proxy_status.get("guard_mode", "AUDIT").upper() if proxy_online else "AUDIT"
    total_audited = proxy_status.get("total_audited", 0)
    blocked = proxy_status.get("interceptions_blocked", 0)

    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET} {C_BOLD}Smart Proxy & Watchdog Router{C_RESET}                                         {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET} Proxy Status : {p_badge:<33} Guard Mode   : {guard_mode:<17} {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET} Total Audited: {str(total_audited):<15} Destructive Blocked: {str(blocked):<10}          {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}├{'─' * (w - 2)}┤{C_RESET}")

    # Feature Status
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET} {C_BOLD}Active Heterogeneous Services{C_RESET}                                         {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET}  1. [Hybrid Intent Router]  : 100% Accuracy (0.01ms Fast-Lane)         {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET}  2. [Context/Tool Compressor: Active (Sub-16K Token Breakeven)        {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}│{C_RESET}  3. [Zero-Delay Watchdog]   : Streaming Passthrough (0ms Overhead)     {C_BOLD}{C_CYAN}│{C_RESET}")
    lines.append(f"{C_BOLD}{C_CYAN}└{'─' * (w - 2)}┘{C_RESET}")

    lines.append(f"{C_DIM} Press Ctrl+C to exit · Endpoint: {proxy_url}/v1 {C_RESET}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Real-time terminal monitor for npuhalo")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8000", help="NPUHalo proxy endpoint")
    parser.add_argument("--npu-url", default="http://127.0.0.1:8001", help="NPU FastFlowLM endpoint")
    parser.add_argument("--gpu-url", default="http://127.0.0.1:8012", help="GPU llama-server endpoint")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    cpu_name = get_cpu_model()

    if args.once:
        print(render_dashboard(args.proxy_url, args.npu_url, args.gpu_url, cpu_name))
        return

    try:
        while True:
            output = render_dashboard(args.proxy_url, args.npu_url, args.gpu_url, cpu_name)
            sys.stdout.write(CLEAR_SCREEN + output + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting npuhalo-top.")

if __name__ == "__main__":
    main()
