#!/usr/bin/env python3
"""
npuhalo — Unified Command Line Interface for AMD Strix Halo / Point heterogeneous inference.

Subcommands:
  doctor     Run hardware, driver, ulimit, and FastFlowLM issue #716 diagnostics
  top        Launch the live terminal monitor (NPU, iGPU, Proxy telemetry)
  proxy      Start the Always-On Watchdog & Reverse Proxy
  route      Classify a prompt using the Hybrid Intent Router
  compress   Compress an input context or tool output using the Dual NPU Compressor
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import argparse
import asyncio

__version__ = "0.1.0"

def cmd_doctor(args):
    from scripts.npuhalo_doctor import run_doctor_checks, print_doctor_report
    checks = run_doctor_checks(npu_url=args.npu_url, gpu_url=args.gpu_url)
    print_doctor_report(checks)
    if any(c.status == "FAIL" for c in checks):
        sys.exit(1)

def cmd_top(args):
    from scripts.npuhalo_top import render_dashboard, get_cpu_model, CLEAR_SCREEN
    import time

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
        print("\nExiting npuhalo top.")

def cmd_proxy(args):
    from scripts.npuhalo_proxy import main as proxy_main
    # Forward sys.argv or run directly
    sys.argv = [sys.argv[0]] + args.proxy_args
    proxy_main()

def cmd_route(args):
    from scripts.npu_router import HybridNPURouter

    async def _run():
        router = HybridNPURouter(url=args.npu_url, model=args.model)
        await router.start()
        try:
            res = await router.classify(args.prompt)
            print(f"\nPrompt : \"{args.prompt}\"")
            print(f"Route  : {res['route'].upper()} ({res['method']})")
            print(f"Reason : {res['reason']}")
            print(f"Latency: {res['latency_ms']:.2f} ms\n")
        finally:
            await router.close()

    asyncio.run(_run())

def cmd_compress(args):
    from verifier.src.compressor_sidecar import DualCompressorSidecar
    import os

    text_to_compress = args.text
    if os.path.isfile(args.text):
        with open(args.text, "r", encoding="utf-8", errors="replace") as f:
            text_to_compress = f.read()

    compressor = DualCompressorSidecar(npu_url=args.npu_url, npu_model=args.model)
    if args.mode == "input":
        res = compressor.compress_input_context(text_to_compress)
    else:
        res = compressor.compress_tool_output(
            command=args.command,
            raw_output=text_to_compress,
            generation_in_flight=False,
        )

    print(f"\nMode     : {args.mode}")
    print(f"Action   : {res.action}")
    print(f"Tokens   : {res.raw_tokens} -> {res.compressed_tokens} (saved {res.tokens_saved})")
    print(f"Latency  : {res.duration_ms:.1f} ms")
    print(f"\n--- Output ---\n{res.inserted_text}\n")

def main():
    parser = argparse.ArgumentParser(
        prog="npuhalo",
        description="AMD Strix Halo / Point Heterogeneous NPU + iGPU Toolkit",
    )
    parser.add_argument("-v", "--version", action="version", version=f"npuhalo {__version__} (XDNA 2 + RDNA 3.5)")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Run hardware, driver, and FastFlowLM diagnostics")
    p_doctor.add_argument("--npu-url", default="http://127.0.0.1:8001", help="FastFlowLM NPU endpoint")
    p_doctor.add_argument("--gpu-url", default="http://127.0.0.1:8012", help="llama-server GPU endpoint")
    p_doctor.set_defaults(func=cmd_doctor)

    # top
    p_top = subparsers.add_parser("top", help="Launch live terminal telemetry dashboard")
    p_top.add_argument("--proxy-url", default="http://127.0.0.1:8000", help="NPUHalo proxy endpoint")
    p_top.add_argument("--npu-url", default="http://127.0.0.1:8001", help="NPU FastFlowLM endpoint")
    p_top.add_argument("--gpu-url", default="http://127.0.0.1:8012", help="GPU llama-server endpoint")
    p_top.add_argument("--interval", type=float, default=1.0, help="Refresh interval (seconds)")
    p_top.add_argument("--once", action="store_true", help="Print dashboard once and exit")
    p_top.set_defaults(func=cmd_top)

    # proxy
    p_proxy = subparsers.add_parser("proxy", help="Start the Always-On Watchdog & Reverse Proxy")
    p_proxy.add_argument("proxy_args", nargs=argparse.REMAINDER, help="Arguments passed to npuhalo_proxy")
    p_proxy.set_defaults(func=cmd_proxy)

    # route
    p_route = subparsers.add_parser("route", help="Route prompt using Hybrid Intent Router")
    p_route.add_argument("prompt", help="Text prompt to classify")
    p_route.add_argument("--npu-url", default="http://127.0.0.1:8001", help="FastFlowLM NPU endpoint")
    p_route.add_argument("--model", default="qwen3.5:0.8b", help="NPU classifier model")
    p_route.set_defaults(func=cmd_route)

    # compress
    p_comp = subparsers.add_parser("compress", help="Compress prompt context or tool outputs on NPU")
    p_comp.add_argument("text", help="Text or file path to compress")
    p_comp.add_argument("--mode", choices=["input", "tool"], default="input", help="Compression mode")
    p_comp.add_argument("--command", default="shell_command", help="Command name (for tool mode)")
    p_comp.add_argument("--npu-url", default="http://127.0.0.1:8001/v1/chat/completions", help="FastFlowLM completions endpoint")
    p_comp.add_argument("--model", default="qwen3.5:0.8b", help="NPU compression model")
    p_comp.set_defaults(func=cmd_compress)

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    args.func(args)

if __name__ == "__main__":
    main()
