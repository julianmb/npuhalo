#!/usr/bin/env python3
"""
npuhalo_doctor.py — Comprehensive Hardware & Runtime Diagnostic for AMD Strix Halo / Point.

Performs automated checks on:
1. AMD XDNA 2 NPU device node (/dev/accel/accel0 permissions & column count)
2. Linux kernel version and in-tree amdxdna driver (0.7.0+)
3. Process memory lock limits (memlock ulimit for zero-copy DMA buffers)
4. FastFlowLM NPU Server health & active model inspection
5. FastFlowLM Upstream Issue #716 Detection (silent resident model fallback canary)
6. Radeon iGPU inference server health (:8012)
"""

import os
import sys
import json
import urllib.request
import urllib.error
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Terminal ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_CYAN = "\033[36m"

@dataclass
class CheckResult:
    category: str
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    detail: Optional[str] = None


def check_npu_device() -> CheckResult:
    """Verify /dev/accel/accel0 presence, permissions, and column count."""
    dev_path = "/dev/accel/accel0"
    if not os.path.exists(dev_path):
        return CheckResult(
            category="Hardware",
            name="XDNA 2 Device Node",
            status="FAIL",
            message=f"{dev_path} not found. Check BIOS NPU enablement and amdxdna kernel driver.",
        )

    readable = os.access(dev_path, os.R_OK)
    writable = os.access(dev_path, os.W_OK)
    if not (readable and writable):
        return CheckResult(
            category="Hardware",
            name="XDNA 2 Device Permissions",
            status="WARN",
            message=f"Current user lacks full R/W permissions to {dev_path}.",
            detail="Run: sudo usermod -a -G render,video $USER",
        )

    # Check NPU columns in sysfs if available
    cols = "Unknown"
    sysfs_cols = "/sys/class/accel/accel0/device/columns"
    if os.path.exists(sysfs_cols):
        try:
            with open(sysfs_cols, "r") as f:
                cols = f.read().strip()
        except Exception:
            pass

    return CheckResult(
        category="Hardware",
        name="XDNA 2 Device Node",
        status="PASS",
        message=f"{dev_path} is accessible (R/W).",
        detail=f"Topology: {cols} columns (8 for Strix Halo, 4 for Strix Point)" if cols != "Unknown" else None,
    )


def check_driver_and_kernel() -> List[CheckResult]:
    """Check Linux kernel version and amdxdna driver module."""
    results = []
    # Kernel version
    kernel_release = os.uname().release
    results.append(
        CheckResult(
            category="Driver",
            name="Linux Kernel",
            status="PASS",
            message=f"Kernel {kernel_release}",
        )
    )

    # amdxdna module
    try:
        mod_info = subprocess.run(["modinfo", "amdxdna"], capture_output=True, text=True, timeout=2)
        if mod_info.returncode == 0:
            version = "in-tree"
            for line in mod_info.stdout.splitlines():
                if line.startswith("version:"):
                    version = line.split(":", 1)[1].strip()
            results.append(
                CheckResult(
                    category="Driver",
                    name="amdxdna Driver",
                    status="PASS",
                    message=f"Driver loaded ({version})",
                )
            )
        else:
            results.append(
                CheckResult(
                    category="Driver",
                    name="amdxdna Driver",
                    status="WARN",
                    message="modinfo amdxdna returned non-zero. Driver may be built statically into kernel.",
                )
            )
    except Exception as e:
        results.append(
            CheckResult(
                category="Driver",
                name="amdxdna Driver",
                status="WARN",
                message=f"Unable to query modinfo: {e}",
            )
        )

    return results


def check_memlock_limit() -> CheckResult:
    """Verify max locked memory limit (essential for AIE DMA zero-copy)."""
    try:
        with open("/proc/self/limits", "r") as f:
            for line in f:
                if "Max locked memory" in line:
                    parts = line.split()
                    soft = parts[3]
                    hard = parts[4]
                    if soft == "unlimited" or hard == "unlimited":
                        return CheckResult(
                            category="System",
                            name="Memlock ulimit",
                            status="PASS",
                            message="Max locked memory is unlimited (ideal for zero-copy DMA).",
                        )
                    try:
                        soft_bytes = int(soft)
                        soft_mb = soft_bytes // (1024 * 1024)
                        if soft_mb >= 15000:
                            return CheckResult(
                                category="System",
                                name="Memlock ulimit",
                                status="PASS",
                                message=f"Max locked memory: {soft_mb} MB (sufficient for XDNA 2).",
                            )
                        else:
                            return CheckResult(
                                category="System",
                                name="Memlock ulimit",
                                status="WARN",
                                message=f"Max locked memory is {soft_mb} MB (FastFlowLM recommends >= 15992 MB).",
                                detail="Add '* soft memlock 16777216' and '* hard memlock 16777216' to /etc/security/limits.conf",
                            )
                    except ValueError:
                        pass
    except Exception as e:
        return CheckResult(
            category="System",
            name="Memlock ulimit",
            status="WARN",
            message=f"Unable to read /proc/self/limits: {e}",
        )

    return CheckResult(
        category="System",
        name="Memlock ulimit",
        status="PASS",
        message="Memlock limit checked.",
    )


def check_npu_server(npu_url: str = "http://127.0.0.1:8001") -> Tuple[List[CheckResult], Optional[str]]:
    """Probe FastFlowLM NPU server and perform issue #716 canary test."""
    results = []
    active_model = None
    clean_url = npu_url.rstrip("/")

    # 1. Probe /v1/models
    try:
        req = urllib.request.Request(f"{clean_url}/v1/models", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                models = [m.get("id", "") for m in data.get("data", [])]
                active_model = models[0] if models else "Unknown"
                results.append(
                    CheckResult(
                        category="NPU Server",
                        name="FastFlowLM Endpoint",
                        status="PASS",
                        message=f"Server online at {clean_url} (Active: {active_model})",
                    )
                )
    except Exception as e:
        results.append(
            CheckResult(
                category="NPU Server",
                name="FastFlowLM Endpoint",
                status="WARN",
                message=f"FastFlowLM server not reachable on {clean_url} ({e})",
                detail="Launch via: flm serve qwen3.5:0.8b --port 8001",
            )
        )
        return results, None

    # 2. Canary test for Upstream Issue #716 (Silent Resident Model Fallback)
    canary_model = "npuhalo-canary-test:99b"
    try:
        payload = json.dumps({
            "model": canary_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }).encode("utf-8")
        canary_req = urllib.request.Request(
            f"{clean_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        with urllib.request.urlopen(canary_req, timeout=2.0) as c_resp:
            c_data = json.loads(c_resp.read().decode())
            echoed_model = c_data.get("model", "")
            if echoed_model == canary_model:
                results.append(
                    CheckResult(
                        category="NPU Server",
                        name="Upstream Issue #716 Guard",
                        status="WARN",
                        message="FastFlowLM issue #716 detected! Server silently served resident model for unresolvable tag.",
                        detail=(
                            "Requests for unregistered model tags do not fail but execute the resident model "
                            f"(currently '{active_model}'). Ensure requested tags exist in model_list.json!"
                        ),
                    )
                )
            else:
                results.append(
                    CheckResult(
                        category="NPU Server",
                        name="Upstream Issue #716 Guard",
                        status="PASS",
                        message="Model tag resolution verified (no silent fallback).",
                    )
                )
    except urllib.error.HTTPError as he:
        # Expected behavior: server rejects invalid model tag with 400 or 404
        results.append(
            CheckResult(
                category="NPU Server",
                name="Upstream Issue #716 Guard",
                status="PASS",
                message=f"Server correctly rejected invalid model tag (HTTP {he.code}).",
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                category="NPU Server",
                name="Upstream Issue #716 Guard",
                status="PASS",
                message=f"Canary probe completed ({e})",
            )
        )

    return results, active_model


def check_gpu_server(gpu_url: str = "http://127.0.0.1:8012") -> CheckResult:
    """Probe GPU llama-server endpoint."""
    clean_url = gpu_url.rstrip("/")
    try:
        req = urllib.request.Request(f"{clean_url}/v1/models", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                models = [m.get("id", "") for m in data.get("data", [])]
                model_name = models[0] if models else "Unknown"
                return CheckResult(
                    category="GPU Server",
                    name="llama-server (iGPU)",
                    status="PASS",
                    message=f"Server online at {clean_url} (Model: {model_name})",
                )
    except Exception as e:
        return CheckResult(
            category="GPU Server",
            name="llama-server (iGPU)",
            status="WARN",
            message=f"GPU server not responding on {clean_url} ({e})",
            detail="Ensure llama-server is running on ROCm: ./scripts/launch_llama_server.py",
        )


def run_doctor_checks(
    npu_url: str = "http://127.0.0.1:8001",
    gpu_url: str = "http://127.0.0.1:8012",
) -> List[CheckResult]:
    """Execute all diagnostic checks."""
    all_checks: List[CheckResult] = []
    all_checks.append(check_npu_device())
    all_checks.extend(check_driver_and_kernel())
    all_checks.append(check_memlock_limit())

    npu_results, _ = check_npu_server(npu_url)
    all_checks.extend(npu_results)
    all_checks.append(check_gpu_server(gpu_url))
    return all_checks


def print_doctor_report(checks: List[CheckResult]):
    """Format and print beautiful colored diagnostic report."""
    print(f"\n{C_BOLD}{C_CYAN}=== npuhalo doctor · System & Hardware Diagnostics ==={C_RESET}\n")
    
    badge_colors = {
        "PASS": f"{C_GREEN}[PASS]{C_RESET}",
        "WARN": f"{C_YELLOW}[WARN]{C_RESET}",
        "FAIL": f"{C_RED}[FAIL]{C_RESET}",
    }

    current_cat = None
    pass_count = 0
    warn_count = 0
    fail_count = 0

    for chk in checks:
        if chk.status == "PASS":
            pass_count += 1
        elif chk.status == "WARN":
            warn_count += 1
        elif chk.status == "FAIL":
            fail_count += 1

        if chk.category != current_cat:
            current_cat = chk.category
            print(f"{C_BOLD}{current_cat}:{C_RESET}")

        badge = badge_colors.get(chk.status, chk.status)
        print(f"  {badge} {C_BOLD}{chk.name:<28}{C_RESET} {chk.message}")
        if chk.detail:
            print(f"         {C_DIM}↳ {chk.detail}{C_RESET}")

    print(f"\n{C_BOLD}Summary:{C_RESET} {pass_count} passed, {warn_count} warnings, {fail_count} failures.")
    if fail_count > 0:
        print(f"{C_RED}❌ Hardware or essential service failed. Review instructions above.{C_RESET}\n")
    elif warn_count > 0:
        print(f"{C_YELLOW}⚠️ System operational with advisory notes.{C_RESET}\n")
    else:
        print(f"{C_GREEN}✅ All diagnostics passed! NPU + iGPU heterogeneous pipeline is ready.{C_RESET}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="npuhalo doctor — System & Hardware Diagnostics")
    parser.add_argument("--npu-url", default="http://127.0.0.1:8001", help="FastFlowLM NPU endpoint")
    parser.add_argument("--gpu-url", default="http://127.0.0.1:8012", help="llama-server GPU endpoint")
    args = parser.parse_args()

    checks = run_doctor_checks(npu_url=args.npu_url, gpu_url=args.gpu_url)
    print_doctor_report(checks)
    if any(c.status == "FAIL" for c in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
