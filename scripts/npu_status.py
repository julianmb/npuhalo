#!/usr/bin/env python3
"""
npu_status.py — AMD XDNA 2 NPU Status & Diagnostic Tool for Strix Halo
"""

import os
import subprocess
from pathlib import Path

def color(text, code): return f"\033[{code}m{text}\033[0m"
def green(text): return color(text, "1;32")
def yellow(text): return color(text, "1;33")
def cyan(text): return color(text, "1;36")
def bold(text): return color(text, "1")
def red(text): return color(text, "1;31")

def main():
    print("\n" + "=" * 80)
    print(bold(" 🧠 AMD XDNA 2 NPU DIAGNOSTIC TRIAGE (STRIX HALO)"))
    print("=" * 80)

    # 1. Device node & SVA access
    accel_node = Path("/dev/accel/accel0")
    if accel_node.exists():
        print(f" • Device Node:          {green('EXISTS')} (/dev/accel/accel0)")
        try:
            fd = os.open(str(accel_node), os.O_RDWR)
            print(f" • SVA Kernel Access:    {green('PASSED')} (fd={fd})")
            os.close(fd)
        except Exception as e:
            print(f" • SVA Kernel Access:    {red('FAILED')} ({e})")
    else:
        print(f" • Device Node:          {red('MISSING')} (/dev/accel/accel0 not found)")

    # 2. Kernel Module
    try:
        lsmod = subprocess.run("lsmod | grep amdxdna", shell=True, capture_output=True, text=True).stdout
        if "amdxdna" in lsmod:
            print(f" • Kernel Driver:        {green('ACTIVE')} (amdxdna.ko)")
        else:
            print(f" • Kernel Driver:        {yellow('INACTIVE')} (amdxdna not loaded)")
    except Exception:
        pass

    # 3. Kernel Command Line
    try:
        with open("/proc/cmdline") as f:
            cmdline = f.read()
            if "iommu.passthrough=0" in cmdline or "amd_iommu=on" in cmdline:
                print(f" • IOMMU SVA Boot Flag:  {green('ENABLED')} (iommu.passthrough=0)")
            else:
                print(f" • IOMMU SVA Boot Flag:  {yellow('WARN')} ({cmdline.strip()})")
    except Exception:
        pass

    # 4. XRT Installation
    xrt_smi = Path("/opt/xilinx/xrt/bin/xrt-smi")
    if xrt_smi.exists() and os.access(xrt_smi, os.X_OK):
        print(f" • XRT 2.26 Toolchain:   {green('INSTALLED')} ({xrt_smi})")
        print("-" * 80)
        print(bold(" Running 'xrt-smi examine -r platform':\n"))
        env = os.environ.copy()
        env["XILINX_XRT"] = "/opt/xilinx/xrt"
        env["PATH"] = f"/opt/xilinx/xrt/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"/opt/xilinx/xrt/lib:{env.get('LD_LIBRARY_PATH', '')}"
        subprocess.run([str(xrt_smi), "examine", "-r", "platform"], env=env)
    else:
        print(f" • XRT Toolchain:        {yellow('NOT FOUND')} (check /opt/xilinx/xrt)")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
