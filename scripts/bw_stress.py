#!/usr/bin/env python3
"""
bw_stress.py — Phase 0 contention proxy: controlled DRAM read-bandwidth generator.

Emulates the memory-bus traffic an NPU draft head would add if it streamed its
weights from DRAM while the iGPU runs the target model (overlapped speculative
drafting). CPU reads contend at the same LPDDR5X memory controller as NPU reads,
making this a valid proxy for NPU-side DRAM traffic.

Usage:
  python3 scripts/bw_stress.py --peak --workers 2 --duration 6      # calibrate max GB/s
  python3 scripts/bw_stress.py --target 16 --workers 2 --duration 120
  python3 scripts/bw_stress.py --target 16 --workers 2 --duration 120 --daemon
"""

import argparse
import multiprocessing as mp
import numpy as np
import os
from pathlib import Path
import time

RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/npuhalo-{os.getuid()}")) / "npuhalo"

def worker(target_gbps: float, duration: float, wid: int, peak: bool):
    # 256 MiB per worker: far exceeds all caches -> genuine DRAM reads
    buf = np.ones(64 * 1024 * 1024, dtype=np.float32)
    nbytes = buf.nbytes
    t_end = time.time() + duration
    t0 = time.time()
    reads = 0
    sink = 0.0
    while time.time() < t_end:
        sink += float(np.add.reduce(buf))  # read-heavy reduction
        reads += nbytes
        if not peak and target_gbps > 0:
            el = time.time() - t0
            if el > 0.2:
                rate = reads / el / 1e9  # GB/s
                if rate > target_gbps:
                    over = 1.0 - target_gbps / rate
                    time.sleep(min(over * 0.1, 0.05))
    el = max(time.time() - t0, 1e-9)
    rate = reads / el / 1e9
    print(f"[worker {wid}] {rate:.2f} GB/s sustained over {el:.1f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.0, help="aggregate target GB/s (0/absent + --peak = unthrottled)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--peak", action="store_true", help="unthrottled calibration mode")
    ap.add_argument("--daemon", action="store_true", help="double-fork into background")
    args = ap.parse_args()

    if args.daemon:
        RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(RUNTIME_DIR, 0o700)
        pid = os.fork()
        if pid > 0:
            print(f"daemonized (intermediate pid {pid})")
            return
        os.setsid()
        if os.fork() > 0:
            os._exit(0)
        log = (RUNTIME_DIR / "bw_stress.log").open("ab", buffering=0)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        fd = os.open(RUNTIME_DIR / "bw_stress.pid", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as pid_file:
            pid_file.write(str(os.getpid()))

    per_worker = args.target / args.workers if args.target > 0 else 0.0
    peak = args.peak or args.target == 0
    procs = []
    for i in range(args.workers):
        p = mp.Process(target=worker, args=(per_worker, args.duration, i, peak))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("bw_stress: all workers done", flush=True)

if __name__ == "__main__":
    main()
