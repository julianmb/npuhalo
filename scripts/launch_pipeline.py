#!/usr/bin/env python3
"""Launch the experimental handoff pipeline as a detached local process."""

import os
from pathlib import Path
import subprocess
import sys

RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/npuhalo-{os.getuid()}")) / "npuhalo"


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("usage: launch_pipeline.py [run_pipeline.py arguments]")
        print("Launches scripts/run_pipeline.py as a detached localhost service.")
        return
    RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(RUNTIME_DIR, 0o700)
    args = sys.argv[1:] or ["--device", "Vulkan0", "--draft-n", "4"]
    repo_root = Path(__file__).resolve().parent.parent
    log_path = RUNTIME_DIR / "pipeline.log"
    log = log_path.open("ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, "-u", str(repo_root / "scripts" / "run_pipeline.py"), *args],
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path = RUNTIME_DIR / "pipeline.pid"
    fd = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as pid_file:
        pid_file.write(str(proc.pid))
    print(f"detached pipeline pid={proc.pid}, log={log_path}")


if __name__ == "__main__":
    main()
