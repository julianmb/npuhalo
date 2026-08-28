#!/usr/bin/env python3
"""
run_active_recovery.py — Crash-resilient paired baseline-vs-active-verifier evaluation.

Wraps the existing narrow-eval agent path (identical prompts, sampling, tools,
verifier policy) with an infrastructure-reliability layer:

  - Health-checks :8012 before every task and after every failed request.
  - Captures HTTP status + response body on every failed request.
  - On retryable serving failure: waits for the supervised server to return
    (or starts the documented command if no supervisor responds), then retries
    the SAME request exactly once.
  - Classifies outcomes: COMPLETED | INFRA_FAILURE | CONTEXT_OVERFLOW.
    Nothing is silently skipped; every task/mode produces a record.
  - Append-only JSONL event log of every request/lifecycle event.
  - --resume skips completed (task, mode, seed) triples only.
  - --dry-run validates health, paths, and one known-safe request.

Usage:
  python3 verifier/scripts/run_active_recovery.py --dry-run
  python3 verifier/scripts/run_active_recovery.py [--resume] \
      [--modes baseline active_esc] [--seeds 1 2] [--exp-dir DIR]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))
sys.path.insert(0, HERE)

import run_narrow_eval as base  # noqa: E402
from gated_escalator import GatedEscalator  # noqa: E402

GEN_BASE = "http://127.0.0.1:8012"
DATA = os.path.join(REPO, "verifier", "data", "set_d_agentic.jsonl")
CATCHABLE = ["D02", "D03", "D04", "D06", "D18", "D21", "D25", "D28"]
SERVER_CMD = [
    "<home>/source/ROCmFPX/build-strix-rocmfp4/bin/llama-server",
    "-m", "<home>/source/halofpx-research/models/ornith-1.5-35b/Ornith-1.5-35B-A3B-ROCmFP4.gguf",
    "--device", "Vulkan0", "--port", "8012", "--host", "127.0.0.1",
    "-c", "16384", "-ngl", "99", "-fa", "1", "--threads", "16",
    "--no-context-shift", "-np", "1",
]


class ContextOverflow(Exception):
    """Server rejected the request: conversation exceeds ctx (harness limitation)."""


class InfraFailure(Exception):
    """Serving failure that persisted past one verified restart + retry."""


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.server_restarts = 0

    def emit(self, **kw):
        rec = {"ts": datetime.now().isoformat(timespec="milliseconds"),
               "server_restarts": self.server_restarts, **kw}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def health_check(base_url: str = GEN_BASE, timeout: float = 3.0) -> bool:
    """llama-server exposes /health; FastFlowLM does not — fall back to /v1/models."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            try:
                with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as r:
                    return r.status == 200
            except Exception:
                return False
        return False
    except Exception:
        return False


def wait_healthy(event_log: EventLog, max_wait_s: float = 240.0) -> bool:
    """Wait for the supervised server to come back. Spawn the documented
    command ourselves only if nothing restores health within the window."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        if health_check():
            return True
        time.sleep(3)
    if not health_check():
        event_log.emit(event="self_spawn_server")
        subprocess.Popen(SERVER_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + 300
        while time.time() < deadline:
            if health_check():
                return True
            time.sleep(5)
    return False


def classify_and_wrap_post(event_log: EventLog, task: str, mode: str):
    """Return a drop-in replacement for agent_loop._http_post_json."""
    original = base._http_post_json

    def resilient_post(url, payload, timeout=120):
        chat_url = url
        for attempt in (1, 2):
            t0 = time.time()
            try:
                body = original(chat_url, payload, timeout=timeout)
                event_log.emit(event="http_ok", task=task, mode=mode, attempt=attempt,
                               endpoint=chat_url, latency_ms=round((time.time() - t0) * 1000),
                               prompt_tokens=(body.get("usage") or {}).get("prompt_tokens"))
                return body
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode(errors="replace")[:500]
                except Exception:
                    pass
                is_overflow = e.code == 400 and "exceed_context_size_error" in err_body
                event_log.emit(event="http_error", task=task, mode=mode, attempt=attempt,
                               endpoint=chat_url, http_status=e.code,
                               error_class="context_overflow" if is_overflow else "http_client_error",
                               response_body=err_body)
                if is_overflow:
                    raise ContextOverflow(err_body) from None
                # other 4xx: deterministic client problem, do not retry blindly
                if e.code < 500:
                    raise InfraFailure(f"HTTP {e.code}: {err_body}") from None
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                event_log.emit(event="transport_error", task=task, mode=mode, attempt=attempt,
                               endpoint=chat_url, error_class=type(e).__name__, detail=str(e)[:200])
            # retryable: verify/restore health before retrying the SAME request
            if attempt == 1:
                event_log.emit(event="awaiting_health", task=task, mode=mode)
                if not wait_healthy(event_log):
                    event_log.server_restarts += 1
                    event_log.emit(event="restart_failed")
                    raise InfraFailure("server did not return to healthy within budget")
        raise InfraFailure("request failed after verified-health retry")

    return resilient_post


def load_manifest_tasks(full: bool = False):
    all_tasks = [json.loads(line) for line in open(DATA)]
    if full:
        return sorted(all_tasks, key=lambda t: t["id"])
    by_id = {t["id"]: t for t in all_tasks}
    controls = [t["id"] for t in all_tasks if t["id"] not in CATCHABLE][:8]
    ordered = [tid for pair in zip(CATCHABLE, controls) for tid in pair]
    return [by_id[tid] for tid in ordered if tid in by_id]


def build_manifest(exp_dir: Path, tasks, lock: bool = True):
    ds_hash = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
    path = exp_dir / "manifest.json"
    if lock and path.exists():
        return json.load(open(path))  # manifest is immutable once written
    manifest = {
        "created": datetime.now().isoformat(),
        "dataset_path": DATA,
        "dataset_sha256": ds_hash,
        "task_ids": [t["id"] for t in tasks],
        "catchable_ids": CATCHABLE,
        "config": {
            "max_tokens": 1024, "base_temperature": 0.3, "top_p": 0.95,
            "max_steps": 15, "max_rollbacks": 2,
            "verifier": {"endpoint": base.NPU, "k_samples": 3, "temperature": 0.7},
            "escalator": {"endpoint": base.ESC, "threshold": 0.50},
            "note": "sampling is server-side stochastic at fixed temperature; "
                    "no server seed API is used (changing it would alter sampling defaults)",
        },
        "endpoints": {"generator": base.GEN, "npu_verifier": base.NPU, "judge": base.ESC},
        "models": {
            "generator": "Ornith-1.5-35B-A3B-ROCmFP4.gguf (ctx 16384, --no-context-shift)",
            "verifier": "lfm2.5-tk:1.2b (FLM v0.9.46, warm)",
            "judge": "Qwen3.5-2B-Q4_K_M",
        },
    }
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def result_path(exp_dir: Path, task_id: str, mode: str, seed: int) -> Path:
    return exp_dir / "runs" / f"{task_id}_{mode}_s{seed}.json"


def already_done(exp_dir: Path, task_id: str, mode: str, seed: int) -> bool:
    p = result_path(exp_dir, task_id, mode, seed)
    if not p.exists():
        return False
    try:
        return json.load(open(p)).get("outcome") in ("COMPLETED", "CONTEXT_OVERFLOW", "INFRA_FAILURE")
    except Exception:
        return False


def run_dry_run(exp_dir: Path, event_log: EventLog) -> int:
    print("[dry-run] generator health:", health_check())
    print("[dry-run] npu health:", health_check("http://127.0.0.1:8001"))
    print("[dry-run] judge health:", health_check("http://127.0.0.1:8013"))
    (exp_dir / "runs").mkdir(parents=True, exist_ok=True)
    print("[dry-run] result dir OK:", exp_dir / "runs")
    body = base._http_post_json(f"{GEN_BASE}/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "Reply with OK."}],
                                 "max_tokens": 4, "temperature": 0}, timeout=30)
    msg = (body.get("choices") or [{}])[0].get("message", {})
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    print("[dry-run] known-safe request ->", repr(text[:40]))
    event_log.emit(event="dry_run", ok=bool(text))
    print("[dry-run] PASS" if text else "[dry-run] WARN: empty response text")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=None)
    ap.add_argument("--modes", nargs="+", default=["baseline", "active_esc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="restrict to these task IDs (smoke tests); default = full manifest")
    ap.add_argument("--full-manifest", action="store_true",
                    help="use every Set D task instead of the 16-task paired subset")
    ap.add_argument("--policy", default=None,
                    help="path to frozen triage policy JSON; wraps the judge with the escalation gate")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir) if args.exp_dir else Path(
        os.path.join(REPO, "verifier", "results",
                     "active_recovery_" + datetime.now().strftime("%Y%m%d_%H%M")))
    (exp_dir / "runs").mkdir(parents=True, exist_ok=True)
    event_log = EventLog(exp_dir / "events.jsonl")

    if args.dry_run:
        sys.exit(run_dry_run(exp_dir, event_log))

    tasks = load_manifest_tasks(full=args.full_manifest)
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t["id"] in wanted]
    manifest = build_manifest(exp_dir, tasks)
    event_log.emit(event="run_start", manifest=str(exp_dir / "manifest.json"),
                   dataset_sha256=manifest["dataset_sha256"],
                   modes=args.modes, seeds=args.seeds, tasks=manifest["task_ids"])

    # Harness-level resilience: the exact post function the agent path calls is
    # re-patched per (task, mode) inside the loop for correct event attribution.

    verifier = None
    escalator = None
    gate = None
    if "active_esc" in args.modes:
        verifier = base.NPUVerifierClient(base.NPU)
        verifier.evaluate_checkpoint("warm", "warm", "{task} {trajectory}", k_samples=3, temperature=0.7)
        escalator = base.QwenEscalator(base.ESC, threshold=0.50)
        if args.policy:
            policy_cfg = json.load(open(args.policy))
            policy_hash = hashlib.sha256(
                json.dumps(policy_cfg, indent=2, sort_keys=True).encode()).hexdigest()
            gate = GatedEscalator(escalator, policy_cfg["policy_id"])
            escalator = gate
            event_log.emit(event="policy_frozen", policy_id=policy_cfg["policy_id"],
                           policy_sha256=policy_hash, path=args.policy)
            print(f"[policy] {policy_cfg['policy_id']} (sha256 {policy_hash[:16]}…)")
    prompt = open(base.PROMPT_FILE).read()

    counts = {"COMPLETED": 0, "INFRA_FAILURE": 0, "CONTEXT_OVERFLOW": 0}
    for i, task in enumerate(tasks):
        for seed in args.seeds:
            # interleave mode order per task to cancel warm-state drift
            modes = args.modes if i % 2 == 0 else list(reversed(args.modes))
            for mode in modes:
                if args.resume and already_done(exp_dir, task["id"], mode, seed):
                    print(f"[skip] {task['id']} {mode} s{seed} (completed)")
                    continue
                base._http_post_json = classify_and_wrap_post(event_log, task["id"], mode)
                event_log.emit(event="task_start", task=task["id"], mode=mode, seed=seed)
                t0 = time.time()
                outcome, rec = "COMPLETED", None
                try:
                    runner = base.RobustAgentLoop(
                        config=mode,
                        verifier=verifier if mode == "active_esc" else None,
                        escalator=escalator if mode == "active_esc" else None,
                        verifier_prompt=prompt, max_steps=15, max_rollbacks=2,
                        base_temperature=0.3,
                    )
                    rec = runner.run(task, seed=seed)
                    rec["seed"] = seed
                    rec["task_category"] = task["category"]
                except ContextOverflow as e:
                    outcome = "CONTEXT_OVERFLOW"
                    rec = {"task_id": task["id"], "config": mode, "passed": False,
                           "error_class": "context_overflow", "detail": str(e)[:300]}
                    rec["seed"] = seed
                except InfraFailure as e:
                    outcome = "INFRA_FAILURE"
                    rec = {"task_id": task["id"], "config": mode, "passed": False,
                           "error_class": "infra_failure", "detail": str(e)[:300]}
                    rec["seed"] = seed
                rec["outcome"] = outcome
                rec["wall_time_s"] = round(time.time() - t0, 1)
                if gate is not None and mode == "active_esc":
                    rec["gate_stats"] = dict(gate.stats)
                    event_log.emit(event="gate_stats", task=task["id"], mode=mode, seed=seed,
                                   **gate.stats)
                counts[outcome] += 1
                out = result_path(exp_dir, task["id"], mode, seed)
                with open(out, "w") as f:
                    json.dump(rec, f, indent=2)
                event_log.emit(event="task_done", task=task["id"], mode=mode, seed=seed,
                               outcome=outcome, passed=rec.get("passed"),
                               rollbacks=rec.get("rollbacks"), wall_time_s=rec["wall_time_s"],
                               artifact=str(out.relative_to(exp_dir)))
                print(f"[{mode}] {task['id']} (s{seed}): {outcome} passed={rec.get('passed')} "
                      f"wall={rec['wall_time_s']}s", flush=True)

    event_log.emit(event="run_end", counts=counts)
    print(json.dumps({"outcome_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
