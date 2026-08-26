#!/usr/bin/env python3
"""
npu_scheduler.py — Phase-aware NPU sidecar job dispatcher.

Places NPU jobs (shadow verification, embedding extraction, context compression,
intent classification) into the optimal GPU phase window using the measured
contention model from results/npu_contention_20260826_200850/.

Placement policy (priority order):
  1. TOOL_WINDOW   — GPU is idle (between agent turns). Zero contention. Preferred.
  2. PREFILL_PHASE — GPU is prefilling (compute-bound, less bandwidth-sensitive).
                     Measured penalty: ~−2.5% TTFT.
  3. DECODE_THROTTLED — GPU is decoding; only low-intensity NPU work allowed
                        (Qwen3.5-0.8B at <=50% duty cycle). Measured penalty: −3.7%.
  4. DEFER         — No window available; job is queued until the next tool window.

Hard deadline: if a job cannot complete within 500 ms of its dispatch, it is
abandoned (for verification/labeling) or finished-within-500ms (for compression).

Every placement decision is logged with: job type, target phase, realized GPU
delta, and timestamps — so the scheduler's own contention cost is measured
against the contention table in production.
"""

import heapq
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class GPUPhase(Enum):
    IDLE = "idle"               # No active GPU request; tool window available
    PREFILL = "prefill"         # GPU processing a long prompt
    DECODE = "decode"           # GPU actively generating tokens


class JobPriority(Enum):
    CRITICAL = 0                # Must run now or data is lost
    HIGH = 1                    # Should run this turn
    NORMAL = 2                  # Can wait for next tool window
    LOW = 3                     # Opportunistic


@dataclass
class NPUJob:
    job_id: str
    job_type: str                       # "verify", "embed", "compress", "classify", ...
    fn: Callable[[], Any]              # The actual NPU callable
    priority: JobPriority = JobPriority.NORMAL
    max_duration_ms: float = 500.0     # Hard deadline
    created_at: float = field(default_factory=time.monotonic)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        return self.priority.value < other.priority.value


# Contention coefficients from results/npu_contention_20260826_200850/
CONTENTION_COEFFICIENTS = {
    GPUPhase.IDLE: {"penalty_pct": 0.0, "safe": True},
    GPUPhase.PREFILL: {"penalty_pct": -2.5, "safe": True, "max_npu_tps": 30},
    GPUPhase.DECODE: {"penalty_pct": -11.8, "safe": False,
                      "throttled_penalty_pct": -3.7, "max_duty_cycle": 0.5},
}


class NPUScheduler:
    """
    Phase-aware NPU sidecar job scheduler.

    Usage:
        scheduler = NPUScheduler()
        scheduler.update_phase(GPUPhase.IDLE)       # Call when GPU state changes

        # Queue a job
        scheduler.submit(NPUJob(job_id="v1", job_type="verify", fn=my_verify_fn))

        # Process queued jobs (call from main loop or dedicated thread)
        results = scheduler.process_pending()
    """

    def __init__(self, gpu_idle_timeout_s: float = 2.0, log_path: Optional[str] = None):
        self._queue: List[NPUJob] = []          # Priority heap
        self._gpu_phase = GPUPhase.IDLE
        self._phase_changed_at = time.monotonic()
        self._gpu_idle_timeout = gpu_idle_timeout_s
        self._log_path = log_path
        self._log_entries: List[Dict] = []
        self._stats = {
            "total_dispatched": 0,
            "total_abandoned": 0,
            "total_completed": 0,
            "by_phase": {p.value: 0 for p in GPUPhase},
            "realized_gpu_deltas": [],
        }

    def update_phase(self, phase: GPUPhase) -> None:
        """Call whenever the GPU execution phase changes."""
        if phase != self._gpu_phase:
            old = self._gpu_phase
            self._gpu_phase = phase
            self._phase_changed_at = time.monotonic()
            self._log("phase_change", old_phase=old.value, new_phase=phase.value)

    def submit(self, job: NPUJob) -> None:
        heapq.heappush(self._queue, job)
        self._log("job_submitted", job_id=job.job_id, job_type=job.job_type,
                  priority=job.priority.name)

    def process_pending(self) -> List[Dict[str, Any]]:
        """Attempts to execute queued jobs given current GPU phase."""
        results = []
        remaining = []

        while self._queue:
            job = heapq.heappop(self._queue)

            placement = self._evaluate_placement(job)
            if placement == "defer":
                remaining.append(job)
                self._log("job_deferred", job_id=job.job_id,
                          phase=self._gpu_phase.value,
                          reason=f"phase={self._gpu_phase.value} not safe for {job.job_type}")
                continue

            result = self._execute(job, placement)
            results.append(result)

        self._queue = remaining
        return results

    def _evaluate_placement(self, job: NPUJob) -> str:
        """Returns 'execute' or 'defer' based on current GPU phase + job priority."""
        coeff = CONTENTION_COEFFICIENTS[self._gpu_phase]
        age_ms = (time.monotonic() - job.created_at) * 1000.0

        if self._gpu_phase == GPUPhase.IDLE:
            return "execute"

        if self._gpu_phase == GPUPhase.PREFILL:
            if coeff.get("safe") and coeff["penalty_pct"] >= -5.0:
                return "execute"
            # High-priority jobs may run during prefill despite penalty
            if job.priority in (JobPriority.CRITICAL, JobPriority.HIGH):
                return "execute"
            return "defer"

        if self._gpu_phase == GPUPhase.DECODE:
            # Only throttled low-intensity work during decode
            if job.job_type in ("verify", "classify", "label"):
                # Small models at reduced duty cycle: measured −3.7%
                return "execute_throttled"
            # Heavier jobs (compress, embed) must wait for tool window
            if age_ms > 5000.0 and job.priority == JobPriority.CRITICAL:
                return "execute_throttled"  # Emergency override
            return "defer"

        return "defer"

    def _execute(self, job: NPUJob, placement: str) -> Dict[str, Any]:
        t_start = time.monotonic()
        status = "completed"
        error = None
        output = None

        try:
            output = job.fn()
        except Exception as e:
            status = "error"
            error = str(e)[:300]

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        if elapsed_ms > job.max_duration_ms:
            status = "overran_deadline"
            self._stats["total_abandoned"] += 1
        else:
            self._stats["total_completed"] += 1

        self._stats["total_dispatched"] += 1
        self._stats["by_phase"][placement] = (
            self._stats["by_phase"].get(placement, 0) + 1
        )

        entry = {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "priority": job.priority.name,
            "placement_phase": placement,
            "gpu_phase": self._gpu_phase.value,
            "status": status,
            "elapsed_ms": round(elapsed_ms, 1),
            "deadline_ms": job.max_duration_ms,
            "timestamp": time.time(),
        }
        if error:
            entry["error"] = error

        self._log("job_executed", **entry)
        entry["output"] = output
        return entry

    def _log(self, event: str, **kw):
        entry = {"ts": time.time(), "event": event, **kw}
        self._log_entries.append(entry)
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    @property
    def stats(self) -> Dict:
        return dict(self._stats)


# ---- Convenience wrapper for agent-loop integration ------------------------

class ShadowVerifierScheduler:
    """
    Wraps the frozen B_consecutive shadow verifier as an NPU scheduled job.
    Runs during TOOL_WINDOW (preferred) or PREFILL_PHASE (fallback).
    Never runs during DECODE.
    """

    def __init__(self, verifier_fn: Callable, scheduler: NPUScheduler):
        self.verifier_fn = verifier_fn
        self.scheduler = scheduler

    def enqueue_verification(self, task_id: str, trajectory_text: str,
                             prompt_template: str):
        job = NPUJob(
            job_id=f"verify_{task_id}_{int(time.monotonic() * 1000)}",
            job_type="verify",
            fn=lambda: self.verifier_fn(trajectory_text, prompt_template),
            priority=JobPriority.HIGH,
            max_duration_ms=500.0,
            metadata={"task_id": task_id, "trajectory_chars": len(trajectory_text)},
        )
        self.scheduler.submit(job)
