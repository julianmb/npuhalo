#!/usr/bin/env python3
"""
agent_loop.py — Minimal agentic tool-use loop for Set D evaluation.

Ornith-1.5 acts as the agent: it reasons, emits exactly one tool call per
step, observes the tool output, and continues up to MAX_STEPS steps. The
verifier observes at *step boundaries* — the natural checkpoints in an
agentic trajectory.

Tools (all sandboxed to a per-run temp workspace):
  shell : run a bash command (timeout, stdout/stderr/exit captured)
  read  : read a file from the workspace
  write : write a file in the workspace
  test  : run the task's objective grader (external test_cmd)

Each run returns a structured trajectory: every prompt, raw model text,
tool call, raw tool output, step boundary, verdict, rollback events,
and the final verdict from the objective grader.
"""

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

GEN_URL_DEFAULT = "http://127.0.0.1:8012/v1"

GEN_TIMEOUT = 120
TOOL_TIMEOUT = 30
MAX_FILE_BYTES = 64_000

TOOL_DESCRIPTIONS = """You are an autonomous software agent working in a sandboxed task directory.
Each step: write ONE short reasoning sentence, then emit EXACTLY one tool call, then STOP.

Tool call format (function call syntax):
<tool_call>
<function=shell>
<parameter=command>ls -la</parameter>
</function>
</tool_call>

Available tools:
- shell : run a bash command in the task directory. Returns stdout, stderr, exit_code.
          <parameter=command>the bash command</parameter>
- read  : read a file. <parameter=path>foo.py</parameter> -> file contents.
- write : create/overwrite a file. <parameter=path>foo.py</parameter> + <parameter=content>...</parameter> -> confirmation.
- test  : run this task's objective test suite and report PASS/FAIL. No parameters.

Rules:
- Only one tool call per step. Never emit a second function block.
- Do not write any explanation after the tool call; the system will return the tool output.
- Solve the objective. When done, call "test". Repeat until "test" reports PASS.
"""


@dataclass
class Step:
    index: int
    temperature: float
    raw_text: str
    tool_call: Optional[Dict[str, Any]]
    tool_result: Dict[str, Any]
    gen_latency_ms: float
    transcript_growth: int = 0
    verdict: Optional[Dict[str, Any]] = None
    rollback_installed: bool = False
    resampled: bool = False
    escalated: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": self.index,
            "temperature": self.temperature,
            "raw_text": self.raw_text,
            "tool_call": self.tool_call,
            "tool_result": self.tool_result,
            "gen_latency_ms": self.gen_latency_ms,
            "verdict": self.verdict,
            "rollback_installed": self.rollback_installed,
            "resampled": self.resampled,
            "escalated": self.escalated,
        }


@dataclass
class TaskResult:
    task_id: str
    config: str
    passed: bool
    num_steps: int
    wall_time: float
    gpu_tokens_in: int = 0
    gpu_tokens_out: int = 0
    steps: List[Step] = field(default_factory=list)
    abort_events: List[Dict[str, Any]] = field(default_factory=list)
    verdicts: Dict[str, int] = field(default_factory=lambda: {"CONTINUE": 0, "SUSPECT": 0, "ABORT": 0})
    rollbacks: int = 0
    transcript_changes: int = 0
    escalations: int = 0
    steps_per_sample_pre_abort: List[int] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "config": self.config,
            "passed": self.passed,
            "num_steps": self.num_steps,
            "wall_time": self.wall_time,
            "gpu_tokens": {"in": self.gpu_tokens_in, "out": self.gpu_tokens_out},
            "verdicts": dict(self.verdicts),
            "rollbacks": self.rollbacks,
            "abort_events": self.abort_events,
            "transcript_changes": self.transcript_changes,
            "escalations": self.escalations,
            "steps_per_sample_pre_abort": self.steps_per_sample_pre_abort,
            "steps": [s.to_json() for s in self.steps],
        }


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = GEN_TIMEOUT) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    return {}


def _url_to_chat_completions(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return endpoint + "/chat/completions"


_TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(\{.*?\})(?=\s*\n|\s*$)", re.DOTALL)
_LOOSE_TOOL_RE = re.compile(r"\{[^{}]*?\"name\"\s*:\s*\"(shell|read|write|test)\"[^{}]*?\}", re.DOTALL)
_NATIVE_FN_RE = re.compile(r"<function=(shell|read|write|test)>")
_NATIVE_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    m = _TOOL_CALL_RE.search(text)
    if m:
        try:
            call = json.loads(m.group(1))
            if isinstance(call, dict) and call.get("name") in ("shell", "read", "write", "test"):
                return {"name": call["name"], "arguments": call.get("arguments") or {}}
        except Exception:
            pass
    m = _LOOSE_TOOL_RE.search(text)
    if m:
        try:
            call = json.loads(m.group(0))
            if isinstance(call, dict) and call.get("name") in ("shell", "read", "write", "test"):
                return {"name": call["name"], "arguments": call.get("arguments") or {}}
        except Exception:
            pass
    fm = _NATIVE_FN_RE.search(text)
    if fm:
        name = fm.group(1)
        args = {}
        for pm in _NATIVE_PARAM_RE.finditer(text):
            key = pm.group(1).strip()
            args[key] = pm.group(2).strip()
        if name == "test":
            return {"name": "test", "arguments": {}}
        if name == "shell" and "command" not in args:
            args = {"command": args.get("path") or args.get("command") or ""}
        if name in ("shell", "read", "write"):
            return {"name": name, "arguments": args}
    return None


def _safe_rel(workspace: str, path: str) -> str:
    p = os.path.abspath(os.path.join(workspace, path))
    ws = os.path.abspath(workspace)
    if not (p == ws or p.startswith(ws + os.sep)):
        raise ValueError(f"path escapes workspace: {path!r}")
    return p


def run_tool(workspace: str, call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments") or {}
    t0 = time.time()
    result = {"name": name, "arguments": args}
    try:
        if name == "shell":
            proc = subprocess.run(
                ["bash", "-lc", str(args.get("command", ""))],
                cwd=workspace, capture_output=True, text=True, timeout=TOOL_TIMEOUT,
            )
            result.update(stdout=proc.stdout[-8000:], stderr=proc.stderr[-4000:],
                          exit_code=proc.returncode)
        elif name == "read":
            target = _safe_rel(workspace, str(args.get("path", "")))
            with open(target, "rb") as f:
                content = f.read(MAX_FILE_BYTES)
            result.update(stdout=content.decode("utf-8", errors="replace"),
                          stderr="", exit_code=0)
        elif name == "write":
            target = _safe_rel(workspace, str(args.get("path", "")))
            d = os.path.dirname(target)
            os.makedirs(d, exist_ok=True) if d else None
            with open(target, "w") as f:
                f.write(str(args.get("content", "")))
            result.update(stdout=f"OK wrote {len(str(args.get('content','')))} chars",
                          stderr="", exit_code=0)
        else:
            result.update(stdout="", stderr=f"unknown tool {name!r}", exit_code=-2)
    except subprocess.TimeoutExpired:
        result.update(stdout="", stderr=f"timed out after {TOOL_TIMEOUT}s", exit_code=-1)
    except Exception as e:
        result.update(stdout="", stderr=f"{type(e).__name__}: {e}", exit_code=-2)
    result["tool_time_ms"] = (time.time() - t0) * 1000
    return result


def setup_workspace(task: Dict[str, Any], run_id: str = "") -> str:
    root = tempfile.mkdtemp(prefix=f"setd_{task['id']}_{run_id}_".replace("/", "_"))
    for spec in task.get("files", []):
        name, content = spec[0], spec[1]
        mode = spec[2] if len(spec) > 2 else None
        target = _safe_rel(root, name)
        d = os.path.dirname(target)
        os.makedirs(d, exist_ok=True) if d else None
        with open(target, "w") as f:
            f.write(content)
        if mode is not None:
            os.chmod(target, mode)
    return root


def grade(workspace: str, task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            ["bash", "-lc", task["test_cmd"]],
            cwd=workspace, capture_output=True, text=True, timeout=60,
        )
        passed = proc.returncode == 0 and "SET_D_FAIL" not in proc.stdout
        return {"passed": passed, "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-2000:], "exit_code": proc.returncode}
    except Exception as e:
        return {"passed": False, "stdout": "", "stderr": str(e), "exit_code": -3}


def shape_initial_prompt(task: Dict[str, Any]) -> str:
    files = "\n".join(f"- {name}" for name, _ in task.get("files", []))
    return (f"TASK OBJECTIVE:\n{task['instruction']}\n\n"
            f"Your current working directory IS the task directory (a fresh temp dir).\n"
            f"All tools (shell/read/write/test) operate inside it by default.\n\n"
            f"Starter files present:\n{files}\n\n"
            f"Call the \"test\" tool to check your work. \"test\" runs the objective "
            f"grader (including hidden cases) and reports PASS/FAIL.")


def shape_followup_message(tool_result: Dict[str, Any], grade_out: Optional[Dict[str, Any]] = None) -> str:
    parts = [
        f"TOOL_OUTPUT (tool={tool_result.get('name')}):",
        f"exit_code: {tool_result.get('exit_code')}",
    ]
    if tool_result.get("stdout"):
        parts.append(f"stdout:\n{tool_result['stdout']}")
    if tool_result.get("stderr"):
        parts.append(f"stderr:\n{tool_result['stderr']}")
    if grade_out:
        parts.append("TEST VERDICT: " + ("PASS" if grade_out["passed"] else "FAIL"))
        if grade_out["stdout"]:
            parts.append(f"test stdout:\n{grade_out['stdout']}")
    return "\n".join(parts)


class AgentLoop:
    """One agentic stream with step-boundary checkpoint verification.

    mode:
      "shadow" -> log verdicts, never intervene
      "active" -> SUSPECT may escalate; ABORT triggers rollback/resample
                  (max_rollbacks, temperature +0.2 per rollback)
    """

    def __init__(self, gen_endpoint: str, verifier, verifier_prompt,
                 model_id: str = None, escalator=None, mode: str = "shadow",
                 max_steps: int = 15, max_rollbacks: int = 2,
                 base_temperature: float = 0.3):
        self.chat_url = _url_to_chat_completions(gen_endpoint)
        self.model_id = model_id
        self.verifier = verifier
        self.verifier_prompt = verifier_prompt
        self.escalator = escalator
        self.mode = mode
        self.max_steps = max_steps
        self.max_rollbacks = max_rollbacks
        self.base_temperature = base_temperature

    def _generate(self, messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
        payload = {
            "messages": messages,
            "max_tokens": 420,
            "temperature": temperature,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\nTOOL_OUTPUT", "\nSystem:"],
        }
        if self.model_id:
            payload["model"] = self.model_id
        t0 = time.time()
        body = _http_post_json(self.chat_url, payload)
        dt = time.time() - t0
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        return {
            "text": msg.get("content") or "",
            "latency_ms": dt * 1000,
            "tokens_in": (body.get("usage") or {}).get("prompt_tokens") or 0,
            "tokens_out": (body.get("usage") or {}).get("completion_tokens") or 0,
        }

    @staticmethod
    def _trajectory_tail(steps: List[Step]) -> str:
        parts = []
        for s in steps[-3:]:
            parts.append(f"[step {s.index}] tool={ (s.tool_call or {}).get('name') } exit={ (s.tool_result or {}).get('exit_code') }")
            out = (s.tool_result.get('stdout') or '')[-600:]
            if s.tool_result.get('stderr'):
                out += "\nstderr: " + s.tool_result['stderr'][-600:]
            parts.append(out.strip())
        return "\n".join(parts)

    def _verify_step(self, step: Step, steps: List[Step], task) -> Dict[str, Any]:
        tail = self._trajectory_tail(steps)
        prompt = self.verifier_prompt.format(
            task=task["instruction"],
            trajectory=tail,
        )
        res = self.verifier.evaluate_checkpoint(
            task_prompt="AGENT TASK",
            trajectory=prompt,
            prompt_template="{task}\n\n{trajectory}",
            k_samples=3,
            temperature=0.7,
        )
        res["checkpoint_index"] = step.index
        res["trigger"] = "step_boundary"
        return res

    def run(self, task: Dict[str, Any], run_id: str = "0") -> Dict[str, Any]:
        workspace = setup_workspace(task, run_id)
        temperature = self.base_temperature
        messages = [{"role": "system", "content": TOOL_DESCRIPTIONS}]
        messages.append({"role": "user", "content": shape_initial_prompt(task)})
        snapshot_len = None

        result = TaskResult(task_id=task["id"], config=self.mode,
                            passed=False, num_steps=0, wall_time=0.0)
        t_wall = time.time()

        steps: List[Step] = []
        last_continue_step: int = 0
        i = 0
        while i < self.max_steps:
            i += 1
            gen = self._generate(messages, temperature)
            raw_text = gen["text"]
            call = parse_tool_call(raw_text)

            if call is None:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content":
                    "I did not detect a valid tool call. Emit exactly one function call in the tool_call format on its own line now."})
                result.transcript_changes += 1
                step = Step(index=i, temperature=temperature, raw_text=raw_text,
                            tool_call=None, tool_result={}, gen_latency_ms=gen["latency_ms"])
                steps.append(step)
                result.steps = steps
                continue

            # execute the tool
            if call["name"] == "test":
                tool = {"name": "test", "arguments": {}}
                g = grade(workspace, task)
                step = Step(index=i, temperature=temperature, raw_text=raw_text,
                            tool_call=call, tool_result=tool, gen_latency_ms=gen["latency_ms"])
                steps.append(step)
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": shape_followup_message(tool, g)})

                if g["passed"] or i >= self.max_steps:
                    result.passed = g["passed"]
                    break

                step.verdict = self._verify_step(step, steps, task)
                v = step.verdict.get("verdict", "CONTINUE")
                result.verdicts[v] += 1
                continue

            tool = run_tool(workspace, call)
            step = Step(index=i, temperature=temperature, raw_text=raw_text,
                        tool_call=call, tool_result=tool, gen_latency_ms=gen["latency_ms"])
            steps.append(step)
            result.steps = steps

            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": shape_followup_message(tool)})

            # step-boundary verification
            verdict = self._verify_step(step, steps, task)
            step.verdict = verdict
            v = verdict.get("verdict", "CONTINUE")
            result.verdicts[v] = result.verdicts.get(v, 0) + 1

            escalated = False
            if self.escalator and v == "SUSPECT":
                score = self.escalator.score_step(task, steps)
                step.escalated = True
                result.escalations += 1
                if score <= self.escalator.threshold:
                    result.verdicts["SUSPECT"] -= 1
                    result.verdicts["ABORT"] += 1
                    verdict["escalated_abort"] = True
                    escalated = True

            if escalated or v == "ABORT":
                # never abort a passing terminal test
                if call["name"] == "test" and tool.get("exit_code") == 0:
                    result.abort_events.append({
                        "step": i, "reason": "false_abort_on_passing_test",
                        "verdict": verdict,
                    })
                    continue

                # decide rollback
                if self.mode == "active" and result.rollbacks < self.max_rollbacks and last_continue_step > 0:
                    result.rollbacks += 1
                    new_temp = min(temperature + 0.2, 1.1)
                    event = {
                        "rollback_step": i,
                        "rollback_to_step": last_continue_step,
                        "abort_reason": verdict.get("evidence", ""),
                        "temperature_before": temperature,
                        "temperature_after": new_temp,
                        "escalated": escalated,
                    }
                    result.abort_events.append(event)
                    # restore transcript to recorded checkpoint
                    if snapshot_len:
                        messages = messages[:snapshot_len]
                    temperature = new_temp
                    result.steps[i - 1].resampled = True
                    i = last_continue_step  # jump back; next loop adds +1
                    continue

                result.abort_events.append({
                    "step": i, "reason": "no_rollback_available",
                    "abort_reason": verdict.get("evidence", ""),
                })
                continue

            if v == "CONTINUE":
                last_continue_step = i
                snapshot_len = len(messages)
                continue

            # SUSPECT (not escalated to ABORT): keep going but block rollback restore target
            if snapshot_len is None:
                snapshot_len = len(messages)

        if not result.passed:
            g = grade(workspace, task)
            result.passed = g["passed"]

        result.num_steps = len(steps)
        result.wall_time = time.time() - t_wall
        return {
            "task_id": task["id"],
            "config": self.mode,
            "passed": result.passed,
            "rollbacks": result.rollbacks,
            "escalations": result.escalations,
            "verdicts": dict(result.verdicts),
            "num_steps": result.num_steps,
            "wall_time": result.wall_time,
            "abort_events": result.abort_events,
            "transcript_changes": result.transcript_changes,
            "steps": [s.to_json() for s in steps],
            "workspace": workspace,
        }


__all__ = [
    "AgentLoop", "TaskResult", "Step", "run_tool", "parse_tool_call",
    "setup_workspace", "grade", "shape_initial_prompt", "shape_followup_message",
    "TOOL_DESCRIPTIONS", "GEN_URL_DEFAULT",
]
