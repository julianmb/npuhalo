#!/usr/bin/env python3
"""
run_narrow_eval.py — Focused evaluation on the 8 failing Set D tasks across 5 seeds.

Configs tested:
  1. baseline      -- standard unverified loop
  2. parser_guard  -- deterministic tool-call schema validator + rollback & format hint
  3. active_esc    -- 3-tier:
                      Tier 1: Parser schema validation (instant, free)
                      Tier 2: LFM2.5-tk SUSPECT triage on NPU (:8001)
                      Tier 3: Qwen3.5-2B logprob scoring on iGPU (:8013) -> ABORT authority

Evaluates:
  - Pass rate & rollback conversion rate on the 8 hard tasks
  - Token & latency economics
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))

from agent_loop import (  # noqa: E402
    Step, TaskResult, grade, parse_tool_call, run_tool,
    setup_workspace, shape_followup_message, shape_initial_prompt,
    TOOL_DESCRIPTIONS, _http_post_json, _url_to_chat_completions
)
from verifier_client import NPUVerifierClient  # noqa: E402
from escalator import QwenEscalator  # noqa: E402

GEN = "http://127.0.0.1:8012/v1"
NPU = "http://127.0.0.1:8001/v1"
ESC = "http://127.0.0.1:8013/v1"
DATA = os.path.join(REPO, "verifier", "data", "set_d_agentic.jsonl")
PROMPT_FILE = os.path.join(REPO, "verifier", "prompts", "final_verdict_prompt.txt")
OUT_DIR = os.path.join(REPO, "verifier", "results", "raw", "narrow")
SUMMARY_FILE = os.path.join(REPO, "verifier", "results", "narrow_summary.json")

TARGET_TASK_IDS = ["D02", "D03", "D04", "D06", "D18", "D21", "D25", "D28"]


def load_target_tasks() -> List[Dict[str, Any]]:
    all_tasks = [json.loads(line) for line in open(DATA)]
    return [t for t in all_tasks if t["id"] in TARGET_TASK_IDS]


class RobustAgentLoop:
    """Robust agent loop supporting baseline, parser_guard, and 3-tier active_esc."""

    def __init__(self, config: str, verifier=None, escalator=None,
                 verifier_prompt: str = "", max_steps: int = 15,
                 max_rollbacks: int = 2, base_temperature: float = 0.3):
        self.config = config  # "baseline", "parser_guard", "active_esc"
        self.chat_url = _url_to_chat_completions(GEN)
        self.verifier = verifier
        self.escalator = escalator
        self.verifier_prompt = verifier_prompt
        self.max_steps = max_steps
        self.max_rollbacks = max_rollbacks
        self.base_temperature = base_temperature

    def _generate(self, messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
        payload = {
            "messages": messages,
            "max_tokens": 1024,  # enough for reasoning + tool call
            "temperature": temperature,
            "top_p": 0.95,
            "stream": False,
        }
        t0 = time.time()
        body = _http_post_json(self.chat_url, payload, timeout=120)
        dt = time.time() - t0
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        raw = content if content else reasoning
        return {
            "text": raw,
            "latency_ms": dt * 1000,
            "tokens_in": (body.get("usage") or {}).get("prompt_tokens") or 0,
            "tokens_out": (body.get("usage") or {}).get("completion_tokens") or 0,
        }

    def _verify_step_npu(self, steps: List[Step], task: Dict[str, Any]) -> Dict[str, Any]:
        parts = []
        for s in steps[-3:]:
            parts.append(f"[step {s.index}] tool={(s.tool_call or {}).get('name')} exit={(s.tool_result or {}).get('exit_code')}")
            out = (s.tool_result.get('stderr') or s.tool_result.get('stdout') or '')[-500:]
            parts.append(out.strip())
        tail = "\n".join(parts)
        prompt = self.verifier_prompt.format(task=task["instruction"], trajectory=tail)
        res = self.verifier.evaluate_checkpoint(
            task_prompt="AGENT TASK",
            trajectory=prompt,
            prompt_template="{task}\n\n{trajectory}",
            k_samples=3,
            temperature=0.7,
        )
        return res

    def run(self, task: Dict[str, Any], seed: int = 1) -> Dict[str, Any]:
        workspace = setup_workspace(task, f"{self.config}_{seed}")
        temperature = self.base_temperature
        messages = [{"role": "system", "content": TOOL_DESCRIPTIONS}]
        messages.append({"role": "user", "content": shape_initial_prompt(task)})

        result = TaskResult(task_id=task["id"], config=self.config,
                            passed=False, num_steps=0, wall_time=0.0)
        t_wall = time.time()

        steps: List[Step] = []
        snapshot_messages: List[Dict[str, str]] = list(messages)
        last_continue_step = 0
        rollbacks = 0
        escalations = 0

        i = 0
        while i < self.max_steps:
            i += 1
            gen = self._generate(messages, temperature)
            result.gpu_tokens_in += gen["tokens_in"]
            result.gpu_tokens_out += gen["tokens_out"]

            raw_text = gen["text"]
            call = parse_tool_call(raw_text)

            # --- TIER 1: Parser Schema Validation ---
            if call is None:
                if self.config in ("parser_guard", "active_esc") and rollbacks < self.max_rollbacks and last_continue_step > 0:
                    rollbacks += 1
                    temperature = min(temperature + 0.2, 1.1)
                    result.abort_events.append({
                        "step": i,
                        "tier": "tier1_parser_guard",
                        "reason": "malformed_or_missing_tool_call",
                        "rollback_to": last_continue_step,
                        "new_temperature": temperature,
                    })
                    # Roll back to last valid checkpoint and inject format correction hint
                    messages = list(snapshot_messages)
                    messages.append({
                        "role": "user",
                        "content": "FORMAT ERROR: Your previous response did not contain a valid <tool_call> block. "
                                   "Emit exactly ONE <tool_call><function=NAME><parameter=KEY>VAL</parameter></function></tool_call> now."
                    })
                    i = last_continue_step
                    continue
                else:
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({"role": "user", "content": "Emit a valid <tool_call> now."})
                    step = Step(index=i, temperature=temperature, raw_text=raw_text,
                                tool_call=None, tool_result={}, gen_latency_ms=gen["latency_ms"])
                    steps.append(step)
                    continue

            # Execute Tool
            if call["name"] == "test":
                tool = {"name": "test", "arguments": {}}
                g = grade(workspace, task)
                step = Step(index=i, temperature=temperature, raw_text=raw_text,
                            tool_call=call, tool_result=tool, gen_latency_ms=gen["latency_ms"])
                steps.append(step)
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": shape_followup_message(tool, g)})

                if g["passed"]:
                    result.passed = True
                    break
                # If test failed, update snapshot
                snapshot_messages = list(messages)
                last_continue_step = i
                continue

            tool = run_tool(workspace, call)
            step = Step(index=i, temperature=temperature, raw_text=raw_text,
                        tool_call=call, tool_result=tool, gen_latency_ms=gen["latency_ms"])
            steps.append(step)
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": shape_followup_message(tool)})

            # --- TIER 2 & 3: Neural Verifier (active_esc only) ---
            if self.config == "active_esc":
                v_res = self._verify_step_npu(steps, task)
                step.verdict = v_res
                v = v_res.get("verdict", "CONTINUE")
                result.verdicts[v] = result.verdicts.get(v, 0) + 1

                abort_decision = False
                escalation_record = None

                # Tier 2: LFM SUSPECT triggers Tier 3 Qwen Escalation
                if v == "SUSPECT" and self.escalator:
                    escalations += 1
                    step.escalated = True
                    escalation_record = self.escalator.score_step(task, steps)
                    if self.escalator.is_jeopardy(escalation_record):
                        abort_decision = True
                        v_res["escalated_abort"] = True
                elif v == "ABORT":
                    abort_decision = True

                if abort_decision and rollbacks < self.max_rollbacks and last_continue_step > 0:
                    rollbacks += 1
                    temperature = min(temperature + 0.2, 1.1)
                    result.abort_events.append({
                        "step": i,
                        "tier": "tier3_qwen_escalator" if step.escalated else "tier2_lfm_npu",
                        "verdict": v,
                        "escalation_score": (escalation_record or {}).get("r"),
                        "rollback_to": last_continue_step,
                        "new_temperature": temperature,
                    })
                    # Revert transcript
                    messages = list(snapshot_messages)
                    i = last_continue_step
                    continue

                if v == "CONTINUE":
                    last_continue_step = i
                    snapshot_messages = list(messages)
            else:
                # baseline and parser_guard update snapshot on every valid tool execution
                last_continue_step = i
                snapshot_messages = list(messages)

        if not result.passed:
            g = grade(workspace, task)
            result.passed = g["passed"]

        result.num_steps = len(steps)
        result.wall_time = time.time() - t_wall
        result.rollbacks = rollbacks
        result.escalations = escalations
        result.steps = steps

        # Cleanup workspace
        try:
            import shutil
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass

        return result.to_json()


def run_sweep(configs: List[str], seeds: List[int], resume: bool = True) -> Dict[str, Any]:
    tasks = load_target_tasks()
    print(f"[*] Loaded {len(tasks)} target tasks: {[t['id'] for t in tasks]}")
    os.makedirs(OUT_DIR, exist_ok=True)

    verifier = NPUVerifierClient(NPU)
    verifier.evaluate_checkpoint("warm", "warm", "{task} {trajectory}", k_samples=3, temperature=0.7)
    escalator = QwenEscalator(ESC, threshold=0.50)
    prompt = open(PROMPT_FILE).read()

    summary = {}
    for cfg in configs:
        print(f"\n==================== CONFIG: {cfg} ====================")
        cfg_results = []
        t0_cfg = time.time()
        for task in tasks:
            for seed in seeds:
                out_path = os.path.join(OUT_DIR, f"{task['id']}_{cfg}_s{seed}.json")
                if resume and os.path.exists(out_path):
                    try:
                        rec = json.load(open(out_path))
                        cfg_results.append(rec)
                        print(f"[{cfg}] {task['id']} (s{seed}): RESUMED passed={rec['passed']} steps={rec['num_steps']}")
                        continue
                    except Exception:
                        pass  # corrupt file: re-run
                runner = RobustAgentLoop(
                    config=cfg,
                    verifier=verifier if cfg == "active_esc" else None,
                    escalator=escalator if cfg == "active_esc" else None,
                    verifier_prompt=prompt,
                    max_steps=15,
                    max_rollbacks=2,
                    base_temperature=0.3,
                )
                rec = runner.run(task, seed=seed)
                rec["seed"] = seed
                rec["task_category"] = task["category"]
                cfg_results.append(rec)
                print(f"[{cfg}] {task['id']} (s{seed}): passed={rec['passed']} steps={rec['num_steps']} "
                      f"rollbacks={rec['rollbacks']} escalations={rec.get('escalations', 0)} wall={rec['wall_time']:.1f}s")
                # Save raw
                with open(out_path, "w") as f:
                    json.dump(rec, f, indent=2)

        # Aggregate for config
        passes = sum(1 for r in cfg_results if r["passed"])
        total = len(cfg_results)
        tot_rollbacks = sum(r["rollbacks"] for r in cfg_results)
        rb_converted = sum(1 for r in cfg_results if r["rollbacks"] > 0 and r["passed"])
        rb_total_tasks = sum(1 for r in cfg_results if r["rollbacks"] > 0)
        gpu_toks = sum(r["gpu_tokens"]["in"] + r["gpu_tokens"]["out"] for r in cfg_results)
        wall_tot = time.time() - t0_cfg

        summary[cfg] = {
            "tasks": len(tasks),
            "runs": total,
            "passed": passes,
            "pass_rate": round(passes / total, 3),
            "total_rollbacks": tot_rollbacks,
            "tasks_with_rollbacks": rb_total_tasks,
            "rollback_converted": rb_converted,
            "rollback_conversion_rate": round(rb_converted / rb_total_tasks, 3) if rb_total_tasks > 0 else 0.0,
            "total_gpu_tokens": gpu_toks,
            "gpu_tokens_per_solve": round(gpu_toks / passes, 1) if passes > 0 else None,
            "wall_time_s": round(wall_tot, 1),
            "wall_per_run_s": round(wall_tot / total, 1),
        }
        print(f"--- Summary for {cfg}: {passes}/{total} ({summary[cfg]['pass_rate']*100:.1f}%) | "
              f"Rollback Conversion: {rb_converted}/{rb_total_tasks} ({summary[cfg]['rollback_conversion_rate']*100:.1f}%) ---")

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Saved complete narrow evaluation summary to {SUMMARY_FILE}")
    return summary


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    configs_to_run = args if args else ["baseline", "parser_guard", "active_esc"]
    seeds_to_run = [1, 2, 3, 4, 5]
    res = run_sweep(configs_to_run, seeds_to_run, resume=not force)
    print("\n" + json.dumps(res, indent=2))
