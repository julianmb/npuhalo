#!/usr/bin/env python3
"""
run_parser_guard_eval.py — Live evaluation of the deterministic tool-call parser.

Modes:
  shadow  — stream generation through the incremental parser, LOG detections,
            take no action. Post-hoc full-text parse must agree 100%.
  active  — on UNRECOVERABLE detection mid-stream: stop the generation
            (client disconnect stops server decoding), re-prompt ONCE with the
            malformed fragment + parser error position as feedback; a second
            failure surfaces the raw text to the normal loop (no rollback, no
            checkpoint restore — this is deliberately not the verifier's
            budget-death pattern).

Both modes reuse RobustAgentLoop (identical prompts/sampling/tools) with only
_generate replaced by a streaming variant. Baseline comparison uses the
recorded baseline runs from active_recovery_20260821_1040 (seeds 1-2).
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "verifier", "src"))
sys.path.insert(0, HERE)

import run_narrow_eval as base  # noqa: E402
from toolcall_parser import IncrementalToolCallParser  # noqa: E402
from grammar_compiler import compile_tool_call_grammar  # noqa: E402

GEN_CHAT = os.environ.get("PG_GEN_CHAT", "http://127.0.0.1:8012/v1/chat/completions")
CHUNK_CHARS = 4


class ParserAgentLoop(base.RobustAgentLoop):
    """RobustAgentLoop with streaming _generate + parser guard + optional grammar."""

    def __init__(self, *a, mode_parser="shadow", grammar=None, **kw):
        super().__init__(*a, **kw)
        self.mode_parser = mode_parser
        self.grammar = grammar
        self.parser_events = []

    def _stream_generate(self, messages, temperature, abort_on_unrecoverable):
        """Stream one completion; feed deltas through the parser.

        Returns dict shaped like RobustAgentLoop._generate output plus
        parser_event metadata."""
        payload = {"messages": messages, "max_tokens": 1024, "temperature": temperature,
                   "top_p": 0.95, "stream": True,
                   "stream_options": {"include_usage": True}}
        if self.grammar:
            payload["grammar"] = self.grammar
        req = urllib.request.Request(GEN_CHAT, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        parser = IncrementalToolCallParser()
        text_parts = []
        tokens_in = tokens_out = 0
        t0 = time.time()
        ttft = None
        event = None
        aborted = False
        try:
            with urllib.request.urlopen(req, timeout=150) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except Exception:
                        continue
                    usage = chunk.get("usage")
                    if usage:
                        tokens_in = usage.get("prompt_tokens", tokens_in)
                        tokens_out = usage.get("completion_tokens", tokens_out)
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    # Under GBNF grammar the chat template routes output through
                    # reasoning_content (no closing think tag); accept both fields.
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if not piece:
                        continue
                    if ttft is None:
                        ttft = (time.time() - t0) * 1000
                    text_parts.append(piece)
                    if parser.finished:
                        continue
                    st = parser.feed(piece)
                    if st.value == "UNRECOVERABLE":
                        event = {
                            "detected_live": True,
                            "char_index": len("".join(text_parts)),
                            "reason": parser.error["reason"],
                            "fragment_head": "".join(text_parts)[-200:],
                        }
                        if abort_on_unrecoverable:
                            aborted = True
                            break  # closing the connection stops server-side decode
        except Exception as e:
            return {"text": "", "latency_ms": (time.time() - t0) * 1000,
                    "tokens_in": tokens_in, "tokens_out": tokens_out,
                    "stream_error": str(e)[:200]}
        latency = (time.time() - t0) * 1000
        return {
            "text": "".join(text_parts),
            "latency_ms": latency, "ttft_ms": ttft,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "parser_event": event, "aborted_early": aborted,
        }

    def _generate(self, messages, temperature):
        attempts = []
        for attempt in range(2):  # cap: 1 original + 1 re-prompt per tool call
            gen = self._stream_generate(messages, temperature,
                                        abort_on_unrecoverable=(self.mode_parser == "active"))
            ev = gen.get("parser_event")
            attempts.append({
                "attempt": attempt + 1, "detected": bool(ev),
                "reason": (ev or {}).get("reason"),
                "char_index": (ev or {}).get("char_index"),
                "aborted_early": gen.get("aborted_early", False),
                "tokens_out": gen["tokens_out"], "latency_ms": round(gen["latency_ms"]),
            })
            full_text = gen["text"]
            self.parser_events.append({"phase": self.mode_parser, **attempts[-1],
                                       "text_head": full_text[:200],
                                       "full_text": full_text,
                                       "text_len": len(full_text),
                                       "has_parsed_call": base.parse_tool_call(full_text) is not None})
            if ev is None or attempt == 1:
                break
            # structured re-prompt: malformed fragment + error position as feedback
            messages = list(messages)
            messages.append({"role": "assistant", "content": gen["text"]})
            messages.append({
                "role": "user",
                "content": f"FORMAT ERROR: your <tool_call> was structurally invalid "
                           f"({ev['reason']}, at character {ev['char_index']}). "
                           f"Emit exactly ONE complete, valid <tool_call> block now and STOP.",
            })
        text = gen["text"]
        return {"text": text, "latency_ms": gen["latency_ms"],
                "tokens_in": gen["tokens_in"], "tokens_out": gen["tokens_out"],
                "parser_attempts": attempts}


def posthoc_agree(text: str) -> dict:
    """Post-hoc parse of the full text — must match live detection."""
    p = IncrementalToolCallParser()
    p.feed(text)
    return {"unrecoverable": p.result()["state"] == "UNRECOVERABLE",
            "error": p.error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shadow", "active"], required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--tasks", nargs="+",
                    default=["D01", "D02", "D03", "D04", "D18", "D21", "D25", "D28"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--grammar", action="store_true",
                    help="force tool-call structure via GBNF (Mode B)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    (exp_dir / "runs").mkdir(parents=True, exist_ok=True)
    events_path = exp_dir / f"events_{args.mode}.jsonl"

    all_tasks = [json.loads(line) for line in open(base.DATA)]
    by_id = {t["id"]: t for t in all_tasks}
    tasks = [by_id[t] for t in args.tasks if t in by_id]

    totals = {"runs": 0, "completed": 0, "live_detections": 0, "posthoc_detections": 0,
              "agreement_mismatches": 0, "reprompts": 0, "false_aborts_on_valid": 0,
              "tokens_out_sum": 0}
    for task in tasks:
        for seed in args.seeds:
            out = exp_dir / "runs" / f"{task['id']}_{args.mode}_s{seed}.json"
            if args.resume and out.exists():
                continue
            grammar = compile_tool_call_grammar() if args.grammar else None
            runner = ParserAgentLoop(config="baseline", mode_parser=args.mode, grammar=grammar,
                                     max_steps=15, max_rollbacks=0, base_temperature=0.3)
            workspace = base.setup_workspace(task, f"pg_{args.mode}_{seed}")
            messages = [{"role": "system", "content": base.TOOL_DESCRIPTIONS},
                        {"role": "user", "content": base.shape_initial_prompt(task)}]
            t0 = time.time()
            steps = 0
            passed = False
            while steps < 15:
                steps += 1
                gen = runner._generate(messages, 0.3)
                totals["tokens_out_sum"] += gen["tokens_out"]
                call = base.parse_tool_call(gen["text"])
                if call is None:
                    messages.append({"role": "assistant", "content": gen["text"]})
                    messages.append({"role": "user", "content": "Emit a valid <tool_call> now."})
                    continue
                if call["name"] == "test":
                    g = base.grade(workspace, task)
                    if g["passed"]:
                        passed = True
                        break
                    # failed test: feed back and keep working (matches RobustAgentLoop)
                    messages.append({"role": "assistant", "content": gen["text"]})
                    messages.append({"role": "user",
                                     "content": base.shape_followup_message(
                                         {"name": "test", "arguments": {}}, g)})
                    continue
                tool = base.run_tool(workspace, call)
                messages.append({"role": "assistant", "content": gen["text"]})
                messages.append({"role": "user", "content": base.shape_followup_message(tool)})
            wall = round(time.time() - t0, 1)

            # Post-hoc verification must parse the SAME full text the live
            # parser saw — parsing text_head[:200] caused the D06 s1 false
            # mismatch (violation at char 4173, head only 200 chars).
            for e in runner.parser_events:
                ph = posthoc_agree(e.get("full_text", ""))
                e["posthoc_unrecoverable"] = ph["unrecoverable"]
                e["agreement"] = (e["posthoc_unrecoverable"] == bool(e.get("detected")))
            live_det = [e for e in runner.parser_events if e.get("detected")]
            mismatch = sum(1 for e in runner.parser_events if not e.get("agreement", True))
            totals["runs"] += 1
            totals["completed"] += 1
            totals["live_detections"] += len(live_det)
            totals["posthoc_detections"] += sum(1 for e in runner.parser_events if e.get("posthoc_unrecoverable"))
            totals["agreement_mismatches"] += mismatch
            totals["reprompts"] += sum(1 for e in runner.parser_events if e.get("attempt") == 2)

            rec = {"task_id": task["id"], "seed": seed, "config": f"parser_{args.mode}{"+g" if args.grammar else ""}",
                   "passed": passed, "steps": steps, "wall_time_s": wall,
                   "parser_events": runner.parser_events,
                   "live_detections": len(live_det),
                   "agreement_mismatch": mismatch,
                   "outcome": "COMPLETED"}
            with open(out, "w") as f:
                json.dump(rec, f, indent=2)
            with open(events_path, "a") as ef:
                for e in runner.parser_events:
                    ef.write(json.dumps({"ts": time.time(), "task": task["id"],
                                         "seed": seed, **e}) + "\n")
            print(f"[{args.mode}] {task['id']} s{seed}: passed={passed} steps={steps} "
                  f"detections={len(live_det)} mismatches={mismatch} wall={wall}s", flush=True)

    print(json.dumps(totals, indent=2))
    with open(exp_dir / f"totals_{args.mode}.json", "w") as f:
        json.dump(totals, f, indent=2)


if __name__ == "__main__":
    main()
