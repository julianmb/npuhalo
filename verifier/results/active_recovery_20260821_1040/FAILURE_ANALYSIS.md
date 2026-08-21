# Failure Analysis — Narrow Active Sweep Death at 59/120

**Experiment dir:** `verifier/results/active_recovery_20260821_*/`
**Audited run:** `python3 verifier/scripts/run_narrow_eval.py baseline parser_guard active_esc`
**Log evidence:** `/tmp/opencode/narrow_full.log` (client), `/tmp/opencode/ornith_server.log` (server)
**Date:** 2026-08-21

## Timeline of the fatal run

| Event | Evidence |
|---|---|
| baseline arm: 40/40 runs completed | client log `[baseline] ...` lines, `raw/narrow/*_baseline_*.json` |
| parser_guard arm: D02–D06 s1–s4 completed (24 runs) | client log, raw files |
| **parser_guard D06 (s5): first `_generate()` call → HTTP 400 → unhandled `HTTPError` → process exit** | traceback at `run_narrow_eval.py:126 → :76 → agent_loop.py:130` |
| active_esc arm: **0 runs attempted** | no files |

## Answers

### 1. Which process failed first?

**Neither process "failed" in the crash sense at the fatal moment — the server was alive and deliberately rejected a request.** The Ornith server had crashed 4 separate times earlier in the session (`ggml_abort` / SIGABRT in `common_context_seq_rm`, exit code 134; supervisor restarted it each time — see `ornith_server.log` lines 228/461/1163/1421), but those crashes produced connection errors, not the fatal error. The run died because the **client harness** treated an expected-condition HTTP response as unrecoverable.

### 2. What exact HTTP 400 body was returned?

The original request's body was not captured (`agent_loop._http_post_json` does not read `HTTPError.body`). However, the condition was **reproduced minimally** against the live endpoint:

```json
{"error": {"code": 400,
           "message": "request (17510 tokens) exceeds the available context size (16384 tokens), try increasing it",
           "type": "exceed_context_size_error",
           "n_prompt_tokens": 17510, "n_ctx": 16384}}
```

Reproduction: POST `/v1/chat/completions` with a 140,000-char user message → HTTP 400 `exceed_context_size_error`; 100/40K/80K-char controls → HTTP 200.

### 3. Was the endpoint healthy before the crash?

Yes. The immediately preceding requests succeeded (`[parser_guard] D06 (s4)` completed 184 s before the failure), and the minimal reproduction confirms the endpoint answers normally before/after. Health was never observed down at the fatal instant.

### 4. Does a restart restore equivalent serving behavior?

Yes for crashes: the supervisor restarted the server after each of the 4 SIGABRT events and serving continued (subsequent successful runs prove functional equivalence; weights are memory-mapped/page-cached, restart ≈ 10–25 s to healthy). For the fatal 400, no restart is involved — the server refuses over-context requests deterministically under `--no-context-shift`.

### 5. Correlation analysis

The failure correlates with **conversation growth, not a specific task or uptime**:

- Context limit is 16,384 tokens; long agentic trajectories (15 steps × tool outputs) can exceed it.
- The death occurred in the **parser_guard arm**, whose rollback path injects format hints into the message history — adding tokens and making overflow *more* likely than baseline. Baseline completed all 40 runs; parser_guard died on run 25.
- Earlier session deaths with the same HTTP-400 signature occurred in different arms/tasks, ruling out a single poisoned task.
- No correlation with concurrency (single stream), NPU state, or time-of-day. Server uptime is anti-correlated if anything (fresh restarts preceded some successes).

### 6. Minimal reproduction?

Yes — see §2. A single self-contained POST reproduces the exact error class deterministically. The server-side SIGABRT crash has a separate known trigger (rapid varying-length prefills, documented in `docs/REPORT.md` §7).

## Consequences for experiment design

1. The harness must classify outcomes explicitly: `COMPLETED`, `INFRA_FAILURE` (transport/crash), and `CONTEXT_OVERFLOW` (harness/server context limit — an evaluation-infrastructure limitation, not a model or verifier outcome).
2. `CONTEXT_OVERFLOW` tasks must not be silently dropped nor counted as model failures; overflow rates must be reported **per mode** — asymmetric overflow between arms is itself evidence about rollback-induced context inflation.
3. All failed requests must capture HTTP status + response body.
4. The supervisor handles crash recovery; the harness must wait for health and retry once.
