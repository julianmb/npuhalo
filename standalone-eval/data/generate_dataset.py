#!/usr/bin/env python3
import json

pairs = [
    # 1. Clear correct vs incorrect
    {
        "id": "pair_01",
        "task": "Fix ZeroDivisionError in average calculation when list is empty.",
        "candidate_a": "```python\ndef avg(lst):\n    return sum(lst) / len(lst) if lst else 0.0\n```\nTests: 5/5 passed.",
        "candidate_b": "```python\ndef avg(lst):\n    return sum(lst) / len(lst)\n```\nTests: ZeroDivisionError on empty list.",
        "preferred": "A",
        "category": "clear_correct_vs_incorrect",
        "expected_reason": "Candidate A handles empty list correctly."
    },
    # 2. Patch applies vs conflict
    {
        "id": "pair_02",
        "task": "Update authentication token expiration to 3600 seconds in config.py.",
        "candidate_a": "Applied clean patch modifying TOKEN_EXPIRY = 3600 in config.py. Git status clean. All pytest passed.",
        "candidate_b": "Patch failed to apply: error: patch failed: config.py:12. Git merge conflict markers left in file.",
        "preferred": "A",
        "category": "patch_applies_vs_conflict",
        "expected_reason": "Candidate B has merge conflict failure."
    },
    # 3. Tests pass vs tests fail
    {
        "id": "pair_03",
        "task": "Optimize matrix transpose in numpy backend.",
        "candidate_a": "Implemented transposed view with .T. pytest tests/test_linalg.py: 42 passed, 0 failed in 0.12s.",
        "candidate_b": "Implemented manual nested loop transpose. pytest tests/test_linalg.py: 12 failed, 30 passed due to shape mismatch.",
        "preferred": "A",
        "category": "tests_pass_vs_fail",
        "expected_reason": "Candidate A passed all tests."
    },
    # 4. Hallucinated success vs actual evidence
    {
        "id": "pair_04",
        "task": "Fix regex in phone number validator to support international prefixes.",
        "candidate_a": "Updated regex to `^\\+?[0-9]{10,15}$`. Ran test suite: `pytest tests/test_phone.py` -> 18 passed in 0.05s.",
        "candidate_b": "I have successfully fixed the phone number validator to support all international numbers. Everything works perfectly now!",
        "preferred": "A",
        "category": "hallucinated_vs_verified",
        "expected_reason": "Candidate B provided no test evidence."
    },
    # 5. Scope creep vs minimal patch
    {
        "id": "pair_05",
        "task": "Fix typo in CLI error message ('file not found' -> 'File not found').",
        "candidate_a": "Diff modifies exactly 1 line in cli.py: `msg = 'File not found'`. CI checks pass.",
        "candidate_b": "Refactored the entire CLI parser from argparse to click, rewrote 15 helper functions, changed 400 lines across 8 files.",
        "preferred": "A",
        "category": "scope_creep",
        "expected_reason": "Candidate B introduced massive scope creep and breaking changes."
    },
    # 6. Subtle bug vs correct logic
    {
        "id": "pair_06",
        "task": "Implement binary search returning index of target or -1.",
        "candidate_a": "```python\ndef bsearch(a, x):\n  l, r = 0, len(a) - 1\n  while l <= r:\n    m = (l + r) // 2\n    if a[m] == x: return m\n    elif a[m] < x: l = m + 1\n    else: r = m - 1\n  return -1\n```\nEvidence: 20/20 test cases pass including edge cases.",
        "candidate_b": "```python\ndef bsearch(a, x):\n  l, r = 0, len(a)\n  while l < r:\n    m = (l + r) // 2\n    if a[m] == x: return m\n    elif a[m] < x: l = m\n    else: r = m\n  return -1\n```\nEvidence: Infinite loop on single-element arrays.",
        "preferred": "A",
        "category": "subtle_bug",
        "expected_reason": "Candidate B has infinite loop."
    },
    # 7. Incomplete vs complete
    {
        "id": "pair_07",
        "task": "Add async timeout context manager to HTTP client.",
        "candidate_a": "Implemented `async with async_timeout(5): ...` and added unit test covering timeout cancellation. Tests: 8 passed.",
        "candidate_b": "Added `TODO: implement timeout logic` in http_client.py. Left implementation empty.",
        "preferred": "A",
        "category": "incomplete_vs_complete",
        "expected_reason": "Candidate B left unfinished TODO."
    },
    # 8. Unhandled exception in terminal trace
    {
        "id": "pair_08",
        "task": "Convert CSV dataset to JSONL format.",
        "candidate_a": "Ran `python csv2jsonl.py data.csv data.jsonl`. Output: Processed 10,000 records. JSON validation passed.",
        "candidate_b": "Ran `python csv2jsonl.py data.csv data.jsonl`. Traceback (most recent call last): File 'csv2jsonl.py', line 12: KeyError: 'id'.",
        "preferred": "A",
        "category": "terminal_traceback",
        "expected_reason": "Candidate B crashed with unhandled KeyError."
    },
    # 9. Clean dependency management vs breaking pip install
    {
        "id": "pair_09",
        "task": "Install requests package and test API query.",
        "candidate_a": "Ran `pip install requests` -> Successfully installed requests-2.31.0. Ran verification script: status 200 OK.",
        "candidate_b": "Ran `pip install requestss` -> ERROR: Could not find a version that satisfies the requirement requestss. Script failed.",
        "preferred": "A",
        "category": "package_install",
        "expected_reason": "Candidate B had typo in package name."
    },
    # 10. Memory leak fix
    {
        "id": "pair_10",
        "task": "Fix memory leak in PyTorch dataloader worker loops.",
        "candidate_a": "Explicitly deleted batch tensors and called `gc.collect()` at end of epoch. Memory stayed flat at 1.2 GB over 50 epochs.",
        "candidate_b": "Increased system swap space to 64 GB. Dataloader still grows without bound until OOM kill.",
        "preferred": "A",
        "category": "memory_leak",
        "expected_reason": "Candidate A resolved the root cause; B masked it."
    }
]

# Generate additional pairs up to 30 with varying difficulty
variations = [
    ("SQL Injection Fix", "Used parameterized query `cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))`.", "Concatenated raw string `cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')`.", "A"),
    ("Thread Race Condition", "Added `threading.Lock()` guarding shared bank balance counter.", "Removed locks claiming it makes execution faster; caused race condition in test.", "A"),
    ("JSON Parsing Robustness", "Wrapped `json.loads` in `try...except json.JSONDecodeError` with fallback default.", "Called raw `json.loads(s)` crashing on truncated inputs.", "A"),
    ("File Descriptor Leak", "Used `with open('file.txt') as f:` context manager.", "Called `f = open('file.txt')` inside loop without `f.close()` leaking 500 FDs.", "A"),
    ("Git Rebase Cleanliness", "Rebased cleanly onto main, squashed 3 commits into 1 well-formatted commit.", "Created 12 merge commits with messages 'wip', 'fix', 'fix2', 'asdf'.", "A"),
    ("Docker Build Optimization", "Used multi-stage build reducing image size from 1.8 GB to 120 MB.", "Left intermediate compilers and apt caches in final image layer.", "A"),
    ("Environment Variable Validation", "Checked `os.getenv('API_KEY')` and raised clear ConfigError if unset.", "Directly accessed `os.environ['API_KEY']` causing unhandled KeyError in production.", "A"),
    ("Correct Decimal Precision", "Used `decimal.Decimal` for currency arithmetic avoiding float inaccuracies.", "Used binary float `0.1 + 0.2` resulting in `0.30000000000000004`.", "A"),
    ("Signal Handling", "Implemented `signal.signal(signal.SIGINT, handler)` for graceful shutdown.", "Ignored SIGINT; worker died leaving orphaned temp files.", "A"),
    ("API Rate Limiting", "Implemented exponential backoff with jitter on HTTP 429 status.", "Spammed retry requests in a tight `while True:` loop getting IP banned.", "A"),
    ("Mocking External Services", "Used `unittest.mock.patch` to mock third-party payment gateway in unit tests.", "Hit live production stripe endpoints inside CI unit tests.", "A"),
    ("Path Traversal Vulnerability", "Sanitized file path with `os.path.abspath` and verified prefix inside base directory.", "Allowed raw user input `../../etc/passwd`.", "A"),
    ("Cache Invalidation", "Invalidated Redis cache entry on record update.", "Updated database without cache eviction leading to stale data reads.", "A"),
    ("Unicode Encoding", "Opened file with explicit `encoding='utf-8'`.", "Opened file with system default ascii crashing on non-ASCII characters.", "A"),
    ("YAML Safe Load", "Used `yaml.safe_load(stream)` preventing arbitrary code execution.", "Used `yaml.load(stream, Loader=yaml.Loader)` exposing RCE vulnerability.", "A"),
    ("Database Migration", "Added backwards-compatible column with nullable default before dropping old column.", "Dropped active database column in production before updating application code.", "A"),
    ("Subprocess Safety", "Used `subprocess.run(['ls', dirname], check=True)` passing argument list.", "Used `subprocess.run(f'ls {dirname}', shell=True)` vulnerable to command injection.", "A"),
    ("Timestamp Timezones", "Stored timestamps in UTC using `datetime.now(timezone.utc)`.", "Used naive local timestamps causing day offset bugs across timezones.", "A"),
    ("Correct Hashing", "Used `hashlib.sha256()` with salt for password hashing.", "Used plaintext MD5 without salt.", "A"),
    ("HTTP Status Code", "Returned HTTP 404 Not Found when entity does not exist.", "Returned HTTP 200 OK with body `{'error': 'not found'}` breaking REST contract.", "A")
]

for idx, (title, sol_a, sol_b, pref) in enumerate(variations, start=11):
    pairs.append({
        "id": f"pair_{idx:02d}",
        "task": f"Implement {title}.",
        "candidate_a": f"{sol_a}\nVerification: All tests passing.",
        "candidate_b": f"{sol_b}\nVerification: Test suite failed or flagged security alert.",
        "preferred": pref,
        "category": title.lower().replace(" ", "_"),
        "expected_reason": f"Candidate {pref} implemented secure and correct solution."
    })

with open("standalone-eval/data/pairs.jsonl", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

print(f"Generated {len(pairs)} benchmark trajectory pairs in standalone-eval/data/pairs.jsonl")
