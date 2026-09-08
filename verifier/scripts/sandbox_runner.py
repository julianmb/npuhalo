#!/usr/bin/env python3
"""
Trusted-code Python execution helper.

This uses a temporary file and timeout, but it is not a security boundary. The
child retains the current user's filesystem and network access. Run untrusted
candidate code only inside a disposable container or VM.
"""

import subprocess
import sys
import tempfile
import os

def run_in_sandbox(code_str: str, test_code: str, timeout_sec: float = 3.0) -> dict:
    indented_tests = "\n".join("    " + line for line in test_code.strip().split("\n") if line.strip())
    full_script = f"""
import sys

# Candidate Implementation
{code_str}

# Unit Test Assertions
try:
{indented_tests}
    print("TESTS_PASSED")
except AssertionError as e:
    print(f"TEST_FAILED: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"RUNTIME_ERROR: {{type(e).__name__}}: {{e}}")
    sys.exit(2)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(full_script)
        temp_path = tf.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        passed = (proc.returncode == 0 and "TESTS_PASSED" in proc.stdout)
        return {
            "passed": passed,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Execution timed out"
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    code = "def is_palindrome(s: str) -> bool:\n    clean = [c.lower() for c in s if c.isalnum()]\n    return clean == clean[::-1]"
    test = "assert is_palindrome('A man, a plan, a canal: Panama') == True"
    print("Testing sandbox:", run_in_sandbox(code, test))
