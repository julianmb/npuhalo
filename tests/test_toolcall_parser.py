#!/usr/bin/env python3
"""
test_toolcall_parser.py — Regression tests for the incremental tool-call parser.

Locks in the v2 "resync-on-error" semantics:
  - a structural violation marks a provisional UNRECOVERABLE and is recorded;
  - the scanner resynchronizes on the next <tool_call> opener;
  - a later complete valid call upgrades the final verdict to VALID_SO_FAR with
    protocol warnings (reference-equivalent: never reject what the reference
    extractor accepts);
  - only violation-without-recovery is finally UNRECOVERABLE;
  - every proper prefix of a valid call is never UNRECOVERABLE.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "verifier", "src"))
from toolcall_parser import IncrementalToolCallParser, classify_stream  # noqa: E402

GOOD = ("<tool_call>\n<function=shell>\n<parameter=command>ls -la</parameter>\n"
        "</function>\n</tool_call>")
WRITE_OK = ("<tool_call>\n<function=write>\n<parameter=path>a.py</parameter>\n"
            "<parameter=content>x=1\ny=2</parameter>\n</function>\n</tool_call>")
# D06 s1 anomaly pattern: corrupted first call, valid second call later
D06_HEAD = "The test failed. Let me reconsider. " + "analysis " * 120
D06_CORRUPTED = ("<tool_call>\n<function=shell>\n<parameter=command>ls</parameter>"
                 "\n\n<function")
D06_TAIL = "\nSome more prose.\n\n<tool_call>\n<function=test>\n</function>\n</tool_call>"


def char_feed(text):
    p = IncrementalToolCallParser()
    for ch in text:
        p.feed(ch)
    return p.result()


class TestValidCalls(unittest.TestCase):
    def test_whole_string(self):
        self.assertEqual(classify_stream([GOOD])["state"], "VALID_SO_FAR")

    def test_char_stream_extracts_arguments(self):
        r = char_feed(GOOD)
        self.assertEqual(r["state"], "VALID_SO_FAR")
        self.assertEqual(r["call"]["arguments"], {"command": "ls -la"})

    def test_multi_param_value_with_newlines(self):
        r = char_feed(WRITE_OK)
        self.assertEqual(r["state"], "VALID_SO_FAR")
        self.assertEqual(r["call"]["arguments"]["content"], "x=1\ny=2")

    def test_prose_prefix_allowed(self):
        self.assertEqual(classify_stream(["Let me analyze.\n\n" + GOOD])["state"],
                         "VALID_SO_FAR")


class TestSchemaTolerance(unittest.TestCase):
    def test_schema_mismatch_is_warning_not_failure(self):
        bad = ("<tool_call>\n<function=write>\n<parameter=command>echo hi</parameter>\n"
               "</function>\n</tool_call>")
        r = classify_stream([bad])
        self.assertEqual(r["state"], "VALID_SO_FAR")
        self.assertTrue(r["call"]["schema_warnings"])

    def test_empty_param_list_is_warning_not_failure(self):
        r = classify_stream(["<tool_call>\n<function=shell></function></tool_call>"])
        self.assertEqual(r["state"], "VALID_SO_FAR")
        self.assertTrue(r["call"]["schema_warnings"])


class TestStructuralFailures(unittest.TestCase):
    def test_unknown_function_unrecoverable(self):
        r = classify_stream(["<tool_call>\n<function=rmrf>\n<x>1</x></function></tool_call>"])
        self.assertEqual(r["state"], "UNRECOVERABLE")

    def test_garbage_in_body_unrecoverable_without_recovery(self):
        p = IncrementalToolCallParser()
        for ch in "<tool_call>\n<function=shell>\nrandom junk":
            p.feed(ch)
        for ch in "\nprose only, no new call":
            p.feed(ch)
        self.assertEqual(p.result()["state"], "UNRECOVERABLE")
        r = classify_stream(["<tool_call>\n<function=shell>\nrandom junk", "\nmore prose"])
        self.assertEqual(r["state"], "UNRECOVERABLE")


class TestResyncOnError(unittest.TestCase):
    """Regression for the D06 s1 live/post-hoc mismatch."""

    def test_recovery_upgrades_final_verdict_char_stream(self):
        r = char_feed(D06_HEAD + D06_CORRUPTED + D06_TAIL)
        self.assertEqual(r["state"], "VALID_SO_FAR")
        self.assertEqual(r["call"]["name"], "test")
        self.assertTrue(any("protocol_violation_recovered" in w
                            for w in r.get("schema_warnings", [])))

    def test_recovery_upgrades_single_chunk(self):
        r = classify_stream([D06_HEAD + D06_CORRUPTED + D06_TAIL])
        self.assertEqual(r["state"], "VALID_SO_FAR")
        self.assertTrue(r["errors"])  # violation recorded even though recovered

    def test_no_recovery_stays_unrecoverable_across_chunks(self):
        r = classify_stream(["<tool_call>\n<function=shell>\nrandom junk", "\nmore prose"])
        self.assertEqual(r["state"], "UNRECOVERABLE")


class TestPrefixSafety(unittest.TestCase):
    def test_no_prefix_of_valid_call_ever_unrecoverable(self):
        for i in range(1, len(WRITE_OK) + 1):
            st = classify_stream([WRITE_OK[:i]])["state"]
            self.assertIn(st, ("RECOVERABLE", "VALID_SO_FAR"), msg=f"i={i}")


class TestMetaMentions(unittest.TestCase):
    def test_literal_tag_in_prose_not_flagged(self):
        for t in ['I\'ve been emitting "Emit a valid <tool_call> now." which seems odd.',
                  'The system says <tool_call> is required.']:
            self.assertEqual(classify_stream([t])["state"], "RECOVERABLE")


if __name__ == "__main__":
    unittest.main()
