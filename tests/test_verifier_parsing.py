import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "verifier", "src"))
sys.path.insert(0, os.path.join(ROOT, "verifier", "scripts"))

from verifier_client import parse_verdict
from verifier_client import NPUVerifierClient
from run_benchmark import extract_numeric_answer, extract_python_code


class ParseVerdictTests(unittest.TestCase):
    def setUp(self):
        self.parse = parse_verdict

    def test_abort_with_evidence_tag(self):
        verdict, evidence = self.parse(
            '<evidence>1 / 0</evidence>\n<verdict>ABORT</verdict>'
        )
        self.assertEqual(verdict, "ABORT")
        self.assertEqual(evidence, "1 / 0")

    def test_abort_with_double_quotes(self):
        verdict, evidence = self.parse('ABORT: line "x = y//" is invalid.')
        self.assertEqual(verdict, "ABORT")
        self.assertEqual(evidence, "x = y//")

    def test_suspect(self):
        verdict, _ = self.parse("Missing evidence; verdict: SUSPECT.")
        self.assertEqual(verdict, "SUSPECT")

    def test_continue_default(self):
        verdict, _ = self.parse("Everything looks fine so far.")
        self.assertEqual(verdict, "CONTINUE")

    def test_lowercase_verdict(self):
        verdict, _ = self.parse("we should abort here")
        self.assertEqual(verdict, "ABORT")


class NumericAnswerTests(unittest.TestCase):
    def test_boxed(self):
        self.assertAlmostEqual(extract_numeric_answer("so \\boxed{42}"), 42.0)

    def test_hash_marker(self):
        self.assertAlmostEqual(extract_numeric_answer("step 1... #### 18"), 18.0)

    def test_trailing_number(self):
        self.assertAlmostEqual(extract_numeric_answer("the total is 7"), 7.0)

    def test_none(self):
        self.assertIsNone(extract_numeric_answer("no numbers here"))


class NPUClientEndpointTests(unittest.TestCase):
    def test_chat_completions_url_from_v1(self):
        c = NPUVerifierClient._endpoint_to_chat_completions("http://127.0.0.1:13305/v1")
        self.assertEqual(c, "http://127.0.0.1:13305/v1/chat/completions")

    def test_models_url_from_v1(self):
        m = NPUVerifierClient._endpoint_to_models("http://127.0.0.1:13305/v1")
        self.assertEqual(m, "http://127.0.0.1:13305/v1/models")

    def test_endpoint_already_chat_completions(self):
        c = NPUVerifierClient._endpoint_to_chat_completions(
            "http://127.0.0.1:13305/v1/chat/completions")
        self.assertEqual(c, "http://127.0.0.1:13305/v1/chat/completions")

    def test_endpoint_without_v1(self):
        m = NPUVerifierClient._endpoint_to_models("http://127.0.0.1:13305")
        self.assertEqual(m, "http://127.0.0.1:13305/models")

    def test_unreachable_endpoint_unhealthy(self):
        self.assertFalse(NPUVerifierClient.healthy("http://127.0.0.1:1"))

    def test_prefers_thinking_model(self):
        c = NPUVerifierClient("http://127.0.0.1:8001/v1")
        c.available_models = lambda timeout=5.0: [
            "lfm2:1.2b", "lfm2.5-it:1.2b", "lfm2.5-tk:1.2b", "qwen3:8b"]
        self.assertEqual(c._pick_model_id(), "lfm2.5-tk:1.2b")

    def test_prefers_it_fallback(self):
        c = NPUVerifierClient("http://127.0.0.1:8001/v1")
        c.available_models = lambda timeout=5.0: ["lfm2.5-it:1.2b", "qwen3:8b"]
        self.assertEqual(c._pick_model_id(), "lfm2.5-it:1.2b")

    def test_backend_attribute(self):
        self.assertEqual(
            NPUVerifierClient("http://127.0.0.1:8001/v1").backend, "npu")


class CodeExtractionTests(unittest.TestCase):
    def test_fenced_block(self):
        text = "Here:\n```python\ndef f():\n    return 1\n```\nDone."
        self.assertEqual(extract_python_code(text), "def f():\n    return 1")

    def test_last_fence_wins(self):
        text = "```python\nv1\n```\nfixed:\n```python\nv2\n```"
        self.assertEqual(extract_python_code(text), "v2")

    def test_no_fence_passthrough(self):
        self.assertEqual(extract_python_code("x = 1"), "x = 1")


if __name__ == "__main__":
    unittest.main()
