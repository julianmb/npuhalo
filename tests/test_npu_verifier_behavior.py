import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "verifier", "src"))

import verifier_client
from verifier_client import NPUVerifierClient


def _mock_response(payload):
    body = json.dumps(payload).encode("utf-8")
    fake = mock.MagicMock()
    fake.__enter__.return_value = fake
    fake.__exit__.return_value = False
    fake.read.return_value = body
    return fake


def _chat_payload(verdicts):
    return {
        "choices": [{"message": {"content": v}} for v in verdicts],
        "usage": {"prompt_tokens": 42, "completion_tokens": 12},
    }


class NPUVerifierBehaviorTests(unittest.TestCase):
    def _run(self, verdicts):
        c = NPUVerifierClient("http://fake/v1")
        c.available_models = lambda timeout=5.0: ["lfm2.5-tk:1.2b"]
        fake = _mock_response(_chat_payload(verdicts))
        with mock.patch.object(verifier_client.urllib.request, "urlopen", return_value=fake):
            return c.evaluate_checkpoint("task", "trajectory",
                                         "Task:\n{task}\n\n{trajectory}",
                                         k_samples=3, temperature=0.7)

    def test_suspect_majority(self):
        res = self._run(["SUSPECT", "SUSPECT", "CONTINUE"])
        self.assertEqual(res["verdict"], "SUSPECT")
        self.assertEqual(res["votes"], {"CONTINUE": 1, "SUSPECT": 2})
        self.assertEqual(res["backend"], "npu")

    def test_abort_majority(self):
        res = self._run(["ABORT", "ABORT", "CONTINUE"])
        self.assertEqual(res["verdict"], "ABORT")
        self.assertEqual(res["votes"], {"CONTINUE": 1, "ABORT": 2})

    def test_continue_majority(self):
        res = self._run(["CONTINUE", "CONTINUE", "SUSPECT"])
        self.assertEqual(res["verdict"], "CONTINUE")
        self.assertEqual(res["votes"], {"CONTINUE": 2, "SUSPECT": 1})

    def test_prompt_token_passthrough(self):
        res = self._run(["CONTINUE", "CONTINUE", "CONTINUE"])
        self.assertEqual(res["prompt_tokens"], 42)

    def test_num_choices_padding(self):
        # Endpoint may return fewer choices than k; missing are SUSPECT-by-default.
        res = self._run(["ABORT", "ABORT"])
        self.assertEqual(res["verdict"], "ABORT")
        self.assertEqual(len(res["raw_responses"]), 3)


if __name__ == "__main__":
    unittest.main()
