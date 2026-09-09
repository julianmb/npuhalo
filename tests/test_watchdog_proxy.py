#!/usr/bin/env python3
"""
test_watchdog_proxy.py — Unit and Integration tests for NPU Watchdog & Smart Proxy.
"""

import sys
import os
import asyncio
from aiohttp.test_utils import TestServer, TestClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "verifier", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from watchdog_analyzer import WatchdogAnalyzer
from npuhalo_proxy import NPUHaloProxy, create_app

class TestWatchdogAnalyzer:
    def setup_method(self):
        self.analyzer = WatchdogAnalyzer(enable_npu_eval=False)

    def test_detects_destructive_root_deletion(self):
        malicious_samples = [
            "<parameter=command>rm -rf /</parameter>",
            "<parameter=command>rm -rf /*</parameter>",
            "```bash\nrm -rf ~\n```",
            "<parameter=command>rm -r .git</parameter>",
            "<parameter=command>mkfs.ext4 /dev/sda1</parameter>",
            "<parameter=command>dd if=/dev/zero of=/dev/nvme0n1</parameter>",
            "<parameter=command>git clean -fdx</parameter>",
        ]
        for sample in malicious_samples:
            verdict = self.analyzer.check_regex_rules(sample)
            assert verdict is not None, f"Failed to detect destructive command: {sample}"
            assert verdict.is_anomaly is True
            assert verdict.category == "DESTRUCTIVE_COMMAND"
            assert verdict.risk_level in ("HIGH", "CRITICAL")

    def test_detects_secret_leakage(self):
        secret_samples = [
            "Here is the token: sk-proj-1234567890abcdef1234567890abcdef",
            "Exporting token: ghp_123456789012345678901234567890123456",
            "Command: <parameter=command>cat .env</parameter>",
            "Command: <parameter=command>cat ~/.ssh/id_rsa</parameter>",
        ]
        for sample in secret_samples:
            verdict = self.analyzer.check_regex_rules(sample)
            assert verdict is not None, f"Failed to detect secret leak: {sample}"
            assert verdict.is_anomaly is True
            assert verdict.category == "SECRET_LEAK"

    def test_allows_benign_commands(self):
        safe_samples = [
            "<parameter=command>ls -la</parameter>",
            "<parameter=command>pytest tests/test_api.py</parameter>",
            "<parameter=command>git status</parameter>",
            "<parameter=command>python3 scripts/npu_benchmark.py</parameter>",
            "```python\ndef solve(): return 42\n```",
        ]
        for sample in safe_samples:
            verdict = self.analyzer.check_regex_rules(sample)
            assert verdict is None, f"False positive on safe sample: {sample}"

    def test_detects_infinite_loops(self):
        session_id = "test_session_1"
        cmd = "<parameter=command>cargo test --workspace</parameter>"
        
        # 1st time: safe
        assert self.analyzer.check_loop_oscillation(session_id, cmd) is None
        # 2nd time: safe
        assert self.analyzer.check_loop_oscillation(session_id, cmd) is None
        # 3rd time: triggers loop anomaly
        v = self.analyzer.check_loop_oscillation(session_id, cmd)
        assert v is not None
        assert v.is_anomaly is True
        assert v.category == "LOOP_DETECTED"
        assert v.risk_level == "HIGH"

def test_proxy_endpoints_sync():
    async def _run():
        proxy = NPUHaloProxy(
            gpu_url="http://127.0.0.1:8012",
            npu_url="http://127.0.0.1:8001",
            guard_mode="block",
        )
        await proxy.start()
        app = create_app(proxy)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()

        try:
            # 1. Test /v1/models
            resp = await client.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            assert "Ornith-1.5-35B-A3B-ROCmFP4.gguf" in model_ids
            assert "qwen3.5:0.8b-watchdog" in model_ids


            # 2. Test /v1/watchdog/status
            resp = await client.get("/v1/watchdog/status")
            assert resp.status == 200
            stats = await resp.json()
            assert "total_audited" in stats
            assert stats["guard_mode"] == "block"

            # 3. Test /v1/watchdog/clear
            resp = await client.post("/v1/watchdog/clear")
            assert resp.status == 200
            res = await resp.json()
            assert res["status"] == "cleared"
        finally:
            await client.close()
            await server.close()
            await proxy.stop()

    asyncio.run(_run())
