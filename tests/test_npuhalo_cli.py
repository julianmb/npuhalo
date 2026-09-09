import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.npuhalo_doctor import (
    check_npu_device,
    check_driver_and_kernel,
    check_memlock_limit,
    check_npu_server,
    check_gpu_server,
    run_doctor_checks,
    print_doctor_report,
)
from scripts.npuhalo_cli import main as cli_main


def test_check_npu_device_present():
    with patch("os.path.exists", return_value=True), \
         patch("os.access", return_value=True):
        res = check_npu_device()
        assert res.status == "PASS"
        assert "accessible" in res.message


def test_check_npu_device_missing():
    with patch("os.path.exists", return_value=False):
        res = check_npu_device()
        assert res.status == "FAIL"
        assert "not found" in res.message


def test_check_driver_and_kernel():
    results = check_driver_and_kernel()
    assert len(results) >= 2
    assert results[0].name == "Linux Kernel"
    assert results[0].status == "PASS"


def test_check_memlock_limit_unlimited():
    mock_limits = "Max locked memory         unlimited            unlimited            bytes     \n"
    with patch("builtins.open", mock_open(read_data=mock_limits)):
        res = check_memlock_limit()
        assert res.status == "PASS"
        assert "unlimited" in res.message


def test_check_memlock_limit_sufficient():
    # 16 GB in bytes: 17179869184
    mock_limits = "Max locked memory         17179869184          17179869184          bytes     \n"
    with patch("builtins.open", mock_open(read_data=mock_limits)):
        res = check_memlock_limit()
        assert res.status == "PASS"
        assert "sufficient" in res.message


def test_check_memlock_limit_insufficient():
    # 64 KB in bytes: 65536
    mock_limits = "Max locked memory         65536                65536                bytes     \n"
    with patch("builtins.open", mock_open(read_data=mock_limits)):
        res = check_memlock_limit()
        assert res.status == "WARN"
        assert "FastFlowLM recommends" in res.message


def test_check_npu_server_issue_716_detected():
    """Verify that if FastFlowLM silently echoes canary tag, issue #716 is flagged."""
    def mock_urlopen(req, timeout=1.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/v1/models" in url:
            resp = MagicMock()
            resp.status = 200
            resp.read.return_value = json.dumps({"data": [{"id": "qwen3.5:0.8b"}]}).encode("utf-8")
            resp.__enter__.return_value = resp
            return resp
        elif "/v1/chat/completions" in url:
            resp = MagicMock()
            resp.status = 200
            # Echoes back the canary tag (buggy upstream #716 behavior)
            resp.read.return_value = json.dumps({
                "model": "npuhalo-canary-test:99b",
                "choices": [{"message": {"content": "pong"}}],
            }).encode("utf-8")
            resp.__enter__.return_value = resp
            return resp
        raise ConnectionRefusedError()

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        results, active_model = check_npu_server("http://127.0.0.1:8001")
        assert active_model == "qwen3.5:0.8b"
        guard_check = [r for r in results if r.name == "Upstream Issue #716 Guard"][0]
        assert guard_check.status == "WARN"
        assert "FastFlowLM issue #716 detected" in guard_check.message


def test_check_npu_server_proper_rejection():
    """Verify that if server rejects invalid tag with HTTPError, status is PASS."""
    import urllib.error

    def mock_urlopen(req, timeout=1.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/v1/models" in url:
            resp = MagicMock()
            resp.status = 200
            resp.read.return_value = json.dumps({"data": [{"id": "qwen3.5:0.8b"}]}).encode("utf-8")
            resp.__enter__.return_value = resp
            return resp
        elif "/v1/chat/completions" in url:
            raise urllib.error.HTTPError(url, 404, "Model Not Found", {}, BytesIO(b"{}"))
        raise ConnectionRefusedError()

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        results, active_model = check_npu_server("http://127.0.0.1:8001")
        assert active_model == "qwen3.5:0.8b"
        guard_check = [r for r in results if r.name == "Upstream Issue #716 Guard"][0]
        assert guard_check.status == "PASS"
        assert "rejected invalid model tag" in guard_check.message


def test_run_doctor_checks_end_to_end():
    checks = run_doctor_checks()
    assert len(checks) >= 5
    # Should not throw when printing report
    print_doctor_report(checks)


def test_cli_route_dispatch(capsys):
    test_args = ["npuhalo", "route", "def solve_matrix(): pass"]
    with patch("sys.argv", test_args):
        cli_main()
    captured = capsys.readouterr()
    assert "Route  : GPU" in captured.out
    assert "Matched GPU requirement" in captured.out


def test_cli_doctor_dispatch(capsys):
    test_args = ["npuhalo", "doctor"]
    with patch("sys.argv", test_args):
        cli_main()
    captured = capsys.readouterr()
    assert "=== npuhalo doctor · System & Hardware Diagnostics ===" in captured.out
