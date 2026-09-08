#!/usr/bin/env python3
"""
test_npu_router.py — Unit tests for the Hybrid NPU Query Router.
"""

import sys
import os
import pytest
import asyncio

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from npu_router import HybridNPURouter

@pytest.mark.asyncio
def test_hybrid_router_classification():
    async def _run():
        router = HybridNPURouter()
        # Fast deterministic checks do not require an active network session
        
        # 1. Trivial queries -> NPU
        r_fact = await router.classify("What is the capital of France?")
        assert r_fact["route"] == "npu"
        assert r_fact["method"] == "deterministic_rule"

        r_math = await router.classify("What is 15 * 24?")
        assert r_math["route"] == "npu"

        r_greet = await router.classify("Hello! How are you today?")
        assert r_greet["route"] == "npu"

        # 2. Complex / Coding / Reasoning queries -> GPU
        r_code = await router.classify("Write a python function to compute fibonacci.")
        assert r_code["route"] == "gpu"
        assert r_code["method"] == "deterministic_rule"

        r_cpp = await router.classify("Implement a lock-free queue in C++.")
        assert r_cpp["route"] == "gpu"

        r_sql = await router.classify("SELECT user_id, count(*) FROM orders GROUP BY user_id")
        assert r_sql["route"] == "gpu"

        r_proof = await router.classify("Prove that there are infinitely many primes.")
        assert r_proof["route"] == "gpu"

    asyncio.run(_run())
