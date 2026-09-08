#!/usr/bin/env python3
"""
async_pipeline.py — Asynchronous Double-Buffering Coordinator for Heterogeneous Speculative Decoding
Coordinates NPU drafting (Window N+1) concurrently while iGPU verifies Window N.
"""

import asyncio
from typing import List, Any, Tuple, Callable
from .ring_buffer import SharedTokenRingBuffer

class AsyncDoubleBufferPipeline:
    def __init__(self, draft_window_size: int = 4):
        self.draft_window_size = draft_window_size
        self.ring_buffer = SharedTokenRingBuffer(create=True)
        self.stats = {
            "total_windows": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "overlap_time_saved_ms": 0.0,
        }

    async def execute_heterogeneous_window(
        self,
        draft_fn: Callable[[str, int], Any],
        verify_fn: Callable[[List[int]], Any],
        prompt: str,
        current_tokens: List[int],
    ) -> Tuple[List[int], int]:
        """
        Executes double-buffered speculation:
        - Launches draft_fn on NPU for window N+1 in background
        - Concurrently executes verify_fn on iGPU for current batch
        """
        # Step 1: Concurrently launch NPU drafter for window N+1 while GPU verifies window N
        draft_task = asyncio.create_task(draft_fn(prompt, self.draft_window_size))

        # Step 2: Run iGPU parallel GEMM verification pass
        verify_result = await verify_fn(current_tokens)

        # Step 3: Await draft tokens for next window
        draft_result = await draft_task

        self.stats["total_windows"] += 1

        return verify_result, draft_result

    def close(self):
        self.ring_buffer.close()
