"""Deterministic test for the detector bridge's 4 s window / 2 s hop."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detector_bridge import DetectorBridge
from app.session_manager import CallSession, manager


async def run() -> None:
    bridge = DetectorBridge()
    session_id = "SLIDING-TEST"
    user_id = "CALLER-A"
    session = CallSession(
        session_id=session_id,
        caller_a=user_id,
        caller_b="CALLER-B",
        detector_progress={user_id: 0.0, "CALLER-B": 0.0},
        detector_pending={user_id: False, "CALLER-B": False},
        risk_history={user_id: [], "CALLER-B": []},
    )
    manager.sessions[session_id] = session
    submitted: list[bytes] = []

    async def capture_inference(key: tuple[str, str], state, window: bytes) -> None:
        submitted.append(window)
        state.pending = False

    bridge._infer = capture_inference  # type: ignore[method-assign]

    try:
        sample_rate = 10
        bytes_per_second = sample_rate * 4

        for second in range(1, 4):
            bridge.feed(session_id, user_id, bytes([second]) * bytes_per_second, sample_rate)
            await asyncio.sleep(0)
            assert len(submitted) == 0, "Inference started before the first 4-second window was full."

        bridge.feed(session_id, user_id, bytes([4]) * bytes_per_second, sample_rate)
        await asyncio.sleep(0)
        assert len(submitted) == 1, "The first inference did not start at 4 seconds."

        bridge.feed(session_id, user_id, bytes([5]) * bytes_per_second, sample_rate)
        await asyncio.sleep(0)
        assert len(submitted) == 1, "Inference started before the 2-second hop elapsed."

        bridge.feed(session_id, user_id, bytes([6]) * bytes_per_second, sample_rate)
        await asyncio.sleep(0)
        assert len(submitted) == 2, "The second inference did not start after the 2-second hop."

        hop_bytes = sample_rate * 2 * 4
        assert submitted[0][hop_bytes:] == submitted[1][:-hop_bytes], "Adjacent windows do not overlap by 50%."
        assert len(submitted[0]) == len(submitted[1]) == sample_rate * 4 * 4
    finally:
        manager.sessions.pop(session_id, None)

    print("Sliding-window test passed: 4-second window, 2-second hop, and exact 50% overlap.")


if __name__ == "__main__":
    asyncio.run(run())
