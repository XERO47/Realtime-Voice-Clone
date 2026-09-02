from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

import httpx

from .session_manager import manager


DETECTOR_URL = os.getenv("VOXGUARD_DETECTOR_URL", "http://127.0.0.1:8001").rstrip("/")
WINDOW_SECONDS = 4
HOP_SECONDS = 2

# Reused across calls so the underlying TCP connection (and the tunnel
# channel it rides on, when the detector is remote) stays warm instead of
# paying a fresh connection setup on every ~2s inference call.
# Browser-like User-Agent: devtunnels.ms's anonymous-access relay blocks
# httpx's default UA (same class of block as curl's default UA).
_client = httpx.AsyncClient(
    timeout=httpx.Timeout(45.0, connect=5.0),
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
)


@dataclass
class ParticipantBuffer:
    sample_rate: int
    pcm: bytearray = field(default_factory=bytearray)
    bytes_since_inference: int = 0
    pending: bool = False
    has_submitted: bool = False

    @property
    def target_bytes(self) -> int:
        return self.sample_rate * WINDOW_SECONDS * 4

    @property
    def hop_bytes(self) -> int:
        return self.sample_rate * HOP_SECONDS * 4


class DetectorBridge:
    def __init__(self) -> None:
        self.buffers: dict[tuple[str, str], ParticipantBuffer] = {}

    async def status(self) -> dict:
        try:
            response = await _client.get(f"{DETECTOR_URL}/health", timeout=2.0)
            response.raise_for_status()
            return {
                "connected": True,
                "url": DETECTOR_URL,
                **response.json(),
                "hop_seconds": HOP_SECONDS,
                "overlap_percent": int((1 - HOP_SECONDS / WINDOW_SECONDS) * 100),
            }
        except Exception as error:
            return {"connected": False, "url": DETECTOR_URL, "detail": str(error)}

    def reset(self, session_id: str, user_id: str) -> None:
        self.buffers.pop((session_id, user_id), None)
        session = manager.sessions.get(session_id)
        if session:
            session.detector_progress[user_id] = 0.0
            session.detector_pending[user_id] = False

    def feed(self, session_id: str, user_id: str, data: bytes, sample_rate: int) -> None:
        key = (session_id, user_id)
        state = self.buffers.get(key)
        if state is None or state.sample_rate != sample_rate:
            state = ParticipantBuffer(sample_rate=sample_rate)
            self.buffers[key] = state
        state.pcm.extend(data)
        state.bytes_since_inference += len(data)
        if len(state.pcm) > state.target_bytes:
            del state.pcm[: len(state.pcm) - state.target_bytes]
        session = manager.sessions.get(session_id)
        if session:
            collected = state.bytes_since_inference if state.has_submitted else len(state.pcm)
            required = state.hop_bytes if state.has_submitted else state.target_bytes
            session.detector_progress[user_id] = min(1.0, collected / required)
            session.detector_pending[user_id] = state.pending
        ready = len(state.pcm) >= state.target_bytes and (
            not state.has_submitted or state.bytes_since_inference >= state.hop_bytes
        )
        if not ready or state.pending:
            return
        state.bytes_since_inference = 0
        state.pending = True
        state.has_submitted = True
        if session:
            session.detector_progress[user_id] = 1.0
            session.detector_pending[user_id] = True
        window = bytes(state.pcm)
        asyncio.create_task(self._infer(key, state, window))

    async def _infer(self, key: tuple[str, str], state: ParticipantBuffer, window: bytes) -> None:
        session_id, user_id = key
        try:
            response = await _client.post(
                f"{DETECTOR_URL}/infer/pcm",
                content=window,
                headers={"X-Sample-Rate": str(state.sample_rate), "Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            result = response.json()
            await manager.update_risk(session_id, user_id, result)
            await manager.broadcast_sessions()
        except Exception as error:
            await manager.update_risk(
                session_id,
                user_id,
                {"state": "model_error", "spoof_probability": None, "detail": str(error), "latency_ms": None},
            )
            await manager.broadcast_sessions()
        finally:
            current = self.buffers.get(key)
            if current is state:
                current.pending = False
                session = manager.sessions.get(session_id)
                if session:
                    session.detector_pending[user_id] = False
                    session.detector_progress[user_id] = min(1.0, current.bytes_since_inference / current.hop_bytes)


detector_bridge = DetectorBridge()
