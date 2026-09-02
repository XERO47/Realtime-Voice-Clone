from __future__ import annotations

import asyncio
import math
import struct
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect

from .session_manager import manager
from .detector_bridge import detector_bridge


class AudioHub:
    def __init__(self) -> None:
        self.producers: dict[tuple[str, str], WebSocket] = {}
        self.sample_rates: dict[tuple[str, str], int] = {}
        self.subscribers: dict[tuple[str, str], set[WebSocket]] = defaultdict(set)
        self._last_level_emit: dict[tuple[str, str], float] = {}

    async def attach_producer(self, session_id: str, user_id: str, sample_rate: int, websocket: WebSocket) -> bool:
        session = manager.sessions.get(session_id)
        if not session or user_id not in (session.caller_a, session.caller_b):
            return False
        key = (session_id, user_id)
        self.producers[key] = websocket
        self.sample_rates[key] = sample_rate
        await manager.update_audio(session_id, user_id, level=0, online=False, voice_active=False)
        await manager.broadcast_sessions()
        return True

    async def detach_producer(self, session_id: str, user_id: str, websocket: WebSocket) -> None:
        key = (session_id, user_id)
        if self.producers.get(key) is websocket:
            self.producers.pop(key, None)
            detector_bridge.reset(session_id, user_id)
            await manager.update_audio(session_id, user_id, level=0, online=False, voice_active=False)
            await manager.broadcast_sessions()

    async def attach_subscriber(self, session_id: str, user_id: str, websocket: WebSocket) -> bool:
        session = manager.sessions.get(session_id)
        if not session or user_id not in (session.caller_a, session.caller_b):
            return False
        key = (session_id, user_id)
        self.subscribers[key].add(websocket)
        await websocket.send_json({
            "type": "tap-ready",
            "session_id": session_id,
            "participant_id": user_id,
            "sample_rate": self.sample_rates.get(key, 48000),
        })
        return True

    def detach_subscriber(self, session_id: str, user_id: str, websocket: WebSocket) -> None:
        key = (session_id, user_id)
        self.subscribers[key].discard(websocket)
        if not self.subscribers[key]:
            self.subscribers.pop(key, None)

    @staticmethod
    def rms(data: bytes) -> float:
        count = len(data) // 4
        if count == 0:
            return 0.0
        samples = struct.unpack(f"<{count}f", data[: count * 4])
        energy = sum(sample * sample for sample in samples) / count
        return min(1.0, math.sqrt(energy) * 4.0)

    async def publish(self, session_id: str, user_id: str, data: bytes) -> None:
        key = (session_id, user_id)
        raw_level = self.rms(data)
        session = manager.sessions.get(session_id)
        previous = session.audio_levels.get(user_id, 0.0) if session else 0.0
        level = max(raw_level, previous * 0.76)
        await manager.update_audio(
            session_id,
            user_id,
            level=level,
            online=True,
            voice_active=level >= 0.015,
            frame_received=True,
        )
        detector_bridge.feed(session_id, user_id, data, self.sample_rates.get(key, 48000))

        stale: list[WebSocket] = []
        sends = []
        for websocket in list(self.subscribers.get(key, set())):
            sends.append(self._safe_send_bytes(websocket, data, stale))
        if sends:
            await asyncio.gather(*sends)
        for websocket in stale:
            self.detach_subscriber(session_id, user_id, websocket)

    @staticmethod
    async def _safe_send_bytes(websocket: WebSocket, data: bytes, stale: list[WebSocket]) -> None:
        try:
            await websocket.send_bytes(data)
        except (RuntimeError, WebSocketDisconnect):
            stale.append(websocket)


audio_hub = AudioHub()
