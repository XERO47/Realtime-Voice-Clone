from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import asdict, dataclass, field

from fastapi import WebSocket, WebSocketDisconnect


@dataclass
class Participant:
    user_id: str
    role: str
    websocket: WebSocket
    session_id: str | None = None


@dataclass
class CallSession:
    session_id: str
    caller_a: str
    caller_b: str
    caller_a_role: str = "consumer"
    caller_b_role: str = "consumer"
    status: str = "ringing"
    created_at: float = field(default_factory=time.time)
    connected_at: float | None = None
    audio_levels: dict[str, float] = field(default_factory=dict)
    audio_online: dict[str, bool] = field(default_factory=dict)
    audio_voice_active: dict[str, bool] = field(default_factory=dict)
    audio_last_frame_at: dict[str, float] = field(default_factory=dict)
    detector_progress: dict[str, float] = field(default_factory=dict)
    detector_pending: dict[str, bool] = field(default_factory=dict)
    risk_raw_scores: dict[str, float | None] = field(default_factory=dict)
    risk_scores: dict[str, float | None] = field(default_factory=dict)
    risk_states: dict[str, str] = field(default_factory=dict)
    risk_latency_ms: dict[str, float | None] = field(default_factory=dict)
    risk_updated_at: dict[str, float] = field(default_factory=dict)
    risk_history: dict[str, list[dict]] = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "session_id": self.session_id,
            "caller_a": self.caller_a,
            "caller_b": self.caller_b,
            "caller_a_role": self.caller_a_role,
            "caller_b_role": self.caller_b_role,
            "status": self.status,
            "created_at": self.created_at,
            "connected_at": self.connected_at,
            "audio_levels": self.audio_levels,
            "audio_online": self.audio_online,
            "audio_voice_active": self.audio_voice_active,
            "audio_last_frame_at": self.audio_last_frame_at,
            "detector_progress": self.detector_progress,
            "detector_pending": self.detector_pending,
            "risk_raw_scores": self.risk_raw_scores,
            "risk_scores": self.risk_scores,
            "risk_states": self.risk_states,
            "risk_latency_ms": self.risk_latency_ms,
            "risk_updated_at": self.risk_updated_at,
            "risk_history": self.risk_history,
        }


class SessionManager:
    def __init__(self) -> None:
        self.participants: dict[str, Participant] = {}
        self.sessions: dict[str, CallSession] = {}
        self.monitors: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @staticmethod
    def normalize_id(value: str) -> str:
        return "".join(char for char in value.strip().upper() if char.isalnum() or char == "-")[:20]

    async def send(self, websocket: WebSocket | None, payload: dict) -> None:
        if websocket is None:
            return
        try:
            await websocket.send_json(payload)
        except (RuntimeError, WebSocketDisconnect):
            pass

    async def register(self, user_id: str, role: str, websocket: WebSocket) -> tuple[bool, str]:
        user_id = self.normalize_id(user_id)
        async with self._lock:
            if not user_id:
                return False, "Enter a valid calling ID."
            if user_id in self.participants:
                return False, "That calling ID is already in use."
            self.participants[user_id] = Participant(user_id, role, websocket)
        await self.broadcast_sessions()
        return True, user_id

    async def unregister(self, user_id: str) -> None:
        participant = self.participants.get(user_id)
        session_id = participant.session_id if participant else None
        async with self._lock:
            self.participants.pop(user_id, None)
        if session_id:
            await self.close_session(session_id, "Participant disconnected")
        await self.broadcast_sessions()

    async def create_session(self, caller_a: str, caller_b: str) -> CallSession:
        async with self._lock:
            if caller_b not in self.participants:
                raise ValueError("That user is not online.")
            if caller_a == caller_b:
                raise ValueError("You cannot call your own ID.")
            if self.participants[caller_a].session_id or self.participants[caller_b].session_id:
                raise ValueError("One participant is already in a call.")
            session_id = f"VG-{secrets.token_hex(3).upper()}"
            session = CallSession(
                session_id=session_id,
                caller_a=caller_a,
                caller_b=caller_b,
                caller_a_role=self.participants[caller_a].role,
                caller_b_role=self.participants[caller_b].role,
                audio_levels={caller_a: 0.0, caller_b: 0.0},
                audio_online={caller_a: False, caller_b: False},
                audio_voice_active={caller_a: False, caller_b: False},
                audio_last_frame_at={},
                detector_progress={caller_a: 0.0, caller_b: 0.0},
                detector_pending={caller_a: False, caller_b: False},
                risk_raw_scores={caller_a: None, caller_b: None},
                risk_scores={caller_a: None, caller_b: None},
                risk_states={caller_a: "insufficient", caller_b: "insufficient"},
                risk_latency_ms={caller_a: None, caller_b: None},
                risk_updated_at={},
                risk_history={caller_a: [], caller_b: []},
            )
            self.sessions[session_id] = session
            self.participants[caller_a].session_id = session_id
            self.participants[caller_b].session_id = session_id
        await self.broadcast_sessions()
        return session

    def get_session_for(self, user_id: str, session_id: str | None = None) -> CallSession | None:
        participant = self.participants.get(user_id)
        resolved = session_id or (participant.session_id if participant else None)
        return self.sessions.get(resolved) if resolved else None

    @staticmethod
    def peer_id(session: CallSession, user_id: str) -> str:
        return session.caller_b if session.caller_a == user_id else session.caller_a

    async def route(self, user_id: str, payload: dict) -> None:
        participant = self.participants.get(user_id)
        await self.send(participant.websocket if participant else None, payload)

    async def set_status(self, session: CallSession, status: str) -> None:
        session.status = status
        if status == "active" and session.connected_at is None:
            session.connected_at = time.time()
        await self.broadcast_sessions()

    async def close_session(self, session_id: str, reason: str) -> None:
        async with self._lock:
            session = self.sessions.pop(session_id, None)
            if not session:
                return
            users = (session.caller_a, session.caller_b)
            for user_id in users:
                participant = self.participants.get(user_id)
                if participant:
                    participant.session_id = None
        for user_id in users:
            await self.route(user_id, {"type": "call-ended", "session_id": session_id, "reason": reason})
        await self.broadcast_sessions()

    async def update_audio(
        self,
        session_id: str,
        user_id: str,
        *,
        level: float | None = None,
        online: bool | None = None,
        voice_active: bool | None = None,
        frame_received: bool = False,
    ) -> None:
        session = self.sessions.get(session_id)
        if not session or user_id not in (session.caller_a, session.caller_b):
            return
        if level is not None:
            session.audio_levels[user_id] = max(0.0, min(1.0, level))
        if online is not None:
            session.audio_online[user_id] = online
        if voice_active is not None:
            session.audio_voice_active[user_id] = voice_active
        if frame_received:
            session.audio_last_frame_at[user_id] = time.time()

    def refresh_audio_health(self) -> None:
        now = time.time()
        for session in self.sessions.values():
            for user_id in (session.caller_a, session.caller_b):
                last_frame = session.audio_last_frame_at.get(user_id)
                if last_frame is None or now - last_frame > 1.0:
                    session.audio_online[user_id] = False
                    session.audio_voice_active[user_id] = False
                    session.audio_levels[user_id] = 0.0
                elif now - last_frame > 0.20:
                    session.audio_levels[user_id] *= 0.72
                    if session.audio_levels[user_id] < 0.015:
                        session.audio_voice_active[user_id] = False

    async def update_risk(self, session_id: str, user_id: str, result: dict) -> None:
        session = self.sessions.get(session_id)
        if not session or user_id not in (session.caller_a, session.caller_b):
            return
        raw_score = result.get("spoof_probability")
        session.risk_raw_scores[user_id] = None if raw_score is None else float(raw_score)
        previous = session.risk_scores.get(user_id)
        if raw_score is None:
            score = None
        elif previous is None:
            score = float(raw_score)
        else:
            score = previous * 0.55 + float(raw_score) * 0.45
        if score is None:
            state = result.get("state", "insufficient")
        elif score < 0.35:
            state = "safe"
        elif score < 0.70:
            state = "verify"
        else:
            state = "high_risk"
        session.risk_scores[user_id] = score
        session.risk_states[user_id] = state
        session.risk_latency_ms[user_id] = result.get("latency_ms")
        session.risk_updated_at[user_id] = time.time()
        history = session.risk_history.setdefault(user_id, [])
        history.append({"time": time.time(), "score": score, "state": state})
        del history[:-30]

    async def broadcast_sessions(self) -> None:
        payload = {"type": "sessions", "sessions": [session.public() for session in self.sessions.values()]}
        stale: list[WebSocket] = []
        for websocket in list(self.monitors):
            try:
                await websocket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                stale.append(websocket)
        for websocket in stale:
            self.monitors.discard(websocket)


manager = SessionManager()
