from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio_hub import audio_hub
from .cloner_bridge import cloner_bridge
from .detector_bridge import detector_bridge
from .session_manager import manager
from .tts_service import tts_service


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
VOICE_CLONE_DIR = Path(os.getenv("VOXGUARD_VOICE_CLONE_DIR", str(ROOT.parent / "demo-voice-clones")))
VOICE_CLONE_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".aac", ".flac", ".webm"}


async def telemetry_loop() -> None:
    while True:
        await asyncio.sleep(0.25)
        manager.refresh_audio_health()
        payloads = [session.public() for session in manager.sessions.values()]
        stale: list[WebSocket] = []
        for websocket in list(manager.monitors):
            try:
                await websocket.send_json({"type": "telemetry", "sessions": payloads})
            except (RuntimeError, WebSocketDisconnect):
                stale.append(websocket)
        for websocket in stale:
            manager.monitors.discard(websocket)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(telemetry_loop())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="VoxGuard VoIP MVP", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def call_page() -> FileResponse:
    return FileResponse(STATIC / "call.html")


@app.get("/dashboard")
async def dashboard_page() -> FileResponse:
    return FileResponse(STATIC / "dashboard.html")


@app.get("/attacker")
async def attacker_page() -> FileResponse:
    return FileResponse(STATIC / "attacker.html")


@app.get("/tts")
async def tts_page() -> FileResponse:
    return FileResponse(STATIC / "tts.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "participants": len(manager.participants),
        "sessions": len(manager.sessions),
        "tts_installed": tts_service.installed,
        "tts_loaded": tts_service.loaded,
    }


@app.get("/api/detector/status")
async def detector_status() -> dict:
    return await detector_bridge.status()


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    voice: str = "af_sarah"
    speed: float = Field(default=1.0, ge=0.75, le=1.35)


@app.get("/api/tts/status")
async def tts_status() -> dict:
    return {
        "engine": "Kokoro-82M ONNX",
        "installed": tts_service.installed,
        "loaded": tts_service.loaded,
        "device": "CPU",
    }


@app.get("/api/tts/voices")
async def tts_voices() -> dict:
    if not tts_service.installed:
        raise HTTPException(status_code=503, detail="TTS model files are not installed.")
    try:
        voices = await asyncio.to_thread(tts_service.voices)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"voices": voices}


@app.post("/api/tts/generate")
async def generate_tts(request: TTSRequest) -> Response:
    if not tts_service.installed:
        raise HTTPException(status_code=503, detail="TTS model files are not installed.")
    try:
        speech = await asyncio.to_thread(tts_service.generate, request.text.strip(), request.voice, request.speed)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return Response(
        speech.wav_bytes,
        media_type="audio/wav",
        headers={
            "X-TTS-Voice": speech.voice,
            "X-Audio-Duration": f"{speech.duration_seconds:.3f}",
            "X-Generation-Time": f"{speech.generation_seconds:.3f}",
            "X-Sample-Rate": str(speech.sample_rate),
        },
    )


@app.get("/api/voice-clones")
async def voice_clones() -> dict:
    if not VOICE_CLONE_DIR.is_dir():
        return {"directory": str(VOICE_CLONE_DIR), "clips": []}
    clips = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "url": f"/api/voice-clones/{path.name}",
        }
        for path in sorted(VOICE_CLONE_DIR.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in VOICE_CLONE_EXTENSIONS
    ]
    return {"directory": str(VOICE_CLONE_DIR), "clips": clips}


@app.get("/api/voice-clones/{clip_name}")
async def voice_clone_clip(clip_name: str) -> FileResponse:
    if Path(clip_name).name != clip_name:
        raise HTTPException(status_code=400, detail="Invalid clone clip name.")
    path = VOICE_CLONE_DIR / clip_name
    if not path.is_file() or path.suffix.lower() not in VOICE_CLONE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Clone clip not found.")
    return FileResponse(path)


@app.get("/api/cloner/status")
async def cloner_status() -> dict:
    return await cloner_bridge.status()


@app.get("/api/cloner/voices")
async def cloner_voices() -> dict:
    try:
        return await cloner_bridge.voices()
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Cloning service unavailable: {error}") from error


class ClonerGenerateRequest(BaseModel):
    voice_id: str
    text: str = Field(min_length=3, max_length=600)
    temperature: float = Field(default=0.8, ge=0.05, le=1.5)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)
    top_k: int = Field(default=1000, ge=50, le=2000)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    consent: bool = False


@app.post("/api/cloner/generate")
async def cloner_generate(request: ClonerGenerateRequest) -> Response:
    if not request.consent:
        raise HTTPException(status_code=400, detail="Confirmed speaker consent is required.")
    try:
        payload, metadata = await cloner_bridge.generate(request.model_dump())
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=error.response.status_code, detail=error.response.text) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Cloning service unavailable: {error}") from error
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={"X-Clone-Metadata": json.dumps(metadata, separators=(",", ":"))},
    )


@app.websocket("/ws/signaling/{raw_user_id}")
async def signaling(websocket: WebSocket, raw_user_id: str) -> None:
    await websocket.accept()
    role = websocket.query_params.get("role", "consumer")
    ok, user_id = await manager.register(raw_user_id, role, websocket)
    if not ok:
        await websocket.send_json({"type": "register-error", "message": user_id})
        await websocket.close(code=4001)
        return
    await websocket.send_json({"type": "registered", "user_id": user_id, "role": role})
    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("type")
            if event == "call-request":
                target = manager.normalize_id(message.get("target", ""))
                try:
                    session = await manager.create_session(user_id, target)
                except ValueError as error:
                    await manager.send(websocket, {"type": "call-error", "message": str(error)})
                    continue
                await manager.send(websocket, {"type": "call-ringing", "session": session.public(), "peer_id": target, "role": "caller_a"})
                await manager.route(target, {"type": "incoming-call", "session": session.public(), "from": user_id, "role": "caller_b"})
                continue

            session = manager.get_session_for(user_id, message.get("session_id"))
            if not session:
                await manager.send(websocket, {"type": "call-error", "message": "That call session is no longer available."})
                continue
            peer_id = manager.peer_id(session, user_id)

            if event == "call-accept":
                await manager.set_status(session, "connecting")
                await manager.route(peer_id, {"type": "call-accepted", "session": session.public(), "by": user_id})
            elif event == "call-decline":
                await manager.route(peer_id, {"type": "call-declined", "session_id": session.session_id, "by": user_id})
                await manager.close_session(session.session_id, "Call declined")
            elif event in {"offer", "answer", "ice-candidate"}:
                await manager.route(peer_id, {**message, "from": user_id, "session_id": session.session_id})
            elif event == "call-connected":
                await manager.set_status(session, "active")
            elif event == "call-end":
                await manager.close_session(session.session_id, "Call ended")
    except WebSocketDisconnect:
        pass
    finally:
        await manager.unregister(user_id)


@app.websocket("/ws/audio/{session_id}/{raw_user_id}")
async def audio_mirror(websocket: WebSocket, session_id: str, raw_user_id: str) -> None:
    await websocket.accept()
    user_id = manager.normalize_id(raw_user_id)
    sample_rate = int(websocket.query_params.get("sample_rate", "48000"))
    if not await audio_hub.attach_producer(session_id, user_id, sample_rate, websocket):
        await websocket.close(code=4004)
        return
    try:
        while True:
            data = await websocket.receive_bytes()
            await audio_hub.publish(session_id, user_id, data)
    except WebSocketDisconnect:
        pass
    finally:
        await audio_hub.detach_producer(session_id, user_id, websocket)


@app.websocket("/ws/monitor")
async def monitor(websocket: WebSocket) -> None:
    await websocket.accept()
    manager.monitors.add(websocket)
    await websocket.send_json({"type": "sessions", "sessions": [session.public() for session in manager.sessions.values()]})
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") != "intervention":
                continue
            session = manager.sessions.get(message.get("session_id"))
            if not session:
                await websocket.send_json({"type": "monitor-error", "message": "Session not found."})
                continue
            payload = {
                "type": "intervention",
                "action": message.get("action", "verify"),
                "target": message.get("target", "both"),
                "session_id": session.session_id,
            }
            await manager.route(session.caller_a, payload)
            await manager.route(session.caller_b, payload)
            if payload["action"] == "end":
                await manager.close_session(session.session_id, "Ended by monitoring console")
    except WebSocketDisconnect:
        pass
    finally:
        manager.monitors.discard(websocket)


@app.websocket("/ws/tap/{session_id}/{raw_user_id}")
async def tap_audio(websocket: WebSocket, session_id: str, raw_user_id: str) -> None:
    await websocket.accept()
    user_id = manager.normalize_id(raw_user_id)
    if not await audio_hub.attach_subscriber(session_id, user_id, websocket):
        await websocket.close(code=4004)
        return
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        audio_hub.detach_subscriber(session_id, user_id, websocket)
