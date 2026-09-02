from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .runtime import CloneRuntime


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
runtime = CloneRuntime(ROOT / "cloner-data")

app = FastAPI(
    title="VoxGuard Voice Cloning Service",
    version="1.0.0",
    description="Consent-gated zero-shot voice cloning for the VoxGuard security demonstration.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Clone-Metadata"],
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class LoadRequest(BaseModel):
    device: str = "auto"


class GenerateRequest(BaseModel):
    voice_id: str
    text: str = Field(min_length=3, max_length=600)
    temperature: float = Field(default=0.8, ge=0.05, le=1.5)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)
    top_k: int = Field(default=1000, ge=50, le=2000)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    consent: bool = False


def api_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Voice profile was not found.")
    if isinstance(exc, (ValueError, RuntimeError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.get("/")
async def dashboard() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict:
    return runtime.status()


@app.get("/api/status")
async def status() -> dict:
    return runtime.status()


@app.post("/api/model/load")
async def load_model(request: LoadRequest) -> dict:
    try:
        return await run_in_threadpool(runtime.load, request.device)
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/api/model/unload")
async def unload_model() -> dict:
    try:
        return await run_in_threadpool(runtime.unload)
    except Exception as exc:
        raise api_error(exc) from exc


@app.get("/api/voices")
async def list_voices() -> dict:
    return {"voices": runtime.list_voices()}


@app.post("/api/voices")
async def add_voice(
    reference: UploadFile = File(...),
    name: str = Form("Reference voice"),
    consent: bool = Form(False),
) -> dict:
    if not consent:
        raise HTTPException(status_code=400, detail="Confirmed speaker consent is required.")
    content = await reference.read()
    try:
        voice = await run_in_threadpool(runtime.add_voice, name, reference.filename or "reference.wav", content)
        return {"voice": voice}
    except Exception as exc:
        raise api_error(exc) from exc


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    try:
        await run_in_threadpool(runtime.delete_voice, voice_id)
        return {"deleted": voice_id}
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/api/voices/clear-cache")
async def clear_voice_cache() -> dict:
    return await run_in_threadpool(runtime.clear_voice_cache)


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> Response:
    if not request.consent:
        raise HTTPException(status_code=400, detail="Confirmed speaker consent is required.")
    try:
        payload, metadata = await run_in_threadpool(
            runtime.generate,
            request.voice_id,
            request.text,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repetition_penalty,
            request.seed,
        )
        safe_name = "clone-" + request.voice_id + ".wav"
        return Response(
            content=payload,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}"',
                "X-Clone-Metadata": json.dumps(metadata, separators=(",", ":")),
                "Cache-Control": "no-store",
            },
        )
    except Exception as exc:
        raise api_error(exc) from exc

