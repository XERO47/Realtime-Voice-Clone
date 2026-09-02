from __future__ import annotations

import asyncio
import math
import time

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request

from .model import BACKBONE_CHECKPOINT_PATH, CHECKPOINT_PATH, SAMPLE_RATE, WINDOW_SECONDS, runtime


app = FastAPI(title="VoxGuard Detection Model Service", version="0.1.0")


def prepare_waveform(raw_pcm: bytes, source_rate: int) -> tuple[torch.Tensor, float]:
    samples = np.frombuffer(raw_pcm, dtype="<f4").copy()
    if not len(samples):
        raise ValueError("The PCM window is empty.")
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    samples = np.clip(samples, -1.0, 1.0)
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    waveform = torch.from_numpy(samples).float().unsqueeze(0)
    if source_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, source_rate, SAMPLE_RATE)
    target = int(SAMPLE_RATE * WINDOW_SECONDS)
    if waveform.shape[1] < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[1]))
    waveform = waveform[:, -target:]  # most recent window, not a random crop — this is live streaming
    return waveform, rms


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "checkpoint": CHECKPOINT_PATH.is_file(),
        "checkpoint_name": CHECKPOINT_PATH.name,
        "checkpoint_bytes": CHECKPOINT_PATH.stat().st_size if CHECKPOINT_PATH.is_file() else None,
        "backbone_checkpoint": BACKBONE_CHECKPOINT_PATH.is_file(),
        "loaded": runtime.loaded,
        "device": str(runtime.device).upper(),
        "window_seconds": WINDOW_SECONDS,
        "label_mapping": runtime.label_map,
        "spoof_class_index": runtime.spoof_class_index,
        "label_mapping_provisional": False,
        "uses_external_backbone": runtime.uses_external_backbone,
        "metadata": runtime.metadata,
    }


@app.post("/load")
async def load_model() -> dict:
    started = time.perf_counter()
    try:
        await asyncio.to_thread(runtime.load)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"loaded": True, "seconds": time.perf_counter() - started, "metadata": runtime.metadata}


@app.post("/infer/pcm")
async def infer_pcm(request: Request) -> dict:
    try:
        source_rate = int(request.headers.get("X-Sample-Rate", "48000"))
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid X-Sample-Rate header.") from error
    if not 8_000 <= source_rate <= 96_000:
        raise HTTPException(status_code=400, detail="Unsupported PCM sample rate.")
    raw_pcm = await request.body()
    try:
        waveform, rms = prepare_waveform(raw_pcm, source_rate)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if not math.isfinite(rms) or rms < 0.002:
        return {
            "state": "insufficient",
            "spoof_probability": None,
            "confidence": 0.0,
            "rms": rms,
            "label_mapping_provisional": False,
        }

    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(runtime.predict, waveform)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    latency_ms = (time.perf_counter() - started) * 1000
    spoof_probability = float(result.probabilities[runtime.spoof_class_index])
    if spoof_probability < 0.35:
        state = "safe"
    elif spoof_probability < 0.70:
        state = "verify"
    else:
        state = "high_risk"
    return {
        "state": state,
        "spoof_probability": spoof_probability,
        "confidence": max(spoof_probability, 1.0 - spoof_probability),
        "probabilities": result.probabilities,
        "logits": result.logits,
        "embedding_norm": result.embedding_norm,
        "latency_ms": latency_ms,
        "rms": rms,
        "label_mapping_provisional": False,
    }
