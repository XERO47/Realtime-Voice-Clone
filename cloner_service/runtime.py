from __future__ import annotations

import copy
import gc
import io
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "models" / "numba-cache"))

import librosa
import numpy as np
import soundfile as sf
import torch


@dataclass
class VoiceProfile:
    id: str
    name: str
    path: Path
    original_name: str
    bytes: int
    duration_seconds: float
    created_at: float = field(default_factory=time.time)
    conditionals: Any | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "original_name": self.original_name,
            "bytes": self.bytes,
            "duration_seconds": round(self.duration_seconds, 2),
            "created_at": self.created_at,
            "cached": self.conditionals is not None,
        }


class CloneRuntime:
    """Owns one non-thread-safe model and its in-memory speaker conditionings."""

    MODEL_NAME = "Chatterbox Nano"
    MODEL_PARAMETERS = "110M"
    SAMPLE_RATE = 24_000

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.reference_dir = data_dir / "references"
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any | None = None
        self._device: str | None = None
        self._voices: dict[str, VoiceProfile] = {}
        self._op_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._busy_operation: str | None = None
        self._last_error: str | None = None
        self._loaded_at: float | None = None
        self._last_generation: dict[str, Any] | None = None

    def _set_busy(self, operation: str | None) -> None:
        with self._state_lock:
            self._busy_operation = operation

    def _resolve_device(self, requested: str) -> str:
        requested = requested.lower().strip()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu, or cuda.")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access a GPU.")
            return "cuda"
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            busy = self._busy_operation
            error = self._last_error
        gpu_name = None
        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_allocated_mb = torch.cuda.memory_allocated(0) / 1024**2
                gpu_reserved_mb = torch.cuda.memory_reserved(0) / 1024**2
            except Exception:
                gpu_name = "CUDA device"
        return {
            "service": "voxguard-cloner",
            "model": self.MODEL_NAME,
            "parameters": self.MODEL_PARAMETERS,
            "loaded": self._model is not None,
            "device": self._device,
            "busy": busy is not None,
            "operation": busy,
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "gpu_name": gpu_name,
            "gpu_allocated_mb": round(gpu_allocated_mb, 1),
            "gpu_reserved_mb": round(gpu_reserved_mb, 1),
            "torch_version": torch.__version__,
            "voices": len(self._voices),
            "cached_voices": sum(v.conditionals is not None for v in self._voices.values()),
            "loaded_at": self._loaded_at,
            "last_generation": self._last_generation,
            "last_error": error,
        }

    def load(self, requested_device: str = "auto") -> dict[str, Any]:
        device = self._resolve_device(requested_device)
        with self._op_lock:
            if self._model is not None and self._device == device:
                return self.status()
            self._set_busy(f"Loading {self.MODEL_NAME} on {device.upper()}")
            self._last_error = None
            try:
                if self._model is not None:
                    self._unload_locked(clear_voice_cache=True)
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                self._model = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
                self._device = device
                self._loaded_at = time.time()
                return self.status()
            except Exception as exc:
                self._model = None
                self._device = None
                self._last_error = str(exc)
                raise
            finally:
                self._set_busy(None)

    def _unload_locked(self, clear_voice_cache: bool = True) -> None:
        self._model = None
        self._device = None
        self._loaded_at = None
        if clear_voice_cache:
            for voice in self._voices.values():
                voice.conditionals = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> dict[str, Any]:
        with self._op_lock:
            self._set_busy("Unloading model")
            try:
                self._unload_locked(clear_voice_cache=True)
                return self.status()
            finally:
                self._set_busy(None)

    @staticmethod
    def _safe_stem(name: str) -> str:
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(name).stem).strip("-")
        return (stem or "reference")[:48]

    def add_voice(self, name: str, original_name: str, content: bytes) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Load the model before encoding a reference voice.")
        if not content:
            raise ValueError("Reference audio is empty.")
        if len(content) > 25 * 1024 * 1024:
            raise ValueError("Reference audio must be smaller than 25 MB.")

        suffix = Path(original_name).suffix.lower()
        if suffix not in {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".webm"}:
            raise ValueError("Use WAV, FLAC, OGG, MP3, M4A, AAC, or WebM reference audio.")
        voice_id = uuid.uuid4().hex[:12]
        path = self.reference_dir / f"{voice_id}-{self._safe_stem(original_name)}{suffix}"
        path.write_bytes(content)

        try:
            duration = float(librosa.get_duration(path=str(path)))
            if duration < 5.1:
                raise ValueError("Reference audio must contain at least 5 seconds of clear speech.")
            with self._op_lock:
                self._set_busy(f"Encoding voice: {name}")
                with torch.inference_mode():
                    self._model.prepare_conditionals(str(path), norm_loudness=True)
                    conditionals = copy.deepcopy(self._model.conds)
            profile = VoiceProfile(
                id=voice_id,
                name=(name.strip() or Path(original_name).stem)[:60],
                path=path,
                original_name=original_name,
                bytes=len(content),
                duration_seconds=duration,
                conditionals=conditionals,
            )
            self._voices[voice_id] = profile
            return profile.public()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            self._set_busy(None)

    def list_voices(self) -> list[dict[str, Any]]:
        return [voice.public() for voice in sorted(self._voices.values(), key=lambda item: item.created_at)]

    def delete_voice(self, voice_id: str) -> None:
        with self._op_lock:
            voice = self._voices.pop(voice_id, None)
            if voice is None:
                raise KeyError(voice_id)
            voice.path.unlink(missing_ok=True)
            del voice
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def clear_voice_cache(self) -> dict[str, Any]:
        with self._op_lock:
            for voice in self._voices.values():
                voice.conditionals = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self.status()

    def _ensure_voice_cached(self, voice: VoiceProfile) -> None:
        if voice.conditionals is None:
            with torch.inference_mode():
                self._model.prepare_conditionals(str(voice.path), norm_loudness=True)
                voice.conditionals = copy.deepcopy(self._model.conds)

    def generate(
        self,
        voice_id: str,
        text: str,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: int,
    ) -> tuple[bytes, dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("Load the model before generating speech.")
        voice = self._voices.get(voice_id)
        if voice is None:
            raise KeyError(voice_id)
        clean_text = " ".join(text.split())
        if len(clean_text) < 12 or len(clean_text.split()) < 3:
            raise ValueError("Enter at least three words for stable generation.")
        if len(clean_text) > 600:
            raise ValueError("Text must be 600 characters or fewer.")

        with self._op_lock:
            self._set_busy(f"Generating with {voice.name}")
            started = time.perf_counter()
            try:
                self._ensure_voice_cached(voice)
                self._model.conds = voice.conditionals
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    wav = self._model.generate(
                        clean_text,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )
                samples = wav.squeeze().detach().float().cpu().numpy()
                samples = np.clip(samples, -1.0, 1.0)
                buffer = io.BytesIO()
                sf.write(buffer, samples, self._model.sr, format="WAV", subtype="PCM_16")
                payload = buffer.getvalue()
                elapsed = time.perf_counter() - started
                audio_duration = len(samples) / self._model.sr
                metadata = {
                    "voice_id": voice.id,
                    "voice_name": voice.name,
                    "model": self.MODEL_NAME,
                    "device": self._device,
                    "generation_seconds": round(elapsed, 3),
                    "audio_seconds": round(audio_duration, 3),
                    "real_time_factor": round(elapsed / max(audio_duration, 0.001), 3),
                    "sample_rate": self._model.sr,
                    "seed": seed,
                    "bytes": len(payload),
                }
                self._last_generation = metadata
                return payload, metadata
            except Exception as exc:
                self._last_error = str(exc)
                raise
            finally:
                self._set_busy(None)

