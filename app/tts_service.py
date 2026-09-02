from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tts" / "kokoro-v1.0.onnx"
VOICES_PATH = ROOT / "models" / "tts" / "voices-v1.0.bin"


@dataclass
class GeneratedSpeech:
    wav_bytes: bytes
    sample_rate: int
    duration_seconds: float
    generation_seconds: float
    voice: str


class TTSService:
    def __init__(self) -> None:
        self._model = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @property
    def installed(self) -> bool:
        return MODEL_PATH.is_file() and VOICES_PATH.is_file()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is not None:
            return self._model
        if not self.installed:
            raise RuntimeError("Kokoro model files are not installed. Run download_tts_model.ps1.")
        with self._load_lock:
            if self._model is None:
                from kokoro_onnx import Kokoro

                self._model = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
        return self._model

    def voices(self) -> list[str]:
        return list(self.load().get_voices())

    def generate(self, text: str, voice: str, speed: float) -> GeneratedSpeech:
        model = self.load()
        voices = model.get_voices()
        if voice not in voices:
            raise ValueError(f"Unknown voice: {voice}")
        started = time.perf_counter()
        with self._generation_lock:
            samples, sample_rate = model.create(
                text,
                voice=voice,
                speed=speed,
                lang="en-us",
                trim=True,
            )
        generation_seconds = time.perf_counter() - started
        duration_seconds = len(samples) / sample_rate
        output = io.BytesIO()
        sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
        return GeneratedSpeech(
            wav_bytes=output.getvalue(),
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            generation_seconds=generation_seconds,
            voice=voice,
        )


tts_service = TTSService()
