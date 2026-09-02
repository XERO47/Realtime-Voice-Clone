"""Probe the live checkpoint across voices and simple telephony degradations."""

from __future__ import annotations

from io import BytesIO

import httpx
import numpy as np
import soundfile as sf


APP_URL = "http://127.0.0.1:8000"
DETECTOR_URL = "http://127.0.0.1:8001"
SCRIPT = (
    "This is an authorized voice cloning detection test over a simulated telephone call. "
    "Please verify my identity before approving any sensitive request."
)
VOICES = ("af_sarah", "am_adam", "bf_emma")


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    target_size = max(1, round(len(samples) * target_rate / source_rate))
    positions = np.linspace(0, len(samples) - 1, target_size)
    return np.interp(positions, np.arange(len(samples)), samples).astype("float32")


def add_noise(samples: np.ndarray, snr_db: float, seed: int = 7) -> np.ndarray:
    signal_power = float(np.mean(samples.astype("float64") ** 2))
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(seed).normal(0, noise_power**0.5, len(samples))
    return np.clip(samples + noise, -1, 1).astype("float32")


def infer(client: httpx.Client, samples: np.ndarray, sample_rate: int) -> dict:
    response = client.post(
        f"{DETECTOR_URL}/infer/pcm",
        content=samples.astype("<f4", copy=False).tobytes(),
        headers={"X-Sample-Rate": str(sample_rate)},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run() -> None:
    rows: list[tuple[str, str, float | None, str]] = []
    with httpx.Client(timeout=60) as client:
        for voice in VOICES:
            speech = client.post(
                f"{APP_URL}/api/tts/generate",
                json={"text": SCRIPT, "voice": voice, "speed": 1.0},
            )
            speech.raise_for_status()
            samples, sample_rate = sf.read(BytesIO(speech.content), dtype="float32", always_2d=True)
            mono = samples.mean(axis=1)
            telephone = resample_linear(mono, sample_rate, 8_000)
            variants = (
                ("clean", mono, sample_rate),
                ("8khz", telephone, 8_000),
                ("8khz+10dB", add_noise(telephone, 10), 8_000),
            )
            for variant, audio, rate in variants:
                result = infer(client, audio, rate)
                rows.append((voice, variant, result.get("spoof_probability"), result["state"]))

    print(f"{'VOICE':<12} {'VARIANT':<12} {'SPOOF':>9}  STATE")
    for voice, variant, probability, state in rows:
        score = "--" if probability is None else f"{probability:8.3%}"
        print(f"{voice:<12} {variant:<12} {score:>9}  {state}")


if __name__ == "__main__":
    run()
