from __future__ import annotations

import io
import wave

import httpx


BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/api/tts/status")
        status.raise_for_status()
        assert status.json()["installed"] is True

        voices_response = client.get("/api/tts/voices")
        voices_response.raise_for_status()
        voices = voices_response.json()["voices"]
        assert "af_sarah" in voices

        response = client.post(
            "/api/tts/generate",
            json={"text": "Authorized synthetic voice test.", "voice": "af_sarah", "speed": 1.0},
        )
        response.raise_for_status()
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.headers["x-tts-voice"] == "af_sarah"

        with wave.open(io.BytesIO(response.content), "rb") as wav:
            assert wav.getframerate() == 24000
            assert wav.getnchannels() == 1
            duration = wav.getnframes() / wav.getframerate()
            assert duration > 0.5

    print(
        "TTS smoke test passed: model status, voice discovery, HTTP generation, "
        f"and WAV validation ({duration:.2f}s)."
    )


if __name__ == "__main__":
    main()
