"""Live detector-service smoke test against the installed synthetic sample."""

from __future__ import annotations

from pathlib import Path

import httpx
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8001"


def run() -> None:
    samples, sample_rate = sf.read(ROOT / "artifacts" / "tts_test.wav", dtype="float32", always_2d=True)
    mono = samples.mean(axis=1).astype("<f4", copy=False)
    with httpx.Client(base_url=BASE_URL, timeout=45.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        health_payload = health.json()
        assert health_payload["checkpoint"] is True
        assert health_payload["checkpoint_name"] == "best_telephony_detector(2).pth"
        assert health_payload["spoof_class_index"] == 1
        assert health_payload["label_mapping_provisional"] is False
        response = client.post("/infer/pcm", content=mono.tobytes(), headers={"X-Sample-Rate": str(sample_rate)})
        response.raise_for_status()
        result = response.json()
        assert result["state"] == "high_risk"
        assert result["spoof_probability"] > 0.95

    print(
        "Detector smoke test passed: checkpoint service classified the installed synthetic sample "
        f"as High Risk ({result['spoof_probability']:.2%}) in {result['latency_ms']:.0f} ms."
    )


if __name__ == "__main__":
    run()
