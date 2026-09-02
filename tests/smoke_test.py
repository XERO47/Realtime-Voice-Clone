"""End-to-end protocol smoke test for the FastAPI MVP."""

import os
import struct
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app


def next_type(socket, expected: str) -> dict:
    for _ in range(12):
        message = socket.receive_json()
        if message.get("type") == expected:
            return message
    raise AssertionError(f"Did not receive event type {expected!r}")


def run() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/").status_code == 200
        assert client.get("/dashboard").status_code == 200
        assert client.get("/attacker").status_code == 200
        clone_library = client.get("/api/voice-clones")
        assert clone_library.status_code == 200
        assert "clips" in clone_library.json()

        with client.websocket_connect("/ws/signaling/ALPHA-1001?role=attacker") as caller_a:
            assert next_type(caller_a, "registered")["user_id"] == "ALPHA-1001"
            with client.websocket_connect("/ws/signaling/BETA-2002") as caller_b:
                assert next_type(caller_b, "registered")["user_id"] == "BETA-2002"
                with client.websocket_connect("/ws/monitor") as monitor:
                    next_type(monitor, "sessions")
                    caller_a.send_json({"type": "call-request", "target": "BETA-2002"})
                    ringing = next_type(caller_a, "call-ringing")
                    incoming = next_type(caller_b, "incoming-call")
                    session_id = ringing["session"]["session_id"]
                    assert incoming["session"]["session_id"] == session_id

                    session_update = next_type(monitor, "sessions")
                    assert session_update["sessions"][0]["caller_a"] == "ALPHA-1001"
                    assert session_update["sessions"][0]["caller_b"] == "BETA-2002"
                    assert session_update["sessions"][0]["caller_a_role"] == "attacker"

                    caller_b.send_json({"type": "call-accept", "session_id": session_id})
                    assert next_type(caller_a, "call-accepted")["session"]["status"] == "connecting"

                    with client.websocket_connect(f"/ws/tap/{session_id}/ALPHA-1001") as tap:
                        assert tap.receive_json()["type"] == "tap-ready"
                        with client.websocket_connect(f"/ws/audio/{session_id}/ALPHA-1001?sample_rate=48000") as producer:
                            samples = [0.1, -0.1, 0.2, -0.2] * 512
                            frame = struct.pack(f"<{len(samples)}f", *samples)
                            producer.send_bytes(frame)
                            assert tap.receive_bytes() == frame

                    caller_a.send_json({"type": "call-end", "session_id": session_id})
                    assert next_type(caller_b, "call-ended")["session_id"] == session_id

    print("Smoke test passed: pages, clone library, signaling, session discovery, PCM audio tap, and call teardown.")


if __name__ == "__main__":
    run()
