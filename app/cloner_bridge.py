from __future__ import annotations

import json
import os

import httpx


CLONER_URL = os.getenv("VOXGUARD_CLONER_URL", "http://127.0.0.1:8002").rstrip("/")
# CPU generation runs at ~4-5x real time; a max-length (600 char) script can take minutes.
GENERATE_TIMEOUT = httpx.Timeout(600.0, connect=5.0)


class ClonerBridge:
    async def status(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{CLONER_URL}/api/status")
                response.raise_for_status()
                return {"connected": True, "url": CLONER_URL, **response.json()}
        except Exception as error:
            return {"connected": False, "url": CLONER_URL, "detail": str(error)}

    async def voices(self) -> dict:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{CLONER_URL}/api/voices")
            response.raise_for_status()
            return response.json()

    async def generate(self, payload: dict) -> tuple[bytes, dict]:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(f"{CLONER_URL}/api/generate", json=payload)
            response.raise_for_status()
            metadata = {}
            header = response.headers.get("X-Clone-Metadata")
            if header:
                metadata = json.loads(header)
            return response.content, metadata


cloner_bridge = ClonerBridge()
