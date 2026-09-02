# Realtime-Voice-Clone / VoxGuard

Detector training pipeline (`model/`) plus the full VoxGuard demo application below: a temporary-ID VoIP calling app, an authorized synthetic-voice attack console, a live risk dashboard, and the real-time detector/cloning services.

# VoxGuard VoIP MVP

A FastAPI-based temporary-ID WebRTC calling service with participant-separated live audio monitoring, an authorized synthetic-voice attack console, and a real-time voice-clone/deepfake detector.

## Services

- **`app/`** (port 8000) — the calling application: temporary-ID signaling, WebRTC offer/answer/ICE, per-participant PCM mirrors, the caller UI (`/`), the monitoring dashboard (`/dashboard`), the authorized attacker console (`/attacker`), and a standalone TTS page (`/tts`).
- **`detector_service/`** (port 8001) — the real-time voice-clone detector (WavLM + LoRA + raw-waveform SincConv branch + cross-attention fusion + graph-attention backend). Runs as an isolated FastAPI service so it can be hosted on a GPU box separate from the calling app.
- **`cloner_service/`** (port 8002) — a standalone Chatterbox-based voice-cloning service used by the attacker console to synthesize authorized clone audio from a short reference recording.

`model/` at the repo root is the separate training/evaluation pipeline used to produce the detector checkpoints (dataset build, training, benchmarking) — it is not required to run the live app below.

## Detector checkpoints (not included)

The trained detector weights are private and are **not** committed to this repo. To run `detector_service` locally, place these two files one directory above `voip-mvp/` (or point at them directly with the env vars below):

- `best_detector_fixed_v5.pth` — the main detector checkpoint (LoRA + custom heads).
- `best_telephony_detector.pth` — supplies the frozen WavLM backbone weights merged into the model at load time.

Override the locations with `VOXGUARD_CHECKPOINT_PATH` and `VOXGUARD_BACKBONE_CHECKPOINT_PATH` if you keep them elsewhere.

## Voice-clone demo audio

Three preloaded demo clips ship in [`demo-voice-clones/`](demo-voice-clones/) so the attacker console's "Voice clone" source and the dashboard demo work out of the box without a live cloning model. Point `VOXGUARD_VOICE_CLONE_DIR` at a different folder to swap the library.

## Quick start (Windows / PowerShell)

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
./download_tts_model.ps1
./setup_detector.ps1
./start.ps1
```

Or without `uv`, plain venv/pip:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
./download_tts_model.ps1
./start.ps1
```

Then open:

- `http://localhost:8000/` in two separate browsers or browser profiles
- `http://localhost:8000/dashboard` in a third tab or device
- `http://localhost:8000/attacker` for the authorized synthetic-audio caller
- `http://localhost:8000/tts` for the standalone TTS utility

Use headphones while tapping live audio from the dashboard to prevent feedback.

## Running the detector on another machine (GPU)

The detector is a plain FastAPI service, so it can run anywhere with the checkpoints available and be pointed at from the calling app:

```powershell
$env:VOXGUARD_DETECTOR_URL = 'http://<detector-host>:8001'
./start.ps1
```

`app/detector_bridge.py` reads `VOXGUARD_DETECTOR_URL` at startup; the detector service itself has no authentication, so only point it at a host you control. When this app is deployed and the detector endpoint changes, only that env var needs updating — no code changes.

## Authorized synthetic-attack demo

1. Open `/` and register the protected user.
2. Open `/attacker`, keep **Synthetic TTS** selected, and go online.
3. Generate a script, dial the protected user's temporary ID, and accept from the protected browser.
4. Select **Transmit into active call**. The generated WAV becomes the attacker's outbound WebRTC audio.
5. Open `/dashboard` to see the red-team session, tap Caller A, and trigger a verification response.

The **Voice clone** source accepts authorized local audio files, clips generated live via the cloning service, or the preloaded demo clips in `demo-voice-clones/`. The selected audio is decoded in the browser and becomes the attacker's actual outbound WebRTC track, so the same PCM reaches the call, monitoring tap, and detector.

If the live cloning service isn't running in a given deployment, the console falls back to the preloaded demo clips / local file upload — see the notice at the top of `/attacker`.

## Important prototype boundaries

- WebRTC is peer-to-peer and uses a public STUN server. It works reliably on the same machine or LAN. Internet-wide operation normally requires a TURN server.
- The monitoring stream is an explicit participant-side microphone mirror, not covert packet interception.
- Audio is forwarded in memory and is not written to disk.
- The MVP has no authentication. Add dashboard authorization before any non-demo deployment.
- Risk-state cutoffs are product-level demo thresholds and should be calibrated on deployment-domain validation data before production use.
