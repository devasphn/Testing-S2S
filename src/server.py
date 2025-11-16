#!/usr/bin/env python3
"""
Serve both API, WS and the built-in /web UI with enhanced WebSocket logging.
"""
import asyncio
from typing import Optional, Deque
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, WebSocketException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torchaudio.functional as AF
import numpy as np
import traceback
import os

from pathlib import Path

from src.models import HybridS2SModel, SpeechTokenizer
from src.models.streaming_processor import StreamingProcessor
from src.models.deterministic_responder import DeterministicResponder
from src.api_config import router as api_router
from src.web_route import router as web_router

# Safer backends (avoid deterministic crash); keep cuDNN deterministic only
try:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except Exception:
    pass

# Get reply mode
REPLY_MODE = os.getenv("REPLY_MODE", "stream").lower()
DETERMINISTIC_POC = os.getenv("DETERMINISTIC_POC", "0").lower() in {"1", "true", "yes", "on"}
DET_METADATA = os.getenv("DETERMINISTIC_METADATA", "wav_files/metadata.json")
DET_WAV_ROOT = os.getenv("DETERMINISTIC_WAV_ROOT", "wav_files")
DET_MIN_SCORE = float(os.getenv("DETERMINISTIC_MIN_SCORE", "0.75"))

app = FastAPI(title="Testing-S2S Realtime Server", version="0.1.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
app.include_router(web_router)

_model: Optional[HybridS2SModel] = None
_tok: Optional[SpeechTokenizer] = None
_proc: Optional[StreamingProcessor] = None
_det_responder: Optional[DeterministicResponder] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Transport format
TRANSPORT_SR = 24000
FRAME_MS = 20  # 20ms frames (480 samples @ 24k)
FRAME_SAMPLES = int(TRANSPORT_SR * FRAME_MS / 1000)
FRAME_PACING_SEC = FRAME_MS / 1000.0

# Checkpoint paths
TOKENIZER_CKPT = os.getenv("TOKENIZER_CKPT", "checkpoints/tokenizer/speech_tokenizer_telugu_epoch0200.pth")
MODEL_CKPT = os.getenv("HYBRID_S2S_CKPT", "checkpoints/hybrid_s2s/hybrid_s2s_telugu.pth")
SPEAKER_EMB = os.getenv("SPEAKER_EMB", "data/speaker/speaker_embedding.pt")

# Connection tracking
active_connections = set()


def _load_state_dict(module: torch.nn.Module, path: str, key: str = "state_dict"):
    if not path or not os.path.exists(path):
        print(f"[WARN] Checkpoint not found at {path} - using randomly initialized {module.__class__.__name__}")
        return
    ckpt = torch.load(path, map_location=_device)
    state = ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt
    module.load_state_dict(state)
    print(f"[INFO] Loaded {module.__class__.__name__} weights from {path}")


def _load_speaker_embedding(path: str) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path):
        print(f"[WARN] Speaker embedding not found at {path}; using default gain")
        return None
    data = torch.load(path, map_location="cpu")
    emb = data.get("embedding")
    if emb is None:
        print(f"[WARN] Speaker embedding file {path} missing 'embedding' key")
        return None
    print(f"[INFO] Loaded speaker embedding from {path} (backend={data.get('encoder_backend', 'unknown')})")
    return emb

# Resampling helper (22.05 kHz -> 24 kHz)
def _resample_hq(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    wav = wav.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    resampled = AF.resample(
        wav,
        orig_freq=src_sr,
        new_freq=dst_sr,
        lowpass_filter_width=64,
        rolloff=0.99,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    return resampled.squeeze(0)

# Soft limiter to avoid clipping
def _limit(x: torch.Tensor, thresh: float = 0.98) -> torch.Tensor:
    return torch.tanh(x / thresh) * thresh

async def _warmup_pipeline():
    """Run a tiny warmup through tokenizer/model to prebuild kernels."""
    global _proc
    if _proc is None:
        return
    print("[INFO] Running pipeline warmup...")
    fake = torch.zeros(int(TRANSPORT_SR * 0.08), dtype=torch.float32, device=_device)
    try:
        await _proc.process_audio_stream(fake)
        print("[INFO] Pipeline warmup completed")
    except Exception as e:
        print(f"[WARN] Pipeline warmup failed: {e}")

@app.on_event("startup")
async def startup():
    global _model, _tok, _proc, _det_responder
    print(f"[INFO] Starting server on device: {_device} with REPLY_MODE={REPLY_MODE}")

    _det_responder = None
    if DETERMINISTIC_POC:
        try:
            _det_responder = DeterministicResponder(
                metadata_path=Path(DET_METADATA),
                wav_root=Path(DET_WAV_ROOT),
                transport_sr=TRANSPORT_SR,
                min_score=DET_MIN_SCORE,
            )
            print("[INFO] Deterministic POC mode enabled - streaming prerecorded Telugu replies")
        except Exception as exc:
            print(f"[ERROR] Failed to initialize DeterministicResponder: {exc}")
            raise

    _tok = SpeechTokenizer().to(_device)
    _load_state_dict(_tok, TOKENIZER_CKPT)
    _model = HybridS2SModel().to(_device)
    _load_state_dict(_model, MODEL_CKPT)
    _model.eval()
    speaker_embedding = _load_speaker_embedding(SPEAKER_EMB)
    _proc = StreamingProcessor(
        model=_model,
        speech_tokenizer=_tok,
        chunk_size_ms=80,
        sample_rate=TRANSPORT_SR,
        max_latency_ms=200,
        vad_threshold=0.65,  # FIXED: Changed from 0.001 to 0.65 to prevent always-on audio
        reply_mode=REPLY_MODE,
        speaker_embedding=speaker_embedding,
        deterministic_responder=_det_responder,
    )
    # Warmup once to reduce first-turn latency/plan selection
    if _proc and not _proc.deterministic:
        await _warmup_pipeline()
    print(f"[INFO] Server startup complete - WebSocket endpoint ready at /ws/stream (mode: {REPLY_MODE})")

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "device": _device, 
        "active_connections": len(active_connections),
        "reply_mode": REPLY_MODE
    }

@app.get("/api/stats")
async def stats():
    return {
        "latency_ms": _proc.get_latency_stats() if _proc else {},
        "connections": len(active_connections)
    }

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"[WS] Connection attempt from {client_id}")
    
    try:
        await websocket.accept()
        active_connections.add(websocket)
        print(f"[WS] ✅ Connection accepted: {client_id} (total: {len(active_connections)}, mode: {REPLY_MODE})")
        
        send_buffer: Deque[np.ndarray] = deque()
        sent_frames_total = 0
        received_chunks = 0
        
        while True:
            # Drain queued audio first, at realtime pace
            while send_buffer:
                frame_np = send_buffer.popleft()
                await websocket.send_bytes(frame_np.tobytes())
                sent_frames_total += 1
                
                # Log based on mode
                if REPLY_MODE == "turn":
                    if sent_frames_total % 25 == 0:  # Every 0.5s in turn mode
                        print(f"[TURN] 🔊 Sent {sent_frames_total} frames to {client_id}")
                else:
                    if sent_frames_total % 50 == 0:  # Every 1s in stream mode
                        print(f"[STREAM] 🔊 Sent {sent_frames_total} frames to {client_id}")
                        
                await asyncio.sleep(FRAME_PACING_SEC)

            # Receive next input
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"[WS] ❌ Receive error from {client_id}: {e}")
                break
                
            if 'bytes' in msg and msg['bytes'] is not None:
                in_bytes = msg['bytes']
                audio_i16 = np.frombuffer(in_bytes, dtype=np.int16)
                received_chunks += 1
                
                # Log received audio less frequently
                if received_chunks % 100 == 0:
                    print(f"[USER] 🎤 Received {received_chunks} audio chunks from {client_id}")
                
                audio = torch.from_numpy(audio_i16.astype(np.float32) / 32767.0).to(_device)

                out = await _proc.process_audio_stream(audio)
                if out is not None:
                    if out.dim() > 1:
                        out = out.view(-1)
                    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                    out = torch.clamp(out, -1.0, 1.0)
                    out = _limit(out, 0.98)

                    src_sr = getattr(_tok.vocoder, 'sample_rate', TRANSPORT_SR) if _tok else TRANSPORT_SR
                    out_cpu = out.detach().cpu()
                    if src_sr != TRANSPORT_SR:
                        out_cpu = _resample_hq(out_cpu, src_sr, TRANSPORT_SR)
                    total_samples = out_cpu.numel()
                    duration = total_samples / TRANSPORT_SR
                    
                    # Log based on mode
                    if REPLY_MODE == "turn":
                        print(f"[TURN] 🤖 Generated response: {total_samples} samples ({duration:.2f}s) for {client_id}")
                    else:
                        print(f"[STREAM] 🤖 Generated response: {total_samples} samples ({duration:.2f}s) for {client_id}")

                    # Segment into 20ms frames and enqueue
                    start = 0
                    queued = 0
                    while start < total_samples:
                        end = min(start + FRAME_SAMPLES, total_samples)
                        frame = out_cpu[start:end]
                        if frame.numel() < FRAME_SAMPLES:
                            frame = torch.nn.functional.pad(frame, (0, FRAME_SAMPLES - frame.numel()))
                        frame_i16 = (frame.numpy() * 32767.0).astype(np.int16)
                        send_buffer.append(frame_i16)
                        start = end
                        queued += 1
                    
                    # Log based on mode
                    if REPLY_MODE == "turn":
                        print(f"[TURN] 📦 Queued {queued} frames ({duration:.2f}s) for {client_id}")
                    else:
                        print(f"[STREAM] 📦 Queued {queued} frames ({duration:.2f}s) for {client_id}")

            await asyncio.sleep(0.001)  # Small yield to prevent blocking
            
    except WebSocketDisconnect:
        print(f"[WS] 🔌 Client {client_id} disconnected normally")
    except Exception as e:
        print(f"[WS] ❌ Unexpected error with {client_id}: {e}")
        print(f"[WS] 🔍 Traceback: {traceback.format_exc()}")
    finally:
        active_connections.discard(websocket)
        print(f"[WS] 🧹 Cleaned up connection {client_id} (remaining: {len(active_connections)})")

if __name__ == "__main__":
    print(f"[INFO] 🚀 Starting Testing-S2S server on http://0.0.0.0:8000")
    print(f"[INFO] 🌍 Web UI available at: http://0.0.0.0:8000/web")
    print(f"[INFO] 🔌 WebSocket endpoint: ws://0.0.0.0:8000/ws/stream")
    print(f"[INFO] 🎯 Mode: {REPLY_MODE.upper()}")
    uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True, log_level="info")
