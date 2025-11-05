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
import numpy as np
import traceback
import os

from src.models import HybridS2SModel, SpeechTokenizer
from src.models.streaming_processor import StreamingProcessor
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
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Transport format
TRANSPORT_SR = 24000
FRAME_MS = 20  # 20ms frames (480 samples @ 24k)
FRAME_SAMPLES = int(TRANSPORT_SR * FRAME_MS / 1000)
FRAME_PACING_SEC = FRAME_MS / 1000.0

# Connection tracking
active_connections = set()

# Resampling helper (22.05 kHz -> 24 kHz)
def _resample_linear(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    wav = wav.detach().cpu().contiguous().view(-1)
    n = wav.numel()
    if n <= 1:
        m = max(1, int(round(dst_sr / max(1, src_sr))))
        return torch.zeros(m, dtype=wav.dtype)
    dur = n / float(src_sr)
    m = max(1, int(round(dur * dst_sr)))
    x = torch.linspace(0, n - 1, steps=m)
    x0 = torch.clamp(x.floor().long(), 0, n - 2)
    x1 = x0 + 1
    frac = (x - x0.float())
    y = wav[x0] * (1.0 - frac) + wav[x1] * frac
    return y

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
    global _model, _tok, _proc
    print(f"[INFO] Starting server on device: {_device} with REPLY_MODE={REPLY_MODE}")
    _tok = SpeechTokenizer().to(_device)
    _model = HybridS2SModel().to(_device).eval()
    _proc = StreamingProcessor(
        model=_model,
        speech_tokenizer=_tok,
        chunk_size_ms=80,
        sample_rate=TRANSPORT_SR,
        max_latency_ms=200,
        vad_threshold=0.001,  # lower threshold to detect speech more easily
        reply_mode=REPLY_MODE
    )
    # Warmup once to reduce first-turn latency/plan selection
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
                        out_cpu = _resample_linear(out_cpu, src_sr, TRANSPORT_SR)
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
