#!/usr/bin/env python3
"""
Serve both API, WS and the built-in /web UI.
"""
import asyncio
from typing import Optional, Deque
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np

from src.models import HybridS2SModel, SpeechTokenizer
from src.models.streaming_processor import StreamingProcessor
from src.api_config import router as api_router
from src.web_route import router as web_router

# Strict determinism and safer backends
try:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except Exception:
    pass

app = FastAPI(title="Testing-S2S Realtime Server", version="0.1.4")

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
    fake = torch.zeros(int(TRANSPORT_SR * 0.08), dtype=torch.float32, device=_device)
    try:
        await _proc.process_audio_stream(fake)
    except Exception:
        pass

@app.on_event("startup")
async def startup():
    global _model, _tok, _proc
    _tok = SpeechTokenizer().to(_device)
    _model = HybridS2SModel().to(_device).eval()
    _proc = StreamingProcessor(
        model=_model,
        speech_tokenizer=_tok,
        chunk_size_ms=80,
        sample_rate=TRANSPORT_SR,
        max_latency_ms=200,
        vad_threshold=0.01,
    )
    # Warmup once to reduce first-turn latency/plan selection
    await _warmup_pipeline()

@app.get("/health")
async def health():
    return {"status": "ok", "device": _device}

@app.get("/api/stats")
async def stats():
    return {"latency_ms": _proc.get_latency_stats() if _proc else {}}

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    send_buffer: Deque[np.ndarray] = deque()
    sent_frames_total = 0
    try:
        while True:
            # Drain queued audio first, at realtime pace
            while send_buffer:
                frame_np = send_buffer.popleft()
                await ws.send_bytes(frame_np.tobytes())
                sent_frames_total += 1
                if sent_frames_total % 20 == 0:
                    print(f"[STREAM] Sent frames: {sent_frames_total}")
                await asyncio.sleep(FRAME_PACING_SEC)

            # Receive next input or ping
            msg = await ws.receive()
            if 'bytes' in msg and msg['bytes'] is not None:
                in_bytes = msg['bytes']
                audio_i16 = np.frombuffer(in_bytes, dtype=np.int16)
                print(f"[USER] Received {len(audio_i16)} samples")
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
                    print(f"[AI] Response samples: {total_samples} at {TRANSPORT_SR} Hz (~{total_samples/TRANSPORT_SR:.2f}s)")

                    # Segment into 20ms frames and enqueue
                    total = total_samples
                    start = 0
                    queued = 0
                    while start < total:
                        end = min(start + FRAME_SAMPLES, total)
                        frame = out_cpu[start:end]
                        if frame.numel() < FRAME_SAMPLES:
                            frame = torch.nn.functional.pad(frame, (0, FRAME_SAMPLES - frame.numel()))
                        frame_i16 = (frame.numpy() * 32767.0).astype(np.int16)
                        send_buffer.append(frame_i16)
                        start = end
                        queued += 1
                    print(f"[STREAM] Queued frames: {queued} (frame_ms={FRAME_MS})")

            await asyncio.sleep(0)
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True)
