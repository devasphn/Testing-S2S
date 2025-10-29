#!/usr/bin/env python3
"""
Serve both API, WS and the built-in /web UI.
"""
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np

from src.models import HybridS2SModel, SpeechTokenizer
from src.models.streaming_processor import StreamingProcessor
from src.api_config import router as api_router
from src.web_route import router as web_router

app = FastAPI(title="Testing-S2S Realtime Server", version="0.1.0")

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

@app.on_event("startup")
async def startup():
    global _model, _tok, _proc
    _tok = SpeechTokenizer().to(_device)
    _model = HybridS2SModel().to(_device).eval()
    _proc = StreamingProcessor(
        model=_model,
        speech_tokenizer=_tok,
        chunk_size_ms=80,
        sample_rate=24000,
        max_latency_ms=200,
        vad_threshold=0.01,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "device": _device}

@app.get("/api/stats")
async def stats():
    return {"latency_ms": _proc.get_latency_stats() if _proc else {}}

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            audio_i16 = np.frombuffer(data, dtype=np.int16)
            audio = torch.from_numpy(audio_i16.astype(np.float32) / 32768.0).to(_device)
            out = await _proc.process_audio_stream(audio)
            if out is not None:
                out_np = (out.detach().cpu().numpy() * 32768.0).astype(np.int16).tobytes()
                await ws.send_bytes(out_np)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True)
