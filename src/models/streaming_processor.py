#!/usr/bin/env python3
"""
Streaming Processor - Performance tweaks
- Ensure device consistency
- Reduce max_new_tokens per step
- Batch detokenize if needed (MVP keeps single stream)
- Avoid excessive history growth
"""
from collections import deque
from typing import Optional, Dict, List
import time
import torch
import torch.nn as nn


class VAD(nn.Module):
    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold
        self.hist = deque(maxlen=10)

    def forward(self, audio: torch.Tensor) -> bool:
        energy = torch.mean(audio**2).item()
        self.hist.append(energy)
        thr = max(self.threshold, (sum(self.hist) / len(self.hist)) * 0.1 if self.hist else self.threshold)
        return energy > thr


class Chunker:
    def __init__(self, chunk_ms: int = 80, sr: int = 24000, overlap_ms: int = 10, device: str = "cpu"):
        self.sr = sr
        self.size = int(chunk_ms * sr / 1000)
        self.overlap = int(overlap_ms * sr / 1000)
        self.device = torch.device(device)
        self.buf = torch.empty(0, device=self.device)

    def to(self, device: torch.device):
        self.device = torch.device(device)
        if self.buf.device != self.device:
            self.buf = self.buf.to(self.device)
        return self

    def add(self, audio: torch.Tensor) -> List[torch.Tensor]:
        if audio.device != self.device:
            audio = audio.to(self.device)
        self.buf = torch.cat([self.buf, audio])
        out = []
        # produce at most 1 chunk per call to keep latency low
        if len(self.buf) >= self.size:
            out.append(self.buf[: self.size].clone())
            self.buf = self.buf[self.size - self.overlap :]
        return out

    def flush(self) -> Optional[torch.Tensor]:
        if len(self.buf) > 0:
            x = self.buf.clone()
            self.buf = torch.empty(0, device=self.device)
            return x
        return None


class StreamingProcessor:
    def __init__(self, model, speech_tokenizer, chunk_size_ms: int = 80, sample_rate: int = 24000, max_latency_ms: int = 200, vad_threshold: float = 0.01):
        self.model = model
        self.tok = speech_tokenizer
        self.device = next(model.parameters()).device
        self.chunker = Chunker(chunk_ms=chunk_size_ms, sr=sample_rate, device=self.device)
        self.vad = VAD(threshold=vad_threshold)
        self.user_hist = deque(maxlen=8)   # smaller history
        self.ai_hist = deque(maxlen=8)
        self.lat_hist = deque(maxlen=200)

    async def process_audio_stream(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        t0 = time.time()
        if audio.dim() > 1:
            audio = audio.view(-1)
        audio = audio.to(self.device)
        self.chunker.to(self.device)

        chunks = self.chunker.add(audio)
        for ch in chunks:
            if not self.vad(ch):
                continue
            with torch.no_grad():
                ids = self.tok.tokenize(ch.unsqueeze(0).to(self.device))
            self.user_hist.append(ids)
            user_ctx = self._context(self.user_hist, 8)
            # generate fewer tokens per step for lower latency
            with torch.no_grad():
                new_ids = self.model.generate_streaming(user_ctx, max_new_tokens=4, temperature=0.95)
                self.ai_hist.append(new_ids)
                out_audio = self.tok.detokenize(new_ids)
            self.lat_hist.append((time.time() - t0) * 1000.0)
            return out_audio.squeeze(0)
        return None

    def _context(self, dq: deque, L: int) -> torch.Tensor:
        if not dq:
            return torch.zeros(1, L, dtype=torch.long, device=self.device)
        cat = torch.cat(list(dq), dim=1)
        if cat.size(1) >= L:
            return cat[:, -L:]
        pad = torch.zeros(1, L - cat.size(1), dtype=torch.long, device=self.device)
        return torch.cat([pad, cat], dim=1)

    def get_latency_stats(self) -> Dict[str, float]:
        if not self.lat_hist:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
        arr = sorted(list(self.lat_hist))
        n = len(arr)
        return {"mean": sum(arr)/n, "min": arr[0], "max": arr[-1], "p95": arr[int(0.95*n)-1] if n>1 else arr[0]}

    def reset(self):
        self.user_hist.clear()
        self.ai_hist.clear()
        self.lat_hist.clear()
        self.chunker.buf = torch.empty(0, device=self.device)
