#!/usr/bin/env python3
"""
Moshi-style streaming processor with energy-based VAD and 80ms chunking
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
    def __init__(self, chunk_ms: int = 80, sr: int = 24000, overlap_ms: int = 10):
        self.sr = sr
        self.size = int(chunk_ms * sr / 1000)
        self.overlap = int(overlap_ms * sr / 1000)
        self.buf = torch.empty(0)

    def add(self, audio: torch.Tensor) -> List[torch.Tensor]:
        self.buf = torch.cat([self.buf, audio])
        out = []
        while len(self.buf) >= self.size:
            out.append(self.buf[: self.size])
            self.buf = self.buf[self.size - self.overlap :]
        return out

    def flush(self) -> Optional[torch.Tensor]:
        if len(self.buf) > 0:
            x = self.buf.clone()
            self.buf = torch.empty(0)
            return x
        return None


class StreamingProcessor:
    def __init__(self, model, speech_tokenizer, chunk_size_ms: int = 80, sample_rate: int = 24000, max_latency_ms: int = 200, vad_threshold: float = 0.01):
        self.model = model
        self.tok = speech_tokenizer
        self.chunker = Chunker(chunk_ms=chunk_size_ms, sr=sample_rate)
        self.vad = VAD(threshold=vad_threshold)
        self.user_hist = deque(maxlen=16)
        self.ai_hist = deque(maxlen=16)
        self.lat_hist = deque(maxlen=100)
        self.device = next(model.parameters()).device

    async def process_audio_stream(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        t0 = time.time()
        # ensure 1D tensor
        if audio.dim() > 1:
            audio = audio.view(-1)
        chunks = self.chunker.add(audio)
        for ch in chunks:
            if not self.vad(ch):
                continue
            # tokenize -> ids
            ids = self.tok.tokenize(ch.unsqueeze(0).to(self.device))  # [1,T]
            self.user_hist.append(ids)
            user_ctx = self._context(self.user_hist, 12)
            ai_ctx = self._context(self.ai_hist, 12)
            # generate new speech ids
            new_ids = self.model.generate_streaming(user_ctx, max_new_tokens=8)
            self.ai_hist.append(new_ids)
            # detokenize -> audio
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
        self.chunker.buf = torch.empty(0)
