#!/usr/bin/env python3
"""
Streaming Processor - Performance tweaks + Turn-based Mode
- Ensure device consistency
- Reduce max_new_tokens per step
- Batch detokenize if needed (MVP keeps single stream)
- Avoid excessive history growth
- REPLY_MODE=turn: VAD-based turn-taking with silence detection
"""
from collections import deque
from typing import Optional, Dict, List
import time
import os
import torch
import torch.nn as nn


class VAD(nn.Module):
    def __init__(self, threshold: float = 0.01, silence_frames: int = 30):
        super().__init__()
        self.threshold = threshold
        self.silence_frames = silence_frames  # Frames of silence to detect turn end
        self.hist = deque(maxlen=10)
        self.silence_count = 0
        self.speaking = False
        
    def forward(self, audio: torch.Tensor) -> Dict[str, bool]:
        energy = torch.mean(audio**2).item()
        self.hist.append(energy)
        
        # Adaptive threshold
        avg_energy = sum(self.hist) / len(self.hist) if self.hist else self.threshold
        dynamic_threshold = max(self.threshold, avg_energy * 0.1)
        
        is_voice = energy > dynamic_threshold
        
        # Track speaking state for turn detection
        if is_voice:
            self.silence_count = 0
            if not self.speaking:
                self.speaking = True
        else:
            self.silence_count += 1
            
        # Detect end of turn (sufficient silence after speaking)
        turn_ended = (self.speaking and 
                     self.silence_count >= self.silence_frames)
        
        if turn_ended:
            self.speaking = False
            self.silence_count = 0
            
        return {
            'is_voice': is_voice,
            'turn_ended': turn_ended,
            'speaking': self.speaking
        }
        
    def reset(self):
        """Reset VAD state for new conversation."""
        self.silence_count = 0
        self.speaking = False
        self.hist.clear()


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
    def __init__(self, model, speech_tokenizer, chunk_size_ms: int = 80, sample_rate: int = 24000, 
                 max_latency_ms: int = 200, vad_threshold: float = 0.01, reply_mode: str = "stream"):
        self.model = model
        self.tok = speech_tokenizer
        self.device = next(model.parameters()).device
        self.chunker = Chunker(chunk_ms=chunk_size_ms, sr=sample_rate, device=self.device)
        
        # Reply mode configuration
        self.reply_mode = os.getenv("REPLY_MODE", reply_mode).lower()
        
        # VAD with turn detection for reply_mode=turn
        silence_frames = 30 if self.reply_mode == "turn" else 5
        self.vad = VAD(threshold=vad_threshold, silence_frames=silence_frames)
        
        self.user_hist = deque(maxlen=8)   # smaller history
        self.ai_hist = deque(maxlen=8)
        self.lat_hist = deque(maxlen=200)
        
        # Turn-based state
        self.turn_buffer = []  # Accumulate user speech during turn
        self.generating_response = False
        self.response_generated = False
        
        print(f"[INFO] StreamingProcessor initialized with REPLY_MODE={self.reply_mode}")

    async def process_audio_stream(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        t0 = time.time()
        if audio.dim() > 1:
            audio = audio.view(-1)
        audio = audio.to(self.device)
        self.chunker.to(self.device)

        chunks = self.chunker.add(audio)
        for ch in chunks:
            vad_result = self.vad(ch)
            
            if self.reply_mode == "turn":
                return await self._process_turn_mode(ch, vad_result, t0)
            else:
                return await self._process_stream_mode(ch, vad_result, t0)
        
        return None
    
    async def _process_stream_mode(self, chunk: torch.Tensor, vad_result: Dict[str, bool], t0: float) -> Optional[torch.Tensor]:
        """Original streaming mode - continuous response."""
        if not vad_result['is_voice']:
            return None
            
        with torch.no_grad():
            ids = self.tok.tokenize(chunk.unsqueeze(0).to(self.device))
        self.user_hist.append(ids)
        user_ctx = self._context(self.user_hist, 8)
        
        # generate fewer tokens per step for lower latency
        with torch.no_grad():
            new_ids = self.model.generate_streaming(user_ctx, max_new_tokens=4, temperature=0.95)
            self.ai_hist.append(new_ids)
            out_audio = self.tok.detokenize(new_ids)
        
        self.lat_hist.append((time.time() - t0) * 1000.0)
        return out_audio.squeeze(0)
    
    async def _process_turn_mode(self, chunk: torch.Tensor, vad_result: Dict[str, bool], t0: float) -> Optional[torch.Tensor]:
        """Turn-based mode - wait for complete user turn before responding."""
        
        # If we're currently generating a response, ignore new input
        if self.generating_response:
            return None
            
        # Accumulate user speech during their turn
        if vad_result['is_voice'] and vad_result['speaking']:
            with torch.no_grad():
                ids = self.tok.tokenize(chunk.unsqueeze(0).to(self.device))
            self.turn_buffer.append(ids)
            self.response_generated = False
            
        # User turn ended - generate response
        elif vad_result['turn_ended'] and self.turn_buffer and not self.response_generated:
            print(f"[DEBUG] Turn ended, generating response from {len(self.turn_buffer)} chunks")
            
            self.generating_response = True
            
            # Combine all chunks from this turn
            if self.turn_buffer:
                turn_ids = torch.cat(self.turn_buffer, dim=1)
                self.user_hist.append(turn_ids)
                user_ctx = self._context(self.user_hist, 16)  # Longer context for turn-based
                
                # Generate complete response (more tokens for turn-based)
                with torch.no_grad():
                    new_ids = self.model.generate_streaming(
                        user_ctx, 
                        max_new_tokens=16,  # Longer response for turn-based
                        temperature=0.9
                    )
                    self.ai_hist.append(new_ids)
                    out_audio = self.tok.detokenize(new_ids)
                
                # Clear turn buffer and reset state
                self.turn_buffer.clear()
                self.response_generated = True
                self.generating_response = False
                
                self.lat_hist.append((time.time() - t0) * 1000.0)
                print(f"[DEBUG] Response generated, latency: {(time.time() - t0)*1000:.1f}ms")
                return out_audio.squeeze(0)
            
            self.generating_response = False
        
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
        return {
            "mean": sum(arr)/n, 
            "min": arr[0], 
            "max": arr[-1], 
            "p95": arr[int(0.95*n)-1] if n>1 else arr[0],
            "mode": self.reply_mode,
            "turn_buffer_size": len(self.turn_buffer)
        }

    def reset(self):
        """Reset processor state for new conversation."""
        self.user_hist.clear()
        self.ai_hist.clear()
        self.lat_hist.clear()
        self.turn_buffer.clear()
        self.generating_response = False
        self.response_generated = False
        self.chunker.buf = torch.empty(0, device=self.device)
        self.vad.reset()
        print(f"[INFO] StreamingProcessor reset (mode: {self.reply_mode})")
