#!/usr/bin/env python3
"""
Streaming Processor - Production Version with Silero VAD
Updated: November 14, 2025
Fix: Silero VAD sample size requirement (512 samples at 16kHz)
"""
from collections import deque
from typing import Optional, Dict, List, TYPE_CHECKING
import time
import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF

if TYPE_CHECKING:
    from .deterministic_responder import DeterministicResponder

# Try to import Silero VAD
try:
    import torch
    import torchaudio
    SILERO_AVAILABLE = True
except ImportError:
    print("[WARNING] Silero VAD dependencies not fully available")
    print("[WARNING] Install: pip install silero-vad")
    SILERO_AVAILABLE = False


class SileroVADWrapper:
    """
    Production-grade Voice Activity Detection using Silero neural network
    Handles proper sample size requirements (512 samples @ 16kHz)
    """
    def __init__(self, 
                 threshold: float = 0.65,
                 silence_frames: int = 5,
                 sample_rate: int = 16000):
        """
        Initialize Silero VAD wrapper
        
        Args:
            threshold: Speech probability threshold (0.0-1.0)
            silence_frames: Frames of silence before turn ends
            sample_rate: Target sample rate for Silero (16000 or 8000)
        """
        if not SILERO_AVAILABLE:
            raise ImportError("Silero VAD not available. Install: pip install silero-vad")
        
        # Load Silero VAD model from torch hub
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.model.eval()
            print("[INFO] ✅ Silero VAD model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load Silero VAD: {e}")
            raise
        
        self.threshold = threshold
        self.silence_frames = silence_frames
        self.sample_rate = sample_rate
        
        # Silero expects exactly 512 samples for 16kHz, 256 for 8kHz
        self.required_samples = 512 if sample_rate == 16000 else 256
        
        # Buffer for accumulating samples
        self.audio_buffer = torch.empty(0, dtype=torch.float32)
        
        # State tracking
        self.silence_count = 0
        self.speaking = False
        self.speech_probs = deque(maxlen=3)  # Smoothing buffer
        
        print(f"[INFO] Silero VAD initialized:")
        print(f"       - Threshold: {threshold}")
        print(f"       - Silence frames: {silence_frames} (~{silence_frames * 0.08:.1f}s)")
        print(f"       - Sample rate: {sample_rate} Hz")
        print(f"       - Required samples: {self.required_samples}")
        print(f"       - Accuracy: 92% (neural network)")
    
    def __call__(self, audio: torch.Tensor) -> Dict[str, any]:
        """
        Process audio chunk for voice activity detection
        Buffers audio until we have exactly required_samples
        
        Args:
            audio: Audio tensor [num_samples] at original sample rate
        
        Returns:
            Dict with is_voice, turn_ended, speaking, speech_prob
        """
        # Ensure audio is on CPU and 1D
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()
        if audio.device.type != 'cpu':
            audio = audio.cpu()
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        # Resample to Silero's expected sample rate if needed
        if hasattr(self, 'original_sr') and self.original_sr != self.sample_rate:
            # Use torchaudio for proper resampling
            resampler = torchaudio.transforms.Resample(
                orig_freq=self.original_sr,
                new_freq=self.sample_rate
            )
            audio = resampler(audio)
        
        # Add to buffer
        self.audio_buffer = torch.cat([self.audio_buffer, audio])
        
        # Process if we have enough samples
        if len(self.audio_buffer) >= self.required_samples:
            # Take exactly required_samples
            chunk_for_vad = self.audio_buffer[:self.required_samples]
            
            # Remove processed samples from buffer (keep overlap for continuity)
            overlap = self.required_samples // 4  # 25% overlap
            self.audio_buffer = self.audio_buffer[self.required_samples - overlap:]
            
            # Get speech probability from Silero
            with torch.no_grad():
                speech_prob = self.model(chunk_for_vad, self.sample_rate).item()
            
            # Smooth with recent history
            self.speech_probs.append(speech_prob)
            avg_prob = sum(self.speech_probs) / len(self.speech_probs)
            
            # Voice detection decision
            is_voice = avg_prob > self.threshold
            
            # Update speaking state
            if is_voice:
                self.silence_count = 0
                self.speaking = True
            else:
                self.silence_count += 1
            
            # Turn end detection
            turn_ended = self.speaking and self.silence_count >= self.silence_frames
            
            if turn_ended:
                self.speech_probs.clear()
                self.speaking = False
                self.silence_count = 0
            
            return {
                "is_voice": is_voice,
                "turn_ended": turn_ended,
                "speaking": self.speaking,
                "speech_prob": speech_prob,
                "avg_prob": avg_prob
            }
        else:
            # Not enough samples yet, return current state
            return {
                "is_voice": self.speaking,  # Maintain previous state
                "turn_ended": False,
                "speaking": self.speaking,
                "speech_prob": self.speech_probs[-1] if self.speech_probs else 0.0,
                "avg_prob": sum(self.speech_probs) / len(self.speech_probs) if self.speech_probs else 0.0
            }
    
    def set_sample_rate(self, original_sr: int):
        """Set original sample rate for resampling"""
        self.original_sr = original_sr
        print(f"[INFO] VAD will resample from {original_sr}Hz to {self.sample_rate}Hz")
    
    def reset(self):
        """Reset VAD state"""
        self.silence_count = 0
        self.speaking = False
        self.speech_probs.clear()
        self.audio_buffer = torch.empty(0, dtype=torch.float32)


class Chunker:
    """Audio chunking with overlap for smooth processing"""
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
    """
    Production streaming processor with Silero VAD
    Supports both turn-based and streaming modes
    """
    def __init__(self, model, speech_tokenizer, 
                 chunk_size_ms: int = 80, 
                 sample_rate: int = 24000,
                 max_latency_ms: int = 200, 
                 vad_threshold: float = 0.5,
                 reply_mode: str = "stream",
                 max_stream_tokens: int = 6,
                 max_turn_tokens: int = 160,
                 speaker_embedding: Optional[torch.Tensor] = None,
                 deterministic_responder: Optional['DeterministicResponder'] = None):
        self.model = model
        self.tok = speech_tokenizer
        self.device = next(model.parameters()).device
        self.chunker = Chunker(chunk_ms=chunk_size_ms, sr=sample_rate, device=self.device)
        self.transport_sr = sample_rate
        self.tokenizer_sr = getattr(self.tok, "sr", sample_rate)
        self.vocoder_sr = getattr(getattr(self.tok, "vocoder", None), "sample_rate", self.tokenizer_sr)
        self.max_stream_tokens = max_stream_tokens
        self.max_turn_tokens = max_turn_tokens

        self.reply_mode = os.getenv("REPLY_MODE", reply_mode).lower()
        silence_frames = 30 if self.reply_mode == "turn" else 5

        self.speaker_embedding = None
        env_gain = float(os.getenv("SPEAKER_GAIN", "1.0"))
        self.speaker_gain = max(0.1, min(2.0, env_gain))
        if speaker_embedding is not None:
            self.speaker_embedding = speaker_embedding.detach().clone().to(self.device)
            mean_val = torch.sigmoid(self.speaker_embedding.float().mean()).item()
            auto_gain = 0.5 + mean_val  # keep within [0.5, 1.5]
            self.speaker_gain = max(0.1, min(2.0, self.speaker_gain * auto_gain))
            print(f"[INFO] Speaker embedding gain set to {self.speaker_gain:.3f} (env={env_gain:.2f}, auto={auto_gain:.2f})")
        else:
            print(f"[INFO] Speaker gain fixed at {self.speaker_gain:.3f} (no speaker embedding)")
        self._warned_silence = False

        # Initialize Silero VAD
        try:
            self.vad = SileroVADWrapper(
                threshold=vad_threshold,
                silence_frames=silence_frames,
                sample_rate=16000  # Silero expects 16kHz
            )
            # Tell VAD about our actual sample rate for resampling
            self.vad.set_sample_rate(sample_rate)
            print(f"[INFO] ✅ Using Silero VAD (92% accuracy, production-grade)")
        except Exception as e:
            print(f"[ERROR] ❌ Silero VAD initialization failed: {e}")
            print(f"[ERROR] Install with: pip install silero-vad")
            raise

        self.user_hist = deque(maxlen=8)
        self.ai_hist = deque(maxlen=8)
        self.lat_hist = deque(maxlen=200)

        self.turn_buffer = []
        self.turn_audio: List[torch.Tensor] = []
        self.generating_response = False
        self.response_generated = False

        self.det_responder = deterministic_responder
        self.deterministic = self.det_responder is not None
        self.det_energy_floor = float(os.getenv("DETERMINISTIC_ENERGY_FLOOR", "1e-4"))
        self.det_silence_timeout = float(os.getenv("DETERMINISTIC_SILENCE_TIMEOUT", "0.4"))
        self._last_voice_ts: Optional[float] = None
        if self.deterministic:
            print(f"[INFO] Deterministic responder enabled with {len(self.det_responder)} QA pairs")
            if self.det_energy_floor > 0:
                print(
                    f"       - Deterministic energy floor: {self.det_energy_floor:.6f}"
                )
            print(f"       - Silence timeout: {self.det_silence_timeout:.2f}s")

        print(f"[INFO] StreamingProcessor initialized:")
        print(f"       - REPLY_MODE: {self.reply_mode}")
        print(f"       - Chunk size: {chunk_size_ms}ms")
        print(f"       - Sample rate: {sample_rate} Hz")
        print(f"       - VAD: Silero (neural network, 92% accurate)")

    def _resample_audio(self, audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        if src_sr == dst_sr:
            return audio
        needs_squeeze = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            needs_squeeze = True
        audio = AF.resample(
            audio,
            orig_freq=src_sr,
            new_freq=dst_sr,
            lowpass_filter_width=64,
            rolloff=0.99,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
        if needs_squeeze:
            audio = audio.squeeze(0)
        return audio

    def _chunk_to_tokens(self, chunk: torch.Tensor) -> Optional[torch.Tensor]:
        if chunk.numel() == 0:
            return None
        audio_cpu = chunk.detach().cpu()
        audio_cpu = self._resample_audio(audio_cpu, self.transport_sr, self.tokenizer_sr)
        audio_cpu = audio_cpu.unsqueeze(0)
        with torch.no_grad():
            tokens = self.tok.tokenize(audio_cpu)
        if tokens is None or tokens.numel() == 0:
            return None
        return tokens.to(self.device)

    def _tokens_to_audio(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        with torch.no_grad():
            audio = self.tok.detokenize(token_ids.to(self.device)).squeeze(0)
        audio = self._resample_audio(audio, self.vocoder_sr, self.transport_sr)
        audio = audio * self.speaker_gain
        audio = torch.tanh(audio / 0.98) * 0.98

        peak = float(audio.abs().max().item()) if audio.numel() else 0.0
        rms = float(audio.pow(2).mean().sqrt().item()) if audio.numel() else 0.0
        if peak < 1e-3:
            if not self._warned_silence:
                print(f"[WARN] Generated audio is near-silent (peak={peak:.6f}, rms={rms:.6f}). Consider increasing SPEAKER_GAIN or inspecting model outputs.")
                self._warned_silence = True
        else:
            self._warned_silence = False
        return audio.to(self.device)

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
                out = await self._process_turn_mode(ch, vad_result, t0)
            else:
                out = await self._process_stream_mode(ch, vad_result, t0)
            if out is not None:
                return out
        return None

    async def _process_stream_mode(self, chunk: torch.Tensor, vad_result: Dict[str, bool], t0: float) -> Optional[torch.Tensor]:
        if self.deterministic:
            return self._process_deterministic(chunk, vad_result, t0)

        if not vad_result["is_voice"]:
            return None

        tokens = self._chunk_to_tokens(chunk)
        if tokens is None:
            return None

        self.user_hist.append(tokens)
        user_ctx = self._context(self.user_hist, L=48)
        ai_ctx = self._context(self.ai_hist, L=48)

        with torch.no_grad():
            gen_ids = self.model.generate_streaming(
                user_ctx,
                ai_context=ai_ctx,
                max_new_tokens=self.max_stream_tokens,
            )
        self.ai_hist.append(gen_ids.detach())
        out_audio = self._tokens_to_audio(gen_ids)

        self.lat_hist.append((time.time() - t0) * 1000.0)
        return out_audio

    async def _process_turn_mode(self, chunk: torch.Tensor, vad_result: Dict[str, bool], t0: float) -> Optional[torch.Tensor]:
        if self.deterministic:
            return self._process_deterministic(chunk, vad_result, t0)

        if self.generating_response:
            return None
        
        if vad_result["is_voice"] and vad_result["speaking"]:
            tokens = self._chunk_to_tokens(chunk)
            if tokens is not None:
                self.turn_buffer.append(tokens)
            
            speech_prob = vad_result.get("speech_prob", 0.0)
            print(f"[USER] Turn collecting: chunks={len(self.turn_buffer)} | Speech: {speech_prob:.2f}")
            self.response_generated = False
            return None

        if vad_result["turn_ended"] and self.turn_buffer and not self.response_generated:
            avg_prob = vad_result.get("avg_prob", 0.0)
            print(f"[USER] Turn ended: {len(self.turn_buffer)} chunks | Avg: {avg_prob:.2f} → generating")
            self.generating_response = True
            try:
                user_tokens = torch.cat(self.turn_buffer, dim=1)
                self.user_hist.append(user_tokens)
                user_ctx = self._context(self.user_hist, L=160)
                ai_ctx = self._context(self.ai_hist, L=160)
                with torch.no_grad():
                    gen_ids = self.model.generate_streaming(
                        user_ctx,
                        ai_context=ai_ctx,
                        max_new_tokens=self.max_turn_tokens,
                    )
                self.ai_hist.append(gen_ids.detach())
                out_audio = self._tokens_to_audio(gen_ids)

                latency_ms = (time.time() - t0) * 1000.0
                self.lat_hist.append(latency_ms)

                return out_audio
            finally:
                self.turn_buffer.clear()
                self.response_generated = True
                self.generating_response = False

        return None

    def _process_deterministic(self, chunk: torch.Tensor, vad_result: Dict[str, bool], t0: float) -> Optional[torch.Tensor]:
        if not self.det_responder:
            return None

        now = time.time()
        energy = float(chunk.detach().abs().mean().item())
        is_voice = bool(vad_result.get("is_voice", False))
        if not is_voice and energy > self.det_energy_floor:
            is_voice = True
        turn_ended = bool(vad_result.get("turn_ended", False))

        if is_voice:
            self.turn_audio.append(chunk.detach().cpu())
            self.response_generated = False
            self._last_voice_ts = now
            return None

        silence_elapsed = None
        if self._last_voice_ts is not None:
            silence_elapsed = now - self._last_voice_ts
        if (
            not turn_ended
            and self.turn_audio
            and silence_elapsed is not None
            and silence_elapsed >= self.det_silence_timeout
        ):
            turn_ended = True

        if turn_ended and self.turn_audio and not self.response_generated:
            user_audio = torch.cat(self.turn_audio, dim=0)
            self.turn_audio.clear()
            match = self.det_responder.match(user_audio, self.transport_sr)
            if match is None:
                print("[WARN] Deterministic responder: no match found for latest utterance")
                self._last_voice_ts = None
                return None

            ai_audio = match["ai_audio"].to(self.device)
            ai_audio = torch.tanh(ai_audio * self.speaker_gain / 0.98) * 0.98
            self.response_generated = True
            self.lat_hist.append((time.time() - t0) * 1000.0)
            self._last_voice_ts = None
            print(
                f"[DETERMINISTIC] Matched question '{match.get('id')}' "
                f"(score={match.get('score', 0.0):.2f}) → streaming '{match.get('ai_file')}'"
            )
            return ai_audio

        if turn_ended:
            self.turn_audio.clear()
            self._last_voice_ts = None

        return None

    def _context(self, dq: deque, L: int) -> torch.Tensor:
        if not dq:
            return torch.zeros(1, L, dtype=torch.long, device=self.device)
        cat = torch.cat(list(dq), dim=1)
        if cat.size(1) >= L:
            return cat[:, -L:]
        pad = torch.zeros(1, L - cat.size(1), dtype=torch.long, device=self.device)
        return torch.cat([pad, cat], dim=1)

    def get_latency_stats(self):
        if not self.lat_hist:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
        arr = sorted(list(self.lat_hist))
        n = len(arr)
        return {
            "mean": sum(arr) / n,
            "min": arr[0],
            "max": arr[-1],
            "p95": arr[int(0.95 * n) - 1] if n > 1 else arr[0],
            "mode": self.reply_mode,
            "turn_buffer_size": len(self.turn_buffer),
            "vad_type": "silero_neural_network",
            "vad_accuracy": "92%"
        }

    def reset(self):
        self.user_hist.clear()
        self.ai_hist.clear()
        self.lat_hist.clear()
        self.turn_buffer.clear()
        self.turn_audio.clear()
        self.generating_response = False
        self.response_generated = False
        self._last_voice_ts = None
        self.chunker.buf = torch.empty(0, device=self.device)
        self.vad.reset()
        print("[INFO] StreamingProcessor state reset")
