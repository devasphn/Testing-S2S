#!/usr/bin/env python3
"""
GLM-4-Voice style ultra-low bitrate speech tokenizer (HiFiGAN enforced)
- Fails with error if HiFiGAN is not available
- Uses only neural vocoder for mel_to_audio
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from .vocoder import Vocoder

class SpeechTokenizer(nn.Module):
    def __init__(self, n_mels: int = 80, sample_rate: int = 24000, hop_ms: int = 80, codebook_size: int = 1024, hidden: int = 256):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sample_rate
        self.hop = int(sample_rate * hop_ms / 1000)
        self.codebook_size = codebook_size
        self.hidden = hidden

        self.enc = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.enc_tf = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden*2, batch_first=True)
        self.dec_tf = nn.TransformerDecoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden*2, batch_first=True)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden, n_mels, 3, padding=1),
        )
        self.codebook = nn.Parameter(torch.randn(codebook_size, hidden) * 0.02)
        self.vocoder = Vocoder()

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mels = []
        for i in range(audio.size(0)):
            a = audio[i].detach().cpu().numpy()
            mel = librosa.feature.melspectrogram(y=a, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop, n_fft=1024)
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = (mel + 80.0) / 80.0
            mels.append(torch.tensor(mel, dtype=torch.float32))
        mel = torch.stack(mels).to(audio.device)  # [B, n_mels, T]
        return mel

    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        wav = self.vocoder.infer(mel)
        if wav.numel() == 0:
            raise RuntimeError("[ERROR] HiFiGAN vocoder is required and was not found. Please install torchaudio >=2.2.0 with CUDA support.")
        return wav

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.audio_to_mel(audio)
        h = self.enc(mel)
        h = h.transpose(1, 2)
        h = self.enc_tf(h)
        return h

    def quantize(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = h.shape
        flat = h.reshape(-1, d)
        dists = (flat.unsqueeze(1) - self.codebook.unsqueeze(0)).pow(2).sum(-1)
        idx = torch.argmin(dists, dim=1)
        q = self.codebook[idx].reshape(b, t, d)
        q = h + (q - h).detach()
        return q, idx.reshape(b, t)

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        x = self.dec_tf(q, q)
        x = x.transpose(1, 2)
        mel = self.dec(x)
        return mel

    def tokenize(self, audio: torch.Tensor) -> torch.Tensor:
        h = self.encode(audio)
        _, idx = self.quantize(h)
        return idx

    def detokenize(self, idx: torch.Tensor) -> torch.Tensor:
        q = self.codebook[idx]
        mel = self.decode(q)
        audio = self.mel_to_audio(mel)
        return audio
