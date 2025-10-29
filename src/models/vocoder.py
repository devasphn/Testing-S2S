#!/usr/bin/env python3
"""
Simple HiFiGAN wrapper for better audio decoding.
Uses pretrained universal vocoder weights if available; otherwise falls back to Griffin-Lim.
"""
from typing import Optional
import torch
import torch.nn as nn
import os

try:
    from torchaudio.pipelines import HIFIGAN_VOCODER
    HAVE_TA = True
except Exception:
    HAVE_TA = False

class Vocoder(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ok = False
        if HAVE_TA:
            try:
                bundle = HIFIGAN_VOCODER # universal vocoder in torchaudio (if present)
                self.vocoder = bundle.get_model().to(self.device).eval()
                self.sample_rate = 24000
                self.ok = True
            except Exception:
                self.ok = False
        if not self.ok:
            self.vocoder = None
            self.sample_rate = 24000

    @torch.no_grad()
    def infer(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mels, T] in linear scale 0..1 (we'll map to expected range)
        returns audio [B, samples]
        """
        if self.ok and self.vocoder is not None:
            # torchaudio HiFiGAN expects log-mel in specific normalization; we approximate mapping
            x = mel.clamp(1e-6, 1.0)
            x = torch.log(x)
            wav = self.vocoder(x.to(self.device)).cpu()
            return wav
        # Fallback: return empty to signal caller to use Griffin-Lim
        return torch.empty(0)
