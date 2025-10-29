#!/usr/bin/env python3
"""
HiFiGAN-only neural vocoder integration. Fails with explicit error if HiFiGAN is unavailable.
No fallback to Griffin-Lim.
"""
from typing import Optional
import torch
import torch.nn as nn
import sys

try:
    from torchaudio.pipelines import HIFIGAN_VOCODER
except ImportError:
    print("[ERROR] HiFiGAN vocoder not available in torchaudio.pipelines! Please install latest torchaudio for CUDA support.")
    sys.exit(1)

class Vocoder(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = HIFIGAN_VOCODER # throws ImportError if not present
        self.vocoder = bundle.get_model().to(self.device).eval()
        self.sample_rate = 24000

    @torch.no_grad()
    def infer(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.clamp(1e-6, 1.0)
        x = torch.log(x)
        wav = self.vocoder(x.to(self.device)).cpu()
        return wav
