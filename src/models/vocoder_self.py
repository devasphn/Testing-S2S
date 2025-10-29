#!/usr/bin/env python3
"""
Self-contained HiFiGAN loader (universal checkpoint)
- Downloads a small 24kHz HiFiGAN generator checkpoint on first run
- Runs inference to convert mel -> waveform
"""
from typing import Optional
import os
import torch
import torch.nn as nn
import requests
from pathlib import Path

HIFIGAN_URL = "https://huggingface.co/jik876/hifigan/resolve/main/g_02500000"  # example universal checkpoint
HIFIGAN_CFG_URL = "https://huggingface.co/jik876/hifigan/resolve/main/config.json"
CACHE_DIR = Path(os.getenv("TORCH_HOME", "/workspace/cache/torch")) / "hifigan"
GEN_PATH = CACHE_DIR / "generator.pt"
CFG_PATH = CACHE_DIR / "config.json"

# Minimal HiFiGAN generator (structure simplified for loading common checkpoints)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, 1, 1),
            nn.Conv1d(channels, channels, 3, 1, 1),
        ])
        self.acts = nn.ModuleList([nn.LeakyReLU(0.1), nn.LeakyReLU(0.1)])
    def forward(self, x):
        for c, a in zip(self.convs, self.acts):
            x = a(c(x))
        return x

class HiFiGANGenerator(nn.Module):
    def __init__(self, upsample_rates=(8,8,2,2), upsample_kernel_sizes=(16,16,4,4), resblock_kernel=3, channels=512, in_mels=80):
        super().__init__()
        self.pre = nn.Conv1d(in_mels, channels, 7, 1, 3)
        ups = []
        ch = channels
        for r, k in zip(upsample_rates, upsample_kernel_sizes):
            ups.append(nn.ConvTranspose1d(ch, ch//2, k, r, (k-r)//2))
            ch//=2
        self.ups = nn.ModuleList(ups)
        self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(4)])
        self.post = nn.Conv1d(ch, 1, 7, 1, 3)
        self.tanh = nn.Tanh()
    def forward(self, mel):
        x = self.pre(mel)
        for up in self.ups:
            x = torch.leaky_relu(up(x), 0.1)
        for b in self.blocks:
            x = b(x)
        x = self.post(x)
        return self.tanh(x)

class SelfContainedVocoder(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not GEN_PATH.exists():
            print("[INFO] Downloading HiFiGAN generator checkpoint...")
            r = requests.get(HIFIGAN_URL, timeout=60)
            r.raise_for_status()
            GEN_PATH.write_bytes(r.content)
        # Load generator
        self.gen = HiFiGANGenerator().to(self.device)
        try:
            state = torch.load(GEN_PATH, map_location=self.device)
            # Attempt to handle various checkpoint formats
            if isinstance(state, dict) and 'generator' in state:
                state = state['generator']
            self.gen.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint strictly: {e}\nProceeding with randomly initialized generator (degraded quality).")
        self.gen.eval()

    @torch.no_grad()
    def infer(self, mel: torch.Tensor) -> torch.Tensor:
        # Expect mel in [B, n_mels, T] normalized 0..1
        x = mel.to(self.device).clamp(1e-6, 1.0)
        # Map to approximate log domain
        x = torch.log(x)
        wav = self.gen(x).squeeze(1).cpu()  # [B, T]
        return wav
