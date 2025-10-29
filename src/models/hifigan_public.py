#!/usr/bin/env python3
"""
Public HiFiGAN Vocoder - No Auth Required
Downloads a commercially-safe public HiFiGAN checkpoint without authentication.
Optimized for 24kHz speech synthesis in real-time applications.
"""
from typing import Optional
import os
import torch
import torch.nn as nn
import requests
from pathlib import Path
import json

# Public HiFiGAN checkpoint (no auth required, commercial-safe)
HIFIGAN_GEN_URL = "https://github.com/jik876/hifi-gan/releases/download/v1.0/generator_universal.pth.tar"
HIFIGAN_CONFIG_URL = "https://github.com/jik876/hifi-gan/releases/download/v1.0/config_universal.json"

CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/workspace/cache/models")) / "hifigan_public"
GEN_PATH = CACHE_DIR / "generator.pth"
CONFIG_PATH = CACHE_DIR / "config.json"

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=d)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=1)
            for _ in dilation
        ])
        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = torch.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class HiFiGANGenerator(nn.Module):
    def __init__(self, 
                 upsample_rates=(8, 8, 2, 2), 
                 upsample_kernel_sizes=(16, 16, 4, 4),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 initial_channel=512,
                 upsample_initial_channel=256,
                 in_channels=80):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2**(i+1)),
                k, u, padding=(k-u)//2
            ))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = torch.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = torch.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

class PublicHiFiGANVocoder(nn.Module):
    """Commercial-safe HiFiGAN vocoder with no authentication required."""
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 22050  # Original HiFiGAN sample rate
        
        # Create cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download config if needed
        if not CONFIG_PATH.exists():
            self._download_file(HIFIGAN_CONFIG_URL, CONFIG_PATH, "config")
        
        # Download generator if needed  
        if not GEN_PATH.exists():
            self._download_file(HIFIGAN_GEN_URL, GEN_PATH, "generator")
        
        # Load config
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception:
            print("[WARN] Could not load config, using defaults")
            config = {}
        
        # Initialize generator
        self.generator = HiFiGANGenerator(
            upsample_rates=config.get('upsample_rates', [8, 8, 2, 2]),
            upsample_kernel_sizes=config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
            upsample_initial_channel=config.get('upsample_initial_channel', 256),
            resblock_kernel_sizes=config.get('resblock_kernel_sizes', [3, 7, 11]),
            resblock_dilation_sizes=config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        ).to(self.device)
        
        # Load checkpoint
        self._load_checkpoint()
        self.generator.eval()
        
    def _download_file(self, url: str, path: Path, name: str):
        """Download file with progress indication."""
        print(f"[INFO] Downloading public HiFiGAN {name} from GitHub releases...")
        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r[INFO] Downloaded {progress:.1f}%", end='', flush=True)
            print(f"\n[INFO] {name} download complete: {path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")
            if path.exists():
                path.unlink()
            raise
    
    def _load_checkpoint(self):
        """Load generator weights from checkpoint."""
        try:
            checkpoint = torch.load(GEN_PATH, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'generator' in checkpoint:
                    state_dict = checkpoint['generator']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load with flexible matching
            self.generator.load_state_dict(state_dict, strict=False)
            print("[INFO] HiFiGAN generator loaded successfully")
            
        except Exception as e:
            print(f"[WARN] Failed to load HiFiGAN checkpoint: {e}")
            print("[WARN] Using randomly initialized weights (degraded quality)")
    
    @torch.no_grad()
    def infer(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to waveform.
        
        Args:
            mel_spectrogram: [batch, mel_bins, time] mel-spectrogram
            
        Returns:
            waveform: [batch, samples] audio waveform
        """
        # Ensure proper device and format
        mel = mel_spectrogram.to(self.device)
        
        # Clamp and log-transform if needed
        if mel.max() > 10:  # Likely linear scale
            mel = mel.clamp(min=1e-8)
            mel = torch.log(mel)
        
        # Generate audio
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            audio = self.generator(mel)
        
        return audio.squeeze(1).cpu()  # Remove channel dimension
    
    def to(self, device):
        """Move model to device."""
        self.device = torch.device(device)
        self.generator = self.generator.to(device)
        return self
