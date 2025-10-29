#!/usr/bin/env python3
"""
Public HiFiGAN Vocoder - PyTorch Hub Implementation
Uses NVIDIA's pretrained HiFi-GAN model from PyTorch Hub.
No authentication required, commercial-safe, automatically cached.
"""
from typing import Optional
import torch
import torch.nn as nn
import warnings

class PublicHiFiGANVocoder(nn.Module):
    """Commercial-safe HiFiGAN vocoder using PyTorch Hub NVIDIA model."""
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 22050  # NVIDIA HiFi-GAN sample rate
        
        print("[INFO] Loading NVIDIA HiFi-GAN from PyTorch Hub...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.hifigan, self.vocoder_config, self.denoiser = torch.hub.load(
                    'NVIDIA/DeepLearningExamples:torchhub',
                    'nvidia_hifigan',
                    force_reload=False,
                    verbose=False
                )
                self.hifigan = self.hifigan.to(self.device).eval()
                if self.denoiser is not None:
                    self.denoiser = self.denoiser.to(self.device).eval()
                if self.vocoder_config and 'sampling_rate' in self.vocoder_config:
                    self.sample_rate = self.vocoder_config['sampling_rate']
                print(f"[INFO] NVIDIA HiFi-GAN loaded successfully ({self.sample_rate}Hz)")
            except Exception as e:
                print(f"[ERROR] Failed to load NVIDIA HiFi-GAN: {e}")
                print("[INFO] Falling back to basic HiFi-GAN implementation...")
                self.hifigan = self._create_fallback_generator()
                self.denoiser = None
                self.vocoder_config = {'sampling_rate': 22050, 'max_wav_value': 32768.0}
    
    def _create_fallback_generator(self) -> nn.Module:
        class BasicHiFiGANGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_pre = nn.Conv1d(80, 256, 7, 1, padding=3)
                self.ups = nn.ModuleList([
                    nn.ConvTranspose1d(256, 128, 16, 8, padding=4),
                    nn.ConvTranspose1d(128, 64, 16, 8, padding=4),
                    nn.ConvTranspose1d(64, 32, 4, 2, padding=1),
                    nn.ConvTranspose1d(32, 16, 4, 2, padding=1),
                ])
                self.conv_post = nn.Conv1d(16, 1, 7, 1, padding=3)
            def forward(self, x):
                x = self.conv_pre(x)
                for up in self.ups:
                    x = torch.leaky_relu(up(x), 0.1)
                x = torch.tanh(self.conv_post(x))
                return x
        return BasicHiFiGANGenerator().to(self.device)
    
    @torch.no_grad()
    def infer(self, mel_spectrogram: torch.Tensor, denoise: bool = False, denoising_strength: float = 0.005) -> torch.Tensor:
        """Convert mel-spectrogram to waveform.
        Returns mono [batch, samples] float32 on CPU.
        Prevent cuDNN use for 1D convs to avoid plan warnings.
        """
        mel = mel_spectrogram.to(self.device)
        if mel.max() > 10:
            mel = mel.clamp(min=1e-8)
            mel = torch.log(mel)
        elif mel.min() >= 0 and mel.max() <= 1:
            mel = mel * 10 - 5
        # Temporarily disable cuDNN to avoid conv1d/transpose1d plan warnings
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            audio = self.hifigan(mel)
            if audio.dim() == 3:
                audio = audio.squeeze(1)
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if denoise and self.denoiser is not None and denoising_strength > 0:
                try:
                    audio = self.denoiser(audio, denoising_strength)
                except Exception:
                    pass
            # Scale safely
            max_wav_value = self.vocoder_config.get('max_wav_value', 1.0) if isinstance(self.vocoder_config, dict) else 1.0
            if max_wav_value != 1.0:
                audio = audio / max_wav_value  # normalize to [-1,1] domain expected by server limiter
            return audio.detach().cpu()
        finally:
            torch.backends.cudnn.enabled = prev
    
    def to(self, device):
        self.device = torch.device(device)
        super().to(device)
        if hasattr(self, 'hifigan'):
            self.hifigan = self.hifigan.to(device)
        if hasattr(self, 'denoiser') and self.denoiser is not None:
            self.denoiser = self.denoiser.to(device)
        return self
    
    def eval(self):
        super().eval()
        if hasattr(self, 'hifigan'):
            self.hifigan.eval()
        if hasattr(self, 'denoiser') and self.denoiser is not None:
            self.denoiser.eval()
        return self
