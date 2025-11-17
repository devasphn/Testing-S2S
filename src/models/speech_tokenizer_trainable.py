#!/usr/bin/env python3
"""
Trainable Speech Tokenizer with Residual Vector Quantization (RVQ)
Based on Encodec/SpeechTokenizer architecture

Supports:
- Multi-scale RVQ for high-quality speech compression
- Checkpoint loading and saving
- Training with reconstruction losses
- Compatible with existing HiFiGAN vocoder
"""
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from .hifigan_public import PublicHiFiGANVocoder


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantization for speech tokens"""
    
    def __init__(self, num_quantizers: int, codebook_size: int, hidden_dim: int):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        
        # Create separate codebooks for each quantizer
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_dim)
            for _ in range(num_quantizers)
        ])
        
        # Initialize codebooks with uniform distribution
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0/codebook_size, 1.0/codebook_size)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, T, D] - Encoded representation
        
        Returns:
            quantized: [B, T, D] - Quantized representation
            indices: [B, num_q, T] - Codebook indices
            commitment_loss: scalar - VQ commitment loss
        """
        B, T, D = z.shape
        
        quantized = torch.zeros_like(z)
        all_indices = []
        residual = z.clone()
        commitment_loss = 0.0
        
        for i, codebook in enumerate(self.codebooks):
            # Flatten for distance computation
            flat_residual = residual.reshape(-1, D)  # [B*T, D]
            
            # Compute distances to all codebook vectors
            distances = torch.cdist(flat_residual, codebook.weight)  # [B*T, codebook_size]
            
            # Find nearest codebook vector
            indices = torch.argmin(distances, dim=1)  # [B*T]
            
            # Quantize
            quantized_layer = codebook(indices).reshape(B, T, D)  # [B, T, D]
            
            # Accumulate quantized representation
            quantized = quantized + quantized_layer
            
            # Straight-through estimator for gradients
            quantized_st = residual + (quantized - residual).detach()
            
            # Commitment loss (encourages encoder to commit to codebook)
            commitment_loss += F.mse_loss(quantized_layer.detach(), residual)
            
            # Update residual for next quantizer
            residual = residual - quantized_layer.detach()
            
            # Store indices
            all_indices.append(indices.reshape(B, T))
        
        # Stack all indices: [B, num_q, T]
        indices_tensor = torch.stack(all_indices, dim=1)
        
        return quantized_st, indices_tensor, commitment_loss / self.num_quantizers


class ConvolutionalEncoder(nn.Module):
    """Multi-scale convolutional encoder for audio"""
    
    def __init__(self, n_mels: int = 80, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        
        channels = [n_mels] + [hidden_dim] * num_layers
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels[i], channels[i+1], kernel_size=7, stride=2 if i > 0 else 1, padding=3),
                nn.GroupNorm(8, channels[i+1]),
                nn.GELU(),
            )
            for i in range(num_layers)
        ])
        
        # Final projection
        self.proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_mels, T] - Mel spectrogram
        
        Returns:
            h: [B, T', D] - Encoded representation (downsampled by 8x)
        """
        for conv in self.convs:
            x = conv(x)
        
        x = self.proj(x)
        x = x.transpose(1, 2)  # [B, T', D]
        return x


class ConvolutionalDecoder(nn.Module):
    """Multi-scale convolutional decoder for audio"""
    
    def __init__(self, hidden_dim: int = 512, n_mels: int = 80, num_layers: int = 4):
        super().__init__()
        
        channels = [hidden_dim] * num_layers + [n_mels]
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(channels[i], channels[i+1], 
                                  kernel_size=8, stride=2 if i < num_layers-1 else 1, 
                                  padding=3, output_padding=1 if i < num_layers-1 else 0),
                nn.GroupNorm(8, channels[i+1]) if i < num_layers-1 else nn.Identity(),
                nn.GELU() if i < num_layers-1 else nn.Identity(),
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - Quantized representation
        
        Returns:
            mel: [B, n_mels, T*8] - Reconstructed mel spectrogram
        """
        x = x.transpose(1, 2)  # [B, D, T]
        
        for conv in self.convs:
            x = conv(x)
        
        return x  # [B, n_mels, T*8]


class TrainableSpeechTokenizer(nn.Module):
    """
    Complete trainable speech tokenizer with:
    - Convolutional encoder/decoder
    - Residual Vector Quantization (RVQ)
    - HiFiGAN vocoder for audio synthesis
    - Checkpoint management
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        sample_rate: int = 24000,
        hop_ms: int = 20,  # 50 Hz frame rate
        codebook_size: int = 1024,
        hidden_dim: int = 512,
        num_quantizers: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.sr = sample_rate
        self.hop = int(sample_rate * hop_ms / 1000)
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        self.num_quantizers = num_quantizers
        
        # Encoder: Mel → Latent
        self.encoder = ConvolutionalEncoder(n_mels, hidden_dim, num_encoder_layers)
        
        # Vector Quantization
        self.quantizer = ResidualVectorQuantizer(num_quantizers, codebook_size, hidden_dim)
        
        # Decoder: Latent → Mel
        self.decoder = ConvolutionalDecoder(hidden_dim, n_mels, num_decoder_layers)
        
        # Vocoder: Mel → Audio (pretrained, frozen)
        print("[INFO] Initializing PublicHiFiGANVocoder...")
        self.vocoder = PublicHiFiGANVocoder()
        self.vocoder.eval()
        for param in self.vocoder.parameters():
            param.requires_grad = False
        print("[INFO] PublicHiFiGANVocoder ready (frozen)")
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert raw audio to mel spectrogram"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        mels = []
        for i in range(audio.size(0)):
            a = audio[i].detach().cpu().numpy()
            
            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=a, sr=self.sr, n_mels=self.n_mels,
                hop_length=self.hop, n_fft=2048,
                fmin=0, fmax=8000
            )
            
            # Convert to log scale
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize to [0, 1]
            mel_norm = (mel_db + 80.0) / 80.0
            mel_norm = np.clip(mel_norm, 0.0, 1.0)
            
            mels.append(torch.tensor(mel_norm, dtype=torch.float32))
        
        return torch.stack(mels).to(audio.device)
    
    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to audio waveform"""
        return self.vocoder.infer(mel)
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Audio → Mel → Latent"""
        mel = self.audio_to_mel(audio)  # [B, n_mels, T]
        h = self.encoder(mel)  # [B, T', D]
        return h
    
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Latent → Mel"""
        mel = self.decoder(quantized)  # [B, n_mels, T]
        return mel
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: Audio → Mel → Encode → Quantize → Decode → Mel → Audio
        
        Returns dict with:
        - mel_original: Original mel spectrogram
        - mel_reconstructed: Reconstructed mel spectrogram
        - audio_reconstructed: Reconstructed audio waveform
        - quantized: Quantized latent representation
        - indices: Codebook indices [B, num_q, T]
        - commitment_loss: VQ commitment loss
        """
        # Encode
        h = self.encode(audio)  # [B, T, D]
        
        # Quantize with RVQ
        quantized, indices, commitment_loss = self.quantizer(h)
        
        # Decode
        mel_recon = self.decode(quantized)  # [B, n_mels, T*8]
        
        # Original mel for comparison
        mel_orig = self.audio_to_mel(audio)
        
        # Synthesize audio
        with torch.no_grad():  # Vocoder is frozen
            audio_recon = self.mel_to_audio(mel_recon)
        
        return {
            "mel_original": mel_orig,
            "mel_reconstructed": mel_recon,
            "audio_reconstructed": audio_recon,
            "quantized": quantized,
            "indices": indices,  # These are your speech tokens!
            "commitment_loss": commitment_loss,
        }
    
    def tokenize(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to discrete tokens (inference only)"""
        with torch.no_grad():
            h = self.encode(audio)
            _, indices, _ = self.quantizer(h)
            return indices  # [B, num_q, T]
    
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete tokens back to audio (inference only)"""
        with torch.no_grad():
            B, num_q, T = indices.shape
            
            # Reconstruct quantized representation from indices
            quantized = torch.zeros(B, T, self.hidden_dim, device=indices.device)
            for i in range(num_q):
                quantized += self.quantizer.codebooks[i](indices[:, i, :])  # [B, T, D]
            
            # Decode to mel
            mel = self.decode(quantized)
            
            # Synthesize audio
            audio = self.mel_to_audio(mel)
            return audio
    
    def save_checkpoint(self, path: str, epoch: int = 0, optimizer_state: Optional[dict] = None, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "config": {
                "n_mels": self.n_mels,
                "sample_rate": self.sr,
                "hop_ms": int(self.hop * 1000 / self.sr),
                "codebook_size": self.codebook_size,
                "hidden_dim": self.hidden_dim,
                "num_quantizers": self.num_quantizers,
            },
            **kwargs,
        }
        
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Saved to {path}")
    
    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint"""
        if not Path(path).exists():
            print(f"[WARNING] Checkpoint not found: {path}")
            return {}
        
        print(f"[CHECKPOINT] Loading from {path}")
        checkpoint = torch.load(path, map_location="cpu")
        
        # Load model weights
        if "model_state_dict" in checkpoint:
            # Don't load vocoder weights (it's pretrained and frozen)
            state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() 
                         if not k.startswith("vocoder.")}
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(checkpoint, strict=strict)
        
        print(f"[CHECKPOINT] Loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint
    
    def to(self, device):
        """Move model to device including vocoder"""
        super().to(device)
        if hasattr(self.vocoder, 'to'):
            self.vocoder.to(device)
        return self
