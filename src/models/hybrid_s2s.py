#!/usr/bin/env python3
"""
Hybrid Speech-to-Speech Model (MVP) - Performance tweaks
- Smaller default hidden size / layers for faster inference
- Fewer attention heads
- Shorter max_seq_len
- Fewer generated tokens per step (handled in processor)
- torch.set_grad_enabled(False) and autocast in inference paths
"""
from typing import Optional, Dict
import torch
import torch.nn as nn
from contextlib import contextmanager


@contextmanager
def inference_mode(device: torch.device):
    torch.set_grad_enabled(False)
    try:
        # prefer fp16 autocast on cuda
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                yield
        else:
            yield
    finally:
        torch.set_grad_enabled(True)


class HybridS2SConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        speech_vocab_size: int = 1024,
        hidden_size: int = 512,            # was 768
        num_hidden_layers: int = 8,        # was 12
        num_attention_heads: int = 8,      # was 12
        intermediate_size: int = 2048,     # was 3072
        max_seq_len: int = 1024,           # was 2048
        chunk_size_ms: int = 80,
        use_dual_stream: bool = True,
    ):
        self.vocab_size = vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.chunk_size_ms = chunk_size_ms
        self.use_dual_stream = use_dual_stream


class MoshiDepthTransformer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,                    # reduce heads
                dim_feedforward=hidden_size * 2,
                dropout=0.05,               # lower dropout for speed
                batch_first=True,
            )
            for _ in range(1)               # fewer layers
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, user_stream: torch.Tensor, ai_stream: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_stream, ai_stream], dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class UnifiedTransformer(nn.Module):
    def __init__(self, cfg: HybridS2SConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size + cfg.speech_vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.mod_embed = nn.Embedding(3, cfg.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.num_attention_heads,
                dim_feedforward=cfg.intermediate_size,
                dropout=0.05,
                batch_first=True,
            )
            for _ in range(cfg.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.text_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.speech_head = nn.Linear(cfg.hidden_size, cfg.speech_vocab_size)

    def forward(self, input_ids: torch.Tensor, modality_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        h = self.token_embed(input_ids) + self.pos_embed(pos) + self.mod_embed(modality_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return {
            "text_logits": self.text_head(h),
            "speech_logits": self.speech_head(h),
            "hidden_states": h,
        }


class HybridS2SModel(nn.Module):
    def __init__(self, cfg: Optional[HybridS2SConfig] = None):
        super().__init__()
        self.cfg = cfg or HybridS2SConfig()
        self.depth = MoshiDepthTransformer(self.cfg.hidden_size)
        self.unified = UnifiedTransformer(self.cfg)
        self.speech_embed = nn.Embedding(self.cfg.speech_vocab_size, self.cfg.hidden_size)

    @torch.no_grad()
    def _ids_to_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        return self.speech_embed(ids)

    def forward(self, user_audio_tokens: torch.Tensor, ai_audio_tokens: torch.Tensor, text_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        device = user_audio_tokens.device
        with inference_mode(device):
            user_h = self._ids_to_hidden(user_audio_tokens)
            ai_h = self._ids_to_hidden(ai_audio_tokens)
            depth_out = self.depth(user_h, ai_h)
            b = user_audio_tokens.size(0)
            depth_len = depth_out.size(1)
            speech_pad_ids = torch.full((b, depth_len), self.cfg.vocab_size, device=device, dtype=torch.long)
            mod_speech = torch.ones_like(speech_pad_ids)
            if text_tokens is not None:
                input_ids = torch.cat([speech_pad_ids, text_tokens], dim=1)
                mod_text = torch.zeros_like(text_tokens)
                modality = torch.cat([mod_speech, mod_text], dim=1)
            else:
                input_ids = speech_pad_ids
                modality = mod_speech
            return self.unified(input_ids=input_ids, modality_ids=modality)

    @torch.no_grad()
    def generate_streaming(self, user_audio_chunk_ids: torch.Tensor, max_new_tokens: int = 6, temperature: float = 0.9) -> torch.Tensor:
        device = user_audio_chunk_ids.device
        with inference_mode(device):
            b = user_audio_chunk_ids.size(0)
            ai_ids = torch.zeros(b, 3, device=device, dtype=torch.long)  # smaller context
            gen = []
            for _ in range(max_new_tokens):
                out = self.forward(user_audio_chunk_ids, ai_ids)
                logits = out["speech_logits"][:, -1, :] / max(temperature, 1e-5)
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                gen.append(next_id)
                ai_ids = torch.cat([ai_ids, next_id], dim=1)[:, -3:]
            return torch.cat(gen, dim=1)
