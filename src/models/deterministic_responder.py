#!/usr/bin/env python3
"""Deterministic Telugu question→answer matcher for the POC."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T


class DeterministicResponder:
    """Matches an incoming utterance to one of the prerecorded user questions.

    Once a match is found, the paired AI response audio (pre-recorded WAV) is
    returned so the server can stream it directly to the client. This bypasses
    the generative HybridS2S model for an immediate, high-fidelity POC.
    """

    def __init__(
        self,
        metadata_path: Path,
        wav_root: Path,
        transport_sr: int = 24_000,
        feature_sr: int = 16_000,
        min_score: float = 0.75,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.wav_root = Path(wav_root)
        self.transport_sr = transport_sr
        self.feature_sr = feature_sr
        self.min_score = min_score

        self._mfcc = T.MFCC(
            sample_rate=self.feature_sr,
            n_mfcc=40,
            melkwargs={"n_mels": 40, "n_fft": 1024, "hop_length": 256},
        )
        self.entries: List[Dict] = []
        self._bank: Optional[torch.Tensor] = None

        self._load_pairs()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_pairs(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        pairs = metadata.get("pairs", [])
        if not pairs:
            raise RuntimeError("Metadata contains no 'pairs' entries for deterministic POC")

        bank_vectors: List[torch.Tensor] = []

        for pair in pairs:
            user_path = self.wav_root / pair["user_file"]
            ai_path = self.wav_root / pair["ai_file"]
            if not user_path.exists() or not ai_path.exists():
                print(f"[WARN] Skipping pair {pair.get('id')} - missing WAVs")
                continue

            user_audio, user_sr = torchaudio.load(str(user_path))
            user_audio = AF.resample(user_audio, user_sr, self.feature_sr)
            user_vec = self._embed(user_audio)
            if user_vec is None:
                print(f"[WARN] Skipping pair {pair.get('id')} - empty user embedding")
                continue

            ai_audio, ai_sr = torchaudio.load(str(ai_path))
            ai_audio = AF.resample(ai_audio, ai_sr, self.transport_sr).squeeze(0)
            ai_audio = torch.clamp(ai_audio, -1.0, 1.0)

            entry = {
                "id": pair.get("id"),
                "user_file": pair.get("user_file"),
                "ai_file": pair.get("ai_file"),
                "user_text": pair.get("user_text"),
                "ai_text": pair.get("ai_text"),
                "user_vec": user_vec,
                "ai_audio": ai_audio,
            }
            self.entries.append(entry)
            bank_vectors.append(user_vec)

        if not self.entries:
            raise RuntimeError("No valid metadata pairs were loaded for deterministic POC")

        self._bank = torch.stack(bank_vectors, dim=0)
        print(f"[INFO] DeterministicResponder loaded {len(self.entries)} QA pairs")

    def _embed(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        with torch.no_grad():
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if audio.numel() == 0:
                return None
            mfcc = self._mfcc(audio)
            # Average over time dimension -> [n_mfcc]
            vec = mfcc.mean(dim=-1).squeeze(0)
            norm = vec.norm(p=2).clamp_min(1e-6)
            return vec / norm

    def match(self, audio: torch.Tensor, audio_sr: int) -> Optional[Dict]:
        if self._bank is None or not self.entries:
            return None
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = AF.resample(audio, audio_sr, self.feature_sr)
        vec = self._embed(audio)
        if vec is None:
            return None

        scores = torch.mv(self._bank, vec)
        best_score, best_idx = torch.max(scores, dim=0)
        if best_score.item() < self.min_score:
            return None

        match = self.entries[int(best_idx)].copy()
        match["score"] = float(best_score.item())
        return match
