#!/usr/bin/env python3
"""Preprocess user question WAVs into tensors for SpeechTokenizer training."""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_mel(audio: np.ndarray, sample_rate: int, n_mels: int, hop_length: int) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80.0) / 80.0  # match SpeechTokenizer normalization
    return torch.tensor(mel, dtype=torch.float32)


def process_user_files(input_dir: Path, metadata: dict, n_mels: int, hop_ms: float) -> dict:
    sample_rate = metadata["audio_format"]["sample_rate"]
    hop_length = int(sample_rate * hop_ms / 1000)
    samples = []

    for pair in metadata.get("pairs", []):
        user_file = pair["user_file"]
        audio_path = input_dir / user_file
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        mel_tensor = compute_mel(audio, sample_rate, n_mels, hop_length)

        samples.append(
            {
                "id": pair["id"],
                "file": user_file,
                "audio": audio_tensor,
                "mel": mel_tensor,
                "audio_seconds": float(len(audio) / sample_rate),
            }
        )

    return {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "hop_ms": hop_ms,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess WAV files for SpeechTokenizer training")
    parser.add_argument("--input-dir", type=Path, default=Path("wav_files"), help="Directory containing WAV assets")
    parser.add_argument("--metadata", type=Path, default=Path("wav_files/metadata.json"), help="Metadata JSON path")
    parser.add_argument(
        "--output", type=Path, default=Path("data/tokenizer/tokenizer_dataset.pt"), help="Output .pt file"
    )
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins")
    parser.add_argument("--hop-ms", type=float, default=80.0, help="Hop size in milliseconds")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    dataset = process_user_files(args.input_dir, metadata, args.n_mels, args.hop_ms)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output)

    total_seconds = sum(sample["audio_seconds"] for sample in dataset["samples"])
    print(f"Saved {len(dataset['samples'])} samples ({total_seconds:.1f}s) to {args.output}")


+if __name__ == "__main__":
+    main()
