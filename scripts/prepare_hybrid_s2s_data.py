#!/usr/bin/env python3
"""Tokenize paired user/AI WAVs for HybridS2S training."""
import argparse
import json
import sys
from pathlib import Path

import librosa
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.models.speech_tokenizer import SpeechTokenizer  # noqa: E402


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


def tokenize_pair(tokenizer: SpeechTokenizer, wav_dir: Path, pair: dict, sample_rate: int) -> dict:
    user_audio = load_audio(wav_dir / pair["user_file"], sample_rate)
    ai_audio = load_audio(wav_dir / pair["ai_file"], sample_rate)

    with torch.no_grad():
        user_ids = tokenizer.tokenize(user_audio)
        ai_ids = tokenizer.tokenize(ai_audio)

    return {
        "id": pair["id"],
        "user_file": pair["user_file"],
        "ai_file": pair["ai_file"],
        "user_ids": user_ids.squeeze(0),
        "ai_ids": ai_ids.squeeze(0),
        "user_text": pair.get("user_text"),
        "ai_text": pair.get("ai_text"),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare HybridS2S training tokens")
    parser.add_argument("--metadata", type=Path, default=Path("wav_files/metadata.json"), help="Metadata JSON path")
    parser.add_argument("--wav-dir", type=Path, default=Path("wav_files"), help="Directory containing WAVs")
    parser.add_argument(
        "--tokenizer-ckpt",
        type=Path,
        default=Path("checkpoints/tokenizer/speech_tokenizer_telugu.pth"),
        help="SpeechTokenizer checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hybrid_s2s/pairs.pt"),
        help="Output .pt file with tokenized pairs",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    sample_rate = metadata["audio_format"]["sample_rate"]

    tokenizer = SpeechTokenizer().to(args.device)
    if args.tokenizer_ckpt.exists():
        state = torch.load(args.tokenizer_ckpt, map_location=args.device)
        tokenizer.load_state_dict(state["state_dict"])
        print(f"Loaded SpeechTokenizer weights from {args.tokenizer_ckpt}")
    tokenizer.eval()

    pairs = []
    for pair in metadata.get("pairs", []):
        token_pair = tokenize_pair(tokenizer, args.wav_dir, pair, sample_rate)
        pairs.append(token_pair)
        print(
            f"Tokenized pair {pair['id']}: user_tokens={token_pair['user_ids'].shape[0]} "
            f"ai_tokens={token_pair['ai_ids'].shape[0]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_rate": sample_rate, "pairs": pairs}, args.output)
    print(f"Saved {len(pairs)} token pairs to {args.output}")


if __name__ == "__main__":
    main()
