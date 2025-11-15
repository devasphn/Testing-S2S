#!/usr/bin/env python3
"""Extract speaker embedding from reference WAV using torchaudio's ECAPA-TDNN."""
import argparse
import sys
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    return wav


def main():
    parser = argparse.ArgumentParser(description="Extract speaker embedding")
    parser.add_argument("--wav", type=Path, default=Path("wav_files/speaker_reference_male.wav"))
    parser.add_argument("--sample-rate", type=int, default=16000, help="Resample target for encoder")
    parser.add_argument(
        "--output", type=Path, default=Path("data/speaker/speaker_embedding.pt"), help="Where to store the embedding"
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    wav = load_audio(args.wav, sample_rate=args.sample_rate).to(args.device)

    bundle = torchaudio.pipelines.SUPERB_XVECTOR
    model = bundle.get_model().to(args.device).eval()

    with torch.inference_mode():
        emb = model(wav)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = torch.nn.functional.normalize(emb, dim=-1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embedding": emb.cpu(), "sample_rate": args.sample_rate, "source": str(args.wav)}, args.output)
    print(f"Saved speaker embedding to {args.output}")


if __name__ == "__main__":
    main()
