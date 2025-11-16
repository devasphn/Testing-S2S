#!/usr/bin/env python3
"""Extract speaker embedding from reference WAV using available speaker encoders."""
import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = AF.resample(
            wav,
            orig_freq=sr,
            new_freq=sample_rate,
            lowpass_filter_width=64,
            rolloff=0.99,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    return wav


def load_torchaudio_encoder(device: str):
    try:
        pipeline = torchaudio.pipelines.SUPERB_XVECTOR
    except AttributeError:
        return None
    model = pipeline.get_model().to(device).eval()
    return {
        "backend": "torchaudio_superb_xvector",
        "model": model,
        "sample_rate": pipeline.sample_rate,
    }


def load_speechbrain_encoder(device: str):
    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError as exc:
        raise ImportError(
            "speechbrain is required for speaker embedding fallback. Install with: pip install speechbrain"
        ) from exc
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return {
        "backend": "speechbrain_ecapa",
        "model": classifier,
        "sample_rate": 16000,
    }


def get_encoder(device: str):
    encoder = load_torchaudio_encoder(device)
    if encoder is not None:
        return encoder
    print("[WARN] torchaudio SUPERB_XVECTOR unavailable, falling back to SpeechBrain ECAPA-TDNN")
    return load_speechbrain_encoder(device)


def main():
    parser = argparse.ArgumentParser(description="Extract speaker embedding")
    parser.add_argument("--wav", type=Path, default=Path("wav_files/speaker_reference_male.wav"))
    parser.add_argument("--sample-rate", type=int, default=16000, help="Resample target for encoder")
    parser.add_argument(
        "--output", type=Path, default=Path("data/speaker/speaker_embedding.pt"), help="Where to store the embedding"
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder = get_encoder(args.device)
    sample_rate = encoder["sample_rate"]
    wav = load_audio(args.wav, sample_rate=sample_rate).to(args.device)

    with torch.inference_mode():
        if encoder["backend"].startswith("torchaudio"):
            emb = encoder["model"](wav)
            if isinstance(emb, tuple):
                emb = emb[0]
        else:
            # SpeechBrain encoder returns [batch, 1, feat]
            emb = encoder["model"].encode_batch(wav)
        emb = torch.nn.functional.normalize(emb.squeeze(), dim=-1, eps=1e-12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding": emb.cpu(),
            "encoder_backend": encoder["backend"],
            "encoder_sample_rate": sample_rate,
            "source": str(args.wav),
        },
        args.output,
    )
    print(f"Saved speaker embedding to {args.output} ({encoder['backend']})")


if __name__ == "__main__":
    main()
