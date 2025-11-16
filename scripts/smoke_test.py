#!/usr/bin/env python3
"""
Smoke test: feeds a short WAV and saves a response WAV from tokenizer loop.
This bypasses the server and exercises the model directly.
"""
import argparse
from pathlib import Path
import os
import soundfile as sf
import torch
import torchaudio.functional as AF

from src.models.hybrid_s2s import HybridS2SModel
from src.models.speech_tokenizer import SpeechTokenizer
from src.models.streaming_processor import StreamingProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=False, help="input wav (24kHz mono)")
    ap.add_argument("--out", type=str, default="out.wav")
    ap.add_argument("--tokenizer-ckpt", type=Path, default=Path(os.getenv("TOKENIZER_CKPT", "checkpoints/tokenizer/speech_tokenizer_telugu_epoch0200.pth")))
    ap.add_argument("--model-ckpt", type=Path, default=Path(os.getenv("HYBRID_S2S_CKPT", "checkpoints/hybrid_s2s/hybrid_s2s_telugu.pth")))
    ap.add_argument("--speaker-emb", type=Path, default=Path(os.getenv("SPEAKER_EMB", "data/speaker/speaker_embedding.pt")))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = SpeechTokenizer().to(device)
    if args.tokenizer_ckpt.exists():
        state = torch.load(args.tokenizer_ckpt, map_location=device)
        tok.load_state_dict(state.get("state_dict", state))
        print(f"[INFO] Loaded SpeechTokenizer from {args.tokenizer_ckpt}")
    else:
        print(f"[WARN] Tokenizer checkpoint missing at {args.tokenizer_ckpt}; using random weights")

    model = HybridS2SModel().to(device).eval()
    if args.model_ckpt.exists():
        state = torch.load(args.model_ckpt, map_location=device)
        model.load_state_dict(state.get("state_dict", state))
        model.eval()
        print(f"[INFO] Loaded HybridS2SModel from {args.model_ckpt}")
    else:
        print(f"[WARN] HybridS2S checkpoint missing at {args.model_ckpt}; using random weights")

    speaker_embedding = None
    if args.speaker_emb.exists():
        data = torch.load(args.speaker_emb, map_location="cpu")
        speaker_embedding = data.get("embedding")
        print(f"[INFO] Loaded speaker embedding from {args.speaker_emb}")
    else:
        print(f"[WARN] Speaker embedding missing at {args.speaker_emb}; proceeding without conditioning")

    proc = StreamingProcessor(
        model,
        tok,
        sample_rate=24000,
        reply_mode="stream",
        speaker_embedding=speaker_embedding,
    )

    tokenizer_sr = getattr(tok, "sr", 22050)
    transport_sr = 24000

    def _resample(x: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        if src == dst:
            return x
        needs_batch = x.dim() == 1
        if needs_batch:
            x = x.unsqueeze(0)
        x = AF.resample(
            x,
            orig_freq=src,
            new_freq=dst,
            lowpass_filter_width=64,
            rolloff=0.99,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
        return x.squeeze(0) if needs_batch else x

    if args.wav and Path(args.wav).exists():
        audio, sr = sf.read(args.wav)
        x = torch.tensor(audio, dtype=torch.float32)
        x = _resample(x, sr, tokenizer_sr).to(device)
        ids = tok.tokenize(x.unsqueeze(0))
        y = tok.detokenize(ids).squeeze(0)
        y = _resample(y, getattr(tok.vocoder, "sample_rate", tokenizer_sr), transport_sr)
        y = y.detach().cpu().numpy()
        sf.write(args.out, y, transport_sr)
        print("Wrote:", args.out)
    else:
        # generate from silence tokens to verify audio path
        ids = torch.zeros(1, 10, dtype=torch.long, device=device)
        y = tok.detokenize(ids).squeeze(0)
        y = _resample(y, getattr(tok.vocoder, "sample_rate", tokenizer_sr), transport_sr)
        sf.write(args.out, y.detach().cpu().numpy(), transport_sr)
        print("Wrote (silence-like):", args.out)


if __name__ == "__main__":
    main()
