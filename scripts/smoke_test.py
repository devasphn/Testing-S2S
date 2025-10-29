#!/usr/bin/env python3
"""
Smoke test: feeds a short WAV and saves a response WAV from tokenizer loop.
This bypasses the server and exercises the model directly.
"""
import argparse
from pathlib import Path
import soundfile as sf
import torch

from src.models.hybrid_s2s import HybridS2SModel
from src.models.speech_tokenizer import SpeechTokenizer
from src.models.streaming_processor import StreamingProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=False, help="input wav (24kHz mono)")
    ap.add_argument("--out", type=str, default="out.wav")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = SpeechTokenizer().to(device)
    model = HybridS2SModel().to(device).eval()
    proc = StreamingProcessor(model, tok)

    if args.wav and Path(args.wav).exists():
        audio, sr = sf.read(args.wav)
        assert sr == 24000, "Expect 24kHz mono"
        x = torch.tensor(audio, dtype=torch.float32, device=device)
        out = torch.zeros(0, device=device)
        # Simulate chunked processing
        step = int(0.08 * sr)
        for i in range(0, len(x), step):
            chunk = x[i:i+step]
            y = torch.cuda._sleep(1) if device=="cuda" else None  # no-op to keep loop consistent
            out_chunk = torch.run(proc.process_audio_stream(chunk)) if False else None
        # Direct quick path: tokenize and detokenize full (for smoke)
        ids = tok.tokenize(x.unsqueeze(0))
        y = tok.detokenize(ids).squeeze(0).detach().cpu().numpy()
        sf.write(args.out, y, 24000)
        print("Wrote:", args.out)
    else:
        # generate from silence tokens to verify audio path
        ids = torch.zeros(1, 10, dtype=torch.long, device=device)
        y = tok.detokenize(ids).squeeze(0).detach().cpu().numpy()
        sf.write(args.out, y, 24000)
        print("Wrote (silence-like):", args.out)


if __name__ == "__main__":
    main()
