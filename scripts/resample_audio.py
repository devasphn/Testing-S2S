#!/usr/bin/env python3
"""Batch-resample WAV files to a target sample rate with high-quality filters."""
import argparse
import json
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF


def resample_file(src: Path, dst: Path, target_sr: int) -> float:
    wav, sr = torchaudio.load(src)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = AF.resample(
            wav,
            orig_freq=sr,
            new_freq=target_sr,
            lowpass_filter_width=64,
            rolloff=0.99,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(dst, wav, target_sr)
    return wav.size(-1) / target_sr


def copy_metadata(src_meta: Path, dst_meta: Path, target_sr: int):
    data = json.loads(src_meta.read_text(encoding="utf-8"))
    audio_fmt = data.setdefault("audio_format", {})
    audio_fmt["sample_rate"] = target_sr
    audio_fmt["vocoder_sample_rate"] = target_sr
    audio_fmt["requires_resampling"] = False
    dst_meta.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Resample WAV files to target sample rate")
    parser.add_argument("--input-dir", type=Path, default=Path("wav_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("wav_files_22k"))
    parser.add_argument("--target-sr", type=int, default=22050)
    parser.add_argument("--metadata", type=Path, default=Path("wav_files/metadata.json"))
    parser.add_argument("--copy-metadata", action="store_true", help="Copy + update metadata.json to output dir")
    args = parser.parse_args()

    wav_paths = sorted(p for p in args.input_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {args.input_dir}")

    total_seconds = 0.0
    for wav_path in wav_paths:
        rel_name = wav_path.name
        dst_path = args.output_dir / rel_name
        seconds = resample_file(wav_path, dst_path, args.target_sr)
        total_seconds += seconds
        print(f"Resampled {rel_name} → {args.target_sr} Hz ({seconds:.2f}s)")

    if args.copy_metadata and args.metadata.exists():
        dst_meta = args.output_dir / args.metadata.name
        copy_metadata(args.metadata, dst_meta, args.target_sr)
        print(f"Copied metadata to {dst_meta}")

    print(f"Done. {len(wav_paths)} files, {total_seconds:.1f}s audio at {args.target_sr} Hz")


if __name__ == "__main__":
    main()
