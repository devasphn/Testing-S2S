#!/usr/bin/env python3
"""
Test Speech Tokenizer Quality

Usage:
    python scripts/test_tokenizer.py \
        --checkpoint checkpoints/tokenizer/tokenizer_best.pt \
        --input test_audio.wav \
        --output reconstructed.wav
"""
import argparse
import sys
from pathlib import Path
import torch
import torchaudio
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_tokenizer_trainable import TrainableSpeechTokenizer


def test_reconstruction(checkpoint_path: str, input_audio: str, output_audio: str):
    """
    Test tokenizer by reconstructing audio
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using {device}")
    
    # Load model
    print(f"[MODEL] Loading checkpoint from {checkpoint_path}")
    model = TrainableSpeechTokenizer(checkpoint_path=checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Load input audio
    print(f"[INPUT] Loading audio from {input_audio}")
    waveform, sr = torchaudio.load(input_audio)
    
    # Resample to 24kHz if needed
    if sr != 24000:
        print(f"[RESAMPLE] {sr}Hz -> 24000Hz")
        resampler = torchaudio.transforms.Resample(sr, 24000)
        waveform = resampler(waveform)
        sr = 24000
    
    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    waveform = waveform.squeeze(0).to(device)
    
    print(f"[AUDIO] Duration: {waveform.size(0) / sr:.2f}s, Samples: {waveform.size(0)}")
    
    # Tokenize
    print("[TOKENIZE] Encoding audio to discrete tokens...")
    start_time = time.time()
    
    with torch.no_grad():
        tokens = model.tokenize(waveform.unsqueeze(0))  # [1, num_q, T]
    
    encode_time = time.time() - start_time
    
    print(f"[TOKENS] Shape: {tokens.shape}")
    print(f"[TOKENS] Num quantizers: {tokens.size(1)}")
    print(f"[TOKENS] Temporal length: {tokens.size(2)}")
    print(f"[TOKENS] Frame rate: {tokens.size(2) / (waveform.size(0) / sr):.1f} Hz")
    print(f"[TIME] Encoding took {encode_time:.3f}s")
    
    # Reconstruct
    print("[RECONSTRUCT] Decoding tokens back to audio...")
    start_time = time.time()
    
    with torch.no_grad():
        reconstructed = model.detokenize(tokens)  # [1, samples]
    
    decode_time = time.time() - start_time
    
    print(f"[TIME] Decoding took {decode_time:.3f}s")
    print(f"[TIME] Total RTF: {(encode_time + decode_time) / (waveform.size(0) / sr):.3f}x")
    
    # Save output
    reconstructed = reconstructed.squeeze(0).cpu()
    
    # Match length of original (vocoder may produce slightly different length)
    if reconstructed.size(0) > waveform.size(0):
        reconstructed = reconstructed[:waveform.size(0)]
    elif reconstructed.size(0) < waveform.size(0):
        import torch.nn.functional as F
        reconstructed = F.pad(reconstructed, (0, waveform.size(0) - reconstructed.size(0)))
    
    print(f"[OUTPUT] Saving to {output_audio}")
    torchaudio.save(output_audio, reconstructed.unsqueeze(0), sr)
    
    # Compute quality metrics
    waveform_cpu = waveform.cpu()
    mse = torch.mean((waveform_cpu - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(waveform_cpu - reconstructed)).item()
    
    print(f"\n{'='*60}")
    print("Quality Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  Peak: {reconstructed.abs().max():.4f}")
    print(f"{'='*60}")
    print(f"\nSuccess! Listen to both files to compare quality:")
    print(f"  Original:      {input_audio}")
    print(f"  Reconstructed: {output_audio}")
    print(f"\nGood quality indicators:")
    print(f"  - MAE < 0.1")
    print(f"  - Clear speech without artifacts")
    print(f"  - Similar loudness and tone")


def analyze_codebook_usage(checkpoint_path: str, test_dir: str, num_files: int = 100):
    """
    Analyze how well the codebooks are being utilized
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = TrainableSpeechTokenizer(checkpoint_path=checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Find test files
    test_files = list(Path(test_dir).rglob("*.flac"))[:num_files]
    print(f"[ANALYZE] Testing on {len(test_files)} files")
    
    # Track codebook usage
    codebook_size = model.codebook_size
    num_quantizers = model.num_quantizers
    usage = torch.zeros(num_quantizers, codebook_size, dtype=torch.long)
    
    with torch.no_grad():
        for audio_path in test_files:
            # Load and preprocess
            waveform, sr = torchaudio.load(audio_path)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)
            
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0).to(device)
            
            # Tokenize
            tokens = model.tokenize(waveform.unsqueeze(0))  # [1, num_q, T]
            
            # Count usage
            for q in range(num_quantizers):
                for token_id in tokens[0, q, :].cpu().numpy():
                    usage[q, token_id] += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Codebook Utilization Analysis:")
    print(f"{'='*60}")
    
    for q in range(num_quantizers):
        used = (usage[q] > 0).sum().item()
        utilization = used / codebook_size * 100
        entropy = -(usage[q].float() / usage[q].sum() * torch.log(usage[q].float() / usage[q].sum() + 1e-10)).sum().item()
        
        print(f"Quantizer {q+1}:")
        print(f"  Used codes: {used}/{codebook_size} ({utilization:.1f}%)")
        print(f"  Entropy: {entropy:.2f} bits")
        print(f"  Most used: {usage[q].topk(5).values.tolist()}")
        print()
    
    print(f"Good utilization: >70% of codebook used per quantizer")
    print(f"Poor utilization: <30% (model may need more training or larger codebook)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Speech Tokenizer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to tokenizer checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input audio file (for reconstruction test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reconstructed.wav",
        help="Output audio file",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze codebook usage instead of reconstruction",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        help="Directory with test files (for analysis)",
    )
    
    args = parser.parse_args()
    
    if args.analyze:
        if not args.test_dir:
            print("ERROR: --test_dir required for analysis mode")
            sys.exit(1)
        analyze_codebook_usage(args.checkpoint, args.test_dir)
    else:
        if not args.input:
            print("ERROR: --input required for reconstruction test")
            sys.exit(1)
        test_reconstruction(args.checkpoint, args.input, args.output)
