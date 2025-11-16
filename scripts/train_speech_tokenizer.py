#!/usr/bin/env python3
"""Minimal SpeechTokenizer training loop for Telugu POC."""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Allow running from repo root or scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.models.speech_tokenizer import SpeechTokenizer  # noqa: E402


class AudioDataset(Dataset):
    def __init__(self, dataset_path: Path):
        data = torch.load(dataset_path)
        self.sample_rate = data["sample_rate"]
        self.samples = [sample["audio"] for sample in data["samples"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pad_collate(batch):
    max_len = max(sample.size(0) for sample in batch)
    padded = torch.zeros(len(batch), max_len)
    for i, sample in enumerate(batch):
        padded[i, : sample.size(0)] = sample
    return padded


def save_checkpoint(path: Path, tokenizer: SpeechTokenizer, optimizer: torch.optim.Optimizer, epoch: int):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": tokenizer.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, tokenizer: SpeechTokenizer, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(path, map_location=tokenizer.codebook.device)
    tokenizer.load_state_dict(ckpt["state_dict"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


def train_tokenizer(
    dataset_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    device: str,
    export_wav: Path | None,
    resume_from: Path | None,
    save_every: int,
    checkpoint_prefix: str,
):
    dataset = AudioDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    tokenizer = SpeechTokenizer().to(device)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=lr)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if resume_from and resume_from.exists():
        start_epoch = load_checkpoint(resume_from, tokenizer, optimizer) + 1
        print(f"Resumed SpeechTokenizer from {resume_from} (starting epoch {start_epoch})")

    for epoch in range(start_epoch, epochs + 1):
        tokenizer.train()
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0

        for batch in loader:
            audio = batch.to(device)
            mel = tokenizer.audio_to_mel(audio)
            h = tokenizer.enc(mel)
            h = h.transpose(1, 2)
            h = tokenizer.enc_tf(h)
            q, _ = tokenizer.quantize(h)
            mel_recon = tokenizer.decode(q)

            recon_loss = F.l1_loss(mel_recon, mel)
            vq_loss = F.mse_loss(h.detach(), q) + beta * F.mse_loss(h, q.detach())
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        steps = len(loader)
        print(
            f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss/steps:.4f} | "
            f"Recon: {total_recon/steps:.4f} | VQ: {total_vq/steps:.4f}"
        )

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = output_dir / f"{checkpoint_prefix}_epoch{epoch:04d}.pth"
            save_checkpoint(ckpt_path, tokenizer, optimizer, epoch)
            print(f"Saved checkpoint: {ckpt_path}")

    if export_wav is not None:
        tokenizer.eval()
        with torch.no_grad():
            audio = dataset[0].unsqueeze(0).to(device)
            ids = tokenizer.tokenize(audio)
            recon = tokenizer.detokenize(ids).squeeze(0).cpu().numpy()
        export_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(export_wav, recon, dataset.sample_rate)
        print(f"Wrote reconstruction sample to {export_wav}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpeechTokenizer on prepared dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/tokenizer/tokenizer_dataset.pt"),
        help="Path to .pt file from prepare_tokenizer_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/tokenizer"),
        help="Directory to store checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.25, help="VQ commitment weight")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--export-wav",
        type=Path,
        default=Path("data/tokenizer/recon_sample.wav"),
        help="Optional path to save a reconstructed sample",
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint to resume training from")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="speech_tokenizer_telugu",
        help="Prefix for checkpoint filenames",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        device=args.device,
        export_wav=args.export_wav,
        resume_from=args.resume_from,
        save_every=args.save_every,
        checkpoint_prefix=args.checkpoint_prefix,
    )
