#!/usr/bin/env python3
"""Fine-tune HybridS2SModel on tokenized user/AI pairs."""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.models.hybrid_s2s import HybridS2SModel  # noqa: E402


class PairDataset(Dataset):
    def __init__(self, data_path: Path):
        data = torch.load(data_path)
        self.pairs = data["pairs"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample = self.pairs[idx]
        return sample["user_ids"], sample["ai_ids"], sample.get("user_text"), sample.get("ai_text")


def pad_stack(seqs, pad_value=0):
    max_len = max(seq.size(0) for seq in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros_like(out, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        length = seq.size(0)
        out[i, :length] = seq
        mask[i, :length] = True
    return out, mask


def collate(samples):
    user_seqs, ai_seqs, user_texts, ai_texts = zip(*samples)
    user_batch, _ = pad_stack(user_seqs)
    ai_batch, ai_mask = pad_stack(ai_seqs)
    return user_batch, ai_batch, ai_mask, user_texts, ai_texts


def train_hybrid(
    data_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
):
    dataset = PairDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = HybridS2SModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for user_batch, ai_batch, ai_mask, _, _ in loader:
            user_batch = user_batch.to(device)
            ai_batch = ai_batch.to(device)
            ai_mask = ai_mask.to(device)

            outputs = model(user_audio_tokens=user_batch, ai_audio_tokens=ai_batch)
            speech_logits = outputs["speech_logits"]

            logits = speech_logits[:, : ai_batch.size(1), :]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ai_batch.reshape(-1),
                ignore_index=0,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        steps = len(loader)
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss/steps:.4f}")

    ckpt_path = output_dir / "hybrid_s2s_telugu.pth"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved HybridS2S checkpoint to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HybridS2S model on token pairs")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/hybrid_s2s/pairs.pt"),
        help="Token pair dataset produced by prepare_hybrid_s2s_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/hybrid_s2s"),
        help="Directory to store checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs (small dataset → overfit)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_hybrid(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
