#!/usr/bin/env python3
"""
Training script for Speech Tokenizer

Usage:
    python training/train_tokenizer.py --config training/configs/tokenizer_config.yaml

Dataset:
    LibriSpeech train-clean-100 (100 hours, free download)
    Download: http://www.openslr.org/resources/12/train-clean-100.tar.gz
"""
import argparse
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from tqdm import tqdm
import wandb
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_tokenizer_trainable import TrainableSpeechTokenizer


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset for speech tokenizer training
    
    Expected directory structure:
        data_dir/
            LibriSpeech/
                train-clean-100/
                    19/
                        198/
                            19-198-0000.flac
                            19-198-0001.flac
                            ...
                dev-clean/
                    ...
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train-clean-100",
        sample_rate: int = 24000,
        duration: float = 3.0,
        max_files: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sr = sample_rate
        self.duration = duration
        self.samples = int(sample_rate * duration)
        
        # Find all audio files
        split_dir = self.data_dir / "LibriSpeech" / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {split_dir}\n"
                f"Download with: wget http://www.openslr.org/resources/12/{split}.tar.gz"
            )
        
        self.audio_files = sorted(list(split_dir.rglob("*.flac")))
        
        if max_files:
            self.audio_files = self.audio_files[:max_files]
        
        print(f"[DATASET] Loaded {len(self.audio_files)} files from {split}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0)
            
            # Random crop or pad to fixed duration
            if waveform.size(0) > self.samples:
                start = torch.randint(0, waveform.size(0) - self.samples, (1,)).item()
                waveform = waveform[start:start + self.samples]
            else:
                # Pad with zeros
                waveform = F.pad(waveform, (0, self.samples - waveform.size(0)))
            
            return waveform
        
        except Exception as e:
            print(f"[ERROR] Failed to load {audio_path}: {e}")
            # Return silence on error
            return torch.zeros(self.samples)


def compute_losses(outputs: dict, config: dict) -> dict:
    """
    Compute training losses
    
    Args:
        outputs: Forward pass outputs from tokenizer
        config: Loss configuration
    
    Returns:
        Dictionary of losses
    """
    mel_orig = outputs["mel_original"]
    mel_recon = outputs["mel_reconstructed"]
    audio_orig = None  # We don't store original audio to save memory
    audio_recon = outputs["audio_reconstructed"]
    commitment = outputs["commitment_loss"]
    
    # 1. Mel reconstruction loss (L1 + MSE)
    loss_mel_l1 = F.l1_loss(mel_recon, mel_orig)
    loss_mel_mse = F.mse_loss(mel_recon, mel_orig)
    loss_mel = loss_mel_l1 + 0.5 * loss_mel_mse
    
    # 2. Commitment loss (VQ-VAE style)
    loss_commit = commitment * config.get("commitment_weight", 0.25)
    
    # 3. Total loss
    total_loss = loss_mel + loss_commit
    
    return {
        "total": total_loss,
        "mel_l1": loss_mel_l1,
        "mel_mse": loss_mel_mse,
        "commitment": loss_commit,
    }


def train_epoch(
    model: TrainableSpeechTokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    config: dict,
    epoch: int,
):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    mel_l1_loss = 0
    mel_mse_loss = 0
    commit_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, audio in enumerate(pbar):
        audio = audio.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(audio)
        
        # Compute losses
        losses = compute_losses(outputs, config["training"])
        
        # Backward pass
        losses["total"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            config["training"].get("grad_clip", 1.0)
        )
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses["total"].item()
        mel_l1_loss += losses["mel_l1"].item()
        mel_mse_loss += losses["mel_mse"].item()
        commit_loss += losses["commitment"].item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "mel_l1": f"{losses['mel_l1'].item():.4f}",
            "commit": f"{losses['commitment'].item():.4f}",
        })
        
        # Log to wandb
        if config["logging"]["use_wandb"] and batch_idx % 10 == 0:
            wandb.log({
                "train/loss": losses["total"].item(),
                "train/mel_l1": losses["mel_l1"].item(),
                "train/mel_mse": losses["mel_mse"].item(),
                "train/commitment": losses["commitment"].item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            })
    
    # Average losses
    n_batches = len(dataloader)
    return {
        "total": total_loss / n_batches,
        "mel_l1": mel_l1_loss / n_batches,
        "mel_mse": mel_mse_loss / n_batches,
        "commitment": commit_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: TrainableSpeechTokenizer,
    dataloader: DataLoader,
    device: str,
    config: dict,
):
    """Validation pass"""
    model.eval()
    
    total_loss = 0
    mel_l1_loss = 0
    mel_mse_loss = 0
    commit_loss = 0
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for audio in pbar:
        audio = audio.to(device)
        
        # Forward pass
        outputs = model(audio)
        
        # Compute losses
        losses = compute_losses(outputs, config["training"])
        
        # Accumulate
        total_loss += losses["total"].item()
        mel_l1_loss += losses["mel_l1"].item()
        mel_mse_loss += losses["mel_mse"].item()
        commit_loss += losses["commitment"].item()
        
        pbar.set_postfix({"val_loss": f"{losses['total'].item():.4f}"})
    
    # Average losses
    n_batches = len(dataloader)
    return {
        "total": total_loss / n_batches,
        "mel_l1": mel_l1_loss / n_batches,
        "mel_mse": mel_mse_loss / n_batches,
        "commitment": commit_loss / n_batches,
    }


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using {device}")
    
    # Initialize wandb
    if config["logging"]["use_wandb"]:
        wandb.init(
            project=config["logging"]["project"],
            name=config["logging"]["run_name"] + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
        )
    
    # Create model
    model = TrainableSpeechTokenizer(
        n_mels=config["model"]["n_mels"],
        sample_rate=config["model"]["sample_rate"],
        hop_ms=config["model"]["hop_ms"],
        codebook_size=config["model"]["codebook_size"],
        hidden_dim=config["model"]["hidden_dim"],
        num_quantizers=config["model"]["num_quantizers"],
    ).to(device)
    
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Load datasets
    full_dataset = LibriSpeechDataset(
        data_dir=config["data"]["data_dir"],
        split=config["data"]["train_split"],
        sample_rate=config["model"]["sample_rate"],
        duration=config["data"]["duration"],
        max_files=config["data"].get("max_files"),
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * config["data"]["val_ratio"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.5, 0.9),
        weight_decay=config["training"].get("weight_decay", 0.01),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=config["training"].get("min_lr", 1e-6),
    )
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config["training"]["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, config, epoch)
        
        # Validate
        val_losses = validate(model, val_loader, device, config)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val Loss:   {val_losses['total']:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")
        
        # Log to wandb
        if config["logging"]["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "val/loss": val_losses["total"],
                "val/mel_l1": val_losses["mel_l1"],
                "val/mel_mse": val_losses["mel_mse"],
                "val/commitment": val_losses["commitment"],
                "lr": optimizer.param_groups[0]["lr"],
            })
        
        # Save checkpoint
        if (epoch + 1) % config["training"]["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"tokenizer_epoch_{epoch+1}.pt"
            model.save_checkpoint(
                str(checkpoint_path),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                train_loss=train_losses["total"],
                val_loss=val_losses["total"],
            )
        
        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_path = checkpoint_dir / "tokenizer_best.pt"
            model.save_checkpoint(
                str(best_path),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                train_loss=train_losses["total"],
                val_loss=val_losses["total"],
            )
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    # Save final model
    final_path = checkpoint_dir / "tokenizer_final.pt"
    model.save_checkpoint(
        str(final_path),
        epoch=config["training"]["epochs"] - 1,
        optimizer_state=optimizer.state_dict(),
    )
    print(f"\n[TRAINING COMPLETE] Final model saved to {final_path}")
    
    if config["logging"]["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Tokenizer")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/tokenizer_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    
    main(args)
