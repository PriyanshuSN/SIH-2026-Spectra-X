"""
train.py — Training loop for SwinIR-Lite.

Supports:
  - L1 loss (primary) + optional SSIM loss
  - Checkpoint saving every epoch
  - Mixed precision training (fp16) for faster free-tier GPU training
  - Logging to CSV
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model.swinir import build_model


class DegradationDataset(Dataset):
    """
    Dataset that loads pre-computed (LR, HR) patch pairs from disk.

    Expected directory structure:
        data_dir/
            lr/  -> low-resolution patches (*.npy)
            hr/  -> high-resolution patches (*.npy)
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.lr_dir = self.data_dir / "lr"
        self.hr_dir = self.data_dir / "hr"
        self.filenames = sorted(os.listdir(self.lr_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        hr = np.load(self.hr_dir / fname).astype(np.float32)
        return torch.from_numpy(lr), torch.from_numpy(hr)


def train(
    data_dir: str,
    output_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 2e-4,
    device: str = "auto",
    model_config: dict | None = None,
):
    """
    Train the SwinIR-Lite model.

    Args:
        data_dir: Path to directory with lr/ and hr/ subdirectories.
        output_dir: Where to save checkpoints.
        epochs: Number of training epochs.
        batch_size: Batch size (keep small for free-tier GPU).
        lr: Learning rate.
        device: 'cuda', 'cpu', or 'auto' (auto-detect).
        model_config: Optional model config overrides.
    """
    # Setup
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Model
    model = build_model(model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.2f}M")

    # Data
    dataset = DegradationDataset(data_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Training log
    log_path = os.path.join(output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "lr", "time_s"])

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for batch_idx, (lr_patches, hr_patches) in enumerate(dataloader):
            lr_patches = lr_patches.to(device)
            hr_patches = hr_patches.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                sr_output = model(lr_patches)
                loss = criterion(sr_output, hr_patches)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Loss: {avg_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, scheduler.get_last_lr()[0], elapsed])

        # Checkpoint EVERY epoch (critical for free-tier GPU sessions)
        ckpt_path = os.path.join(output_dir, f"swinir_epoch{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            },
            ckpt_path,
        )
        print(f"  -> Saved checkpoint: {ckpt_path}")

    print("Training complete!")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SwinIR-Lite")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
