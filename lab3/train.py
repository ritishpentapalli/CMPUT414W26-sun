import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import J2KPairDataset
from losses import DualDomainLoss
from metrics import psnr, ssim
from models import UNetRestorer, ViTRestorer


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "unet":
        return UNetRestorer(base_channels=args.base_channels)
    if args.model == "vit":
        return ViTRestorer(
            image_size=args.patch_size,
            patch_size=args.vit_patch,
            embed_dim=args.vit_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            mlp_ratio=args.vit_mlp_ratio,
            dropout=args.vit_dropout,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_spatial = 0.0
    total_freq = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)

            pred = model(x)
            losses = criterion(pred, y)

            total_loss += losses.total.item()
            total_spatial += losses.spatial.item()
            total_freq += losses.frequency.item()
            total_psnr += psnr(pred, y).item()
            total_ssim += ssim(pred, y).item()

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "spatial": total_spatial / n,
        "frequency": total_freq / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
    }


def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = J2KPairDataset(
        data_root=args.data_root,
        split="train",
        patch_size=args.patch_size,
        train=True,
    )
    val_ds = J2KPairDataset(
        data_root=args.data_root,
        split="validate",
        patch_size=args.patch_size,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))
    criterion = DualDomainLoss(alpha=args.alpha, beta=args.beta, phase_weight=args.phase_weight)

    log_path = Path(args.output_dir) / f"{args.model}_log.csv"
    ckpt_path = Path(args.output_dir) / f"{args.model}_best.pt"

    best_psnr = -1.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_spatial",
                "train_frequency",
                "val_loss",
                "val_psnr",
                "val_ssim",
                "time_sec",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            start = time.time()

            run_loss = 0.0
            run_spatial = 0.0
            run_freq = 0.0

            for batch in train_loader:
                x = batch["input"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and args.amp)):
                    pred = model(x)
                    losses = criterion(pred, y)

                scaler.scale(losses.total).backward()
                scaler.step(optimizer)
                scaler.update()

                run_loss += losses.total.item()
                run_spatial += losses.spatial.item()
                run_freq += losses.frequency.item()

            train_n = max(len(train_loader), 1)
            train_loss = run_loss / train_n
            train_spatial = run_spatial / train_n
            train_freq = run_freq / train_n

            val_stats = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - start

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_spatial:.6f}",
                    f"{train_freq:.6f}",
                    f"{val_stats['loss']:.6f}",
                    f"{val_stats['psnr']:.4f}",
                    f"{val_stats['ssim']:.4f}",
                    f"{elapsed:.2f}",
                ]
            )
            f.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.4f} (sp={train_spatial:.4f}, fr={train_freq:.4f}) | "
                f"val={val_stats['loss']:.4f} | "
                f"PSNR={val_stats['psnr']:.2f} | SSIM={val_stats['ssim']:.4f}"
            )

            if val_stats["psnr"] > best_psnr:
                best_psnr = val_stats["psnr"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_psnr": best_psnr,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 3 training for JPEG2000 artifact removal")

    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--model", type=str, default="unet", choices=["unet", "vit"])
    parser.add_argument("--patch_size", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--phase_weight", type=float, default=0.5)

    parser.add_argument("--base_channels", type=int, default=64)

    parser.add_argument("--vit_patch", type=int, default=8)
    parser.add_argument("--vit_dim", type=int, default=256)
    parser.add_argument("--vit_depth", type=int, default=6)
    parser.add_argument("--vit_heads", type=int, default=8)
    parser.add_argument("--vit_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--vit_dropout", type=float, default=0.0)

    args = parser.parse_args()

    if args.patch_size % args.vit_patch != 0:
        raise ValueError("--patch_size must be divisible by --vit_patch")

    train(args)
