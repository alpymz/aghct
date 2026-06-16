"""Supervised training script — AGHCT, U-Net, TransUNet.

Examples (Kaggle)::

    python train.py --config configs/config.yaml --model aghct  --dataset isic --fold 0
    python train.py --config configs/config.yaml --model unet   --dataset isic --fold 0 --fraction 0.25
    python train.py --config configs/config.yaml --model aghct  --dataset isic --fold 0 --resume

Notes:
* Checkpoints are saved every ``checkpoint.save_every`` epochs **and** whenever
  a new best validation Dice is reached. ``--resume`` automatically continues
  from ``<run>_last.pth``.
* Mixed precision is enabled if ``training.amp: true`` in the config.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Optional, Tuple

# Make stdout UTF-8 safe on Windows consoles (cp1254 etc.). Harmless on Linux.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from data.isic_dataset import ISICDataset  # noqa: E402
from data.drive_dataset import DRIVEDataset  # noqa: E402
from data.augmentations import (  # noqa: E402
    get_train_transforms,
    get_val_transforms,
    get_drive_train_transforms,
    get_drive_val_transforms,
)
from losses.losses import BCEDiceLoss  # noqa: E402
from models.aghct import AGHCT  # noqa: E402
from models.unet import UNet  # noqa: E402
from utils.metrics import dice_coefficient, iou_score, count_parameters  # noqa: E402


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
def build_model(model_name: str, config: dict) -> nn.Module:
    if model_name == "aghct":
        return AGHCT(config["model"])
    if model_name == "unet":
        return UNet(base_channels=config["model"].get("base_channels", 64))
    if model_name == "transunet":
        from models.transunet import TransUNet  # lazy: requires timm

        return TransUNet()
    if model_name == "attention_unet":
        from models.attention_unet import AttentionUNet
        return AttentionUNet(base_channels=config["model"].get("base_channels", 64))
    raise ValueError(f"Unknown model: {model_name}")


def build_dataset(
    dataset_name: str, config: dict, split: str, fold: int, fraction: float, transform
):
    if dataset_name == "isic":
        cfg = config["dataset"]["isic"]
        return ISICDataset(
            root_dir=cfg["root"],
            split=split,
            transform=transform,
            data_fraction=fraction,
            seed=int(config.get("seed", 42)),
            fold=fold,
            n_folds=int(config.get("cv_folds", 5)),
        )
    if dataset_name == "drive":
        cfg = config["dataset"]["drive"]
        return DRIVEDataset(
            root_dir=cfg["root"],
            split=split if split != "val" else "test",
            transform=transform,
            use_green_channel=bool(cfg.get("use_green_channel", True)),
            data_fraction=fraction,
            seed=int(config.get("seed", 42)),
            fold=fold,
            n_folds=int(config.get("cv_folds", 5)),
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_transforms(dataset_name: str, img_size: int):
    if dataset_name == "isic":
        return get_train_transforms(img_size), get_val_transforms(img_size)
    if dataset_name == "drive":
        return get_drive_train_transforms(img_size), get_drive_val_transforms(img_size)
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 1e-3,
) -> LambdaLR:
    """Linear warmup then cosine annealing (factor returned per epoch)."""

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float = 0.0,
    log_prefix: str = "train",
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=log_prefix, leave=False)
    use_amp = scaler is not None

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(images)
            loss = criterion(preds, masks)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    n_batches = 0
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        preds = model(images)
        total_loss += criterion(preds, masks).item()
        total_dice += dice_coefficient(preds, masks).item()
        total_iou += iou_score(preds, masks).item()
        n_batches += 1

    n = max(1, n_batches)
    return total_loss / n, total_dice / n, total_iou / n


# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            pass
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_dice = float(ckpt.get("best_dice", 0.0))
    return start_epoch, best_dice


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["aghct", "unet", "transunet", "attention_unet"], required=True)
    parser.add_argument("--dataset", choices=["isic", "drive"], required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--no-pretrain", action="store_true",
        help="Train AGHCT from scratch (skip SimCLR encoder loading)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override config epochs (handy for quick smoke tests)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("seed", 42)))

    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA not available → CPU")
        device = torch.device("cpu")

    # -------------------- Paths --------------------
    ckpt_dir = config["checkpoint"]["save_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = f"{args.model}_{args.dataset}_fold{args.fold}_frac{args.fraction}"
    if args.no_pretrain and args.model == "aghct":
        run_name += "_nopretrain"
    last_ckpt = os.path.join(ckpt_dir, f"{run_name}_last.pth")
    best_ckpt = os.path.join(ckpt_dir, f"{run_name}_best.pth")

    # -------------------- Data --------------------
    ds_cfg = config["dataset"][args.dataset]
    img_size = int(ds_cfg["img_size"])
    batch_size = int(ds_cfg["batch_size"])
    train_tf, val_tf = build_transforms(args.dataset, img_size)

    train_ds = build_dataset(args.dataset, config, "train", args.fold, args.fraction, train_tf)
    val_ds = build_dataset(args.dataset, config, "val", args.fold, 1.0, val_tf)

    num_workers = int(config["kaggle"]["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # -------------------- Model --------------------
    model = build_model(args.model, config).to(device)
    print(f"[train] Model={args.model} params={count_parameters(model):.2f} M")

    if args.model == "aghct" and config["pretrain"].get("enabled", False) and not args.no_pretrain:
        pretrain_path = os.path.join(ckpt_dir, f"pretrained_encoder_{args.dataset}.pth")
        if os.path.exists(pretrain_path):
            model.load_pretrained_encoder(pretrain_path)
        else:
            print(f"[train] no pre-trained encoder found at {pretrain_path} — training from scratch")

    # -------------------- Loss / optimizer / scheduler --------------------
    criterion = BCEDiceLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    total_epochs = (
        int(args.epochs) if args.epochs is not None
        else int(config["training"]["epochs"]) if args.fraction == 1.0
        else int(config["training"]["low_data_epochs"])
    )
    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=int(config["training"].get("warmup_epochs", 0)),
        total_epochs=total_epochs,
    )

    use_amp = bool(config["training"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # -------------------- Resume --------------------
    start_epoch, best_dice = 0, 0.0
    if args.resume and os.path.exists(last_ckpt):
        start_epoch, best_dice = load_checkpoint(
            last_ckpt, model, optimizer, scheduler, scaler
        )
        print(f"[train] Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    # -------------------- Loop --------------------
    grad_clip = float(config["training"].get("grad_clip", 0.0))
    save_every = int(config["checkpoint"].get("save_every", 5))

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch+1}/{total_epochs}] lr={lr_now:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip=grad_clip,
            log_prefix=f"train e{epoch+1}",
        )
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        dt = time.time() - t0
        print(
            f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | {dt:.1f}s"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_dice": best_dice,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "config": config,
            "args": vars(args),
        }

        if (epoch + 1) % save_every == 0 or (epoch + 1) == total_epochs:
            save_checkpoint(state, last_ckpt)

        if val_dice > best_dice:
            best_dice = val_dice
            state["best_dice"] = best_dice
            save_checkpoint(state, best_ckpt)
            print(f"  * new best dice = {best_dice:.4f}")

    print(f"\n[train] Done. Best dice={best_dice:.4f}")
    print(f"[train] Best checkpoint: {best_ckpt}")
    print(f"[train] Last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
