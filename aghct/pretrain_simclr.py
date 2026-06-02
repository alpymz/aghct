"""SimCLR pre-training entry point.

Usage::

    python pretrain_simclr.py --dataset isic --config configs/config.yaml
    python pretrain_simclr.py --dataset drive --config configs/config.yaml

Saves the encoder weights to::

    /kaggle/working/checkpoints/pretrained_encoder_simclr_<dataset>.pth
"""
from __future__ import annotations

import argparse
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow `python pretrain_simclr.py` from either project root or inside aghct/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from data.isic_dataset import ISICUnlabeledDataset  # noqa: E402
from data.drive_dataset import DRIVEUnlabeledDataset  # noqa: E402
from models.cnn_encoder import CNNEncoder  # noqa: E402
from pretrain.simclr import SimCLRAugmentation, SimCLRTrainer  # noqa: E402


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
def build_dataset(dataset: str, config: dict):
    if dataset == "isic":
        root = config["dataset"]["isic"]["root"]
        img_size = int(config["dataset"]["isic"]["img_size"])
        aug = SimCLRAugmentation(img_size=img_size)
        return ISICUnlabeledDataset(root_dir=root, augmentation=aug)
    elif dataset == "drive":
        root = config["dataset"]["drive"]["root"]
        # SimCLR uses dense crops at a smaller patch size for retinal images
        aug = SimCLRAugmentation(img_size=128)
        return DRIVEUnlabeledDataset(
            root_dir=root,
            augmentation=aug,
            patch_size=128,
            stride=64,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SimCLR pre-training for AGHCT encoder")
    parser.add_argument("--dataset", choices=["isic", "drive"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("seed", 42)))
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[pretrain_simclr] CUDA not available, falling back to CPU")
        device = "cpu"

    pre_cfg = config["pretrain"]
    pre_cfg.setdefault("amp", config["training"].get("amp", True))

    # --- Dataset / Loader -------------------------------------------------
    dataset = build_dataset(args.dataset, config)
    loader = DataLoader(
        dataset,
        batch_size=int(pre_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(config["kaggle"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )

    # --- Encoder ----------------------------------------------------------
    encoder = CNNEncoder(
        in_channels=3,
        base_channels=int(config["model"]["base_channels"]),
    )

    # --- Trainer ----------------------------------------------------------
    save_dir = config["checkpoint"]["save_dir"]
    run_name = f"simclr_{args.dataset}"
    resume_ckpt = os.path.join(save_dir, f"{run_name}_last.pth") if args.resume else None

    trainer = SimCLRTrainer(
        encoder=encoder,
        config=pre_cfg,
        device=device,
        checkpoint_path=resume_ckpt,
    )

    encoder_path = trainer.train(
        dataloader=loader,
        epochs=int(pre_cfg["epochs"]),
        save_every=max(1, int(pre_cfg.get("save_every", 10))),
        save_dir=save_dir,
        run_name=run_name,
    )

    print(f"\n[pretrain_simclr] Encoder weights saved → {encoder_path}")
    # Also save a stable name for the train.py loader
    final_name = os.path.join(save_dir, f"pretrained_encoder_{args.dataset}.pth")
    if encoder_path != final_name:
        import shutil

        shutil.copy2(encoder_path, final_name)
        print(f"[pretrain_simclr] Mirror → {final_name}")


if __name__ == "__main__":
    main()
