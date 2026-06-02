"""SimCLR self-supervised pre-training for the AGHCT CNN encoder.

Pipeline::

    image → 2 random augmented views (i, j)
          → encoder  (CNN)
          → global average pool over the deepest feature map
          → projection head (MLP, 256 → 128)
          → NT-Xent contrastive loss

After training, the encoder's ``state_dict`` is saved and re-used by AGHCT;
the projection head is discarded.
"""
from __future__ import annotations

import os
from typing import Optional

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# ===========================================================================
# 1. Two-view augmentation
# ===========================================================================
def _random_resized_crop(img_size: int, scale=(0.2, 1.0)):
    """Build :class:`A.RandomResizedCrop` regardless of Albumentations version.

    * v2.x: ``RandomResizedCrop(size=(h, w), ...)``
    * v1.x: ``RandomResizedCrop(height=h, width=w, ...)``
    """
    try:
        return A.RandomResizedCrop(size=(img_size, img_size), scale=scale, p=1.0)
    except TypeError:
        return A.RandomResizedCrop(
            height=img_size, width=img_size, scale=scale, p=1.0
        )


class SimCLRAugmentation:
    """Generate two stochastic views per image (random-resized-crop + jitter)."""

    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.transform = A.Compose(
            [
                _random_resized_crop(img_size, scale=(0.2, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
                ),
                A.ToGray(p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    def __call__(self, image: np.ndarray):
        view_i = self.transform(image=image)["image"]
        view_j = self.transform(image=image)["image"]
        return view_i, view_j


# ===========================================================================
# 2. Projection head
# ===========================================================================
class ProjectionHead(nn.Module):
    """2-layer MLP — ``encoder_dim → hidden_dim → out_dim``."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================================================
# 3. NT-Xent loss
# ===========================================================================
class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy (SimCLR loss).

    Given two embedding batches ``z_i`` and ``z_j`` of shape ``(B, D)`` we
    treat ``z_i[k]`` and ``z_j[k]`` as positives and all other ``2B-2`` samples
    in the joint batch as negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        B = z_i.size(0)
        device = z_i.device

        z = torch.cat([z_i, z_j], dim=0)            # (2B, D)
        z = F.normalize(z, dim=1)

        # Cosine similarity matrix (2B, 2B)
        sim = z @ z.t() / self.temperature

        # Mask self-similarity
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_self, float("-inf"))

        # Positive pair indices: i ↔ i+B
        targets = torch.arange(2 * B, device=device)
        targets = (targets + B) % (2 * B)

        return F.cross_entropy(sim, targets)


# ===========================================================================
# 4. Trainer
# ===========================================================================
class SimCLRTrainer:
    """Run SimCLR pre-training on a given encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        config: dict,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.config = config
        self.encoder = encoder.to(self.device)

        # Detect the encoder's final-feature channel count
        encoder_dim = self._infer_encoder_dim(self.encoder)
        self.encoder_dim = encoder_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = ProjectionHead(
            in_dim=encoder_dim,
            hidden_dim=256,
            out_dim=int(config.get("projection_dim", 128)),
        ).to(self.device)

        params = list(self.encoder.parameters()) + list(self.projection.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=float(config.get("lr", 3e-4)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )

        self.criterion = NTXentLoss(temperature=float(config.get("temperature", 0.1)))
        self.amp = bool(config.get("amp", True))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.device.type == "cuda")

        self.start_epoch = 0
        self.checkpoint_path = checkpoint_path
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------
    def _infer_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "out_channels") and encoder.out_channels:
            return encoder.out_channels[-1]
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64, device=self.device)
            out = encoder(dummy)
            feat = out[-1] if isinstance(out, (list, tuple)) else out
            return feat.shape[1]

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "encoder_state_dict": self.encoder.state_dict(),
                "projection_state_dict": self.projection.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        if "projection_state_dict" in ckpt:
            self.projection.load_state_dict(ckpt["projection_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[SimCLR] resumed from epoch {self.start_epoch} ({path})")

    # ------------------------------------------------------------------
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feat = feats[-1] if isinstance(feats, (list, tuple)) else feats
        h = self.pool(feat).flatten(1)        # (B, encoder_dim)
        z = self.projection(h)                # (B, projection_dim)
        return z

    # ------------------------------------------------------------------
    def train(
        self,
        dataloader,
        epochs: int = 100,
        save_every: int = 10,
        save_dir: str = "/kaggle/working/checkpoints",
        run_name: str = "simclr_isic",
    ) -> str:
        """Train and return the path to the encoder-only checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"{run_name}_last.pth")
        encoder_path = os.path.join(save_dir, f"pretrained_encoder_{run_name}.pth")

        self.encoder.train()
        self.projection.train()

        for epoch in range(self.start_epoch, epochs):
            running_loss = 0.0
            n_batches = 0
            pbar = tqdm(dataloader, desc=f"SimCLR epoch {epoch+1}/{epochs}")
            for view_i, view_j in pbar:
                view_i = view_i.to(self.device, non_blocking=True)
                view_j = view_j.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
                    z_i = self._embed(view_i)
                    z_j = self._embed(view_j)
                    loss = self.criterion(z_i, z_j)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{running_loss / n_batches:.4f}"})

            avg = running_loss / max(1, n_batches)
            print(f"[SimCLR] epoch {epoch+1}: loss={avg:.4f}")

            # Save every N epochs + always after the last
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                self._save_checkpoint(epoch, ckpt_path)
                self.save_encoder(encoder_path)
                print(f"[SimCLR] checkpoint saved → {ckpt_path}")

        return encoder_path

    # ------------------------------------------------------------------
    def save_encoder(self, path: str) -> None:
        """Save **only** the encoder weights (projection head is discarded)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
