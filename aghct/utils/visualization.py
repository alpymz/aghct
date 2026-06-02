"""Visualisation helpers for qualitative results + DAGM attention gates."""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """ImageNet-normalized ``(3, H, W)`` tensor → uint8 ``(H, W, 3)`` ndarray."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def save_qualitative_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    save_dir: str,
    prefix: str = "sample",
    threshold: float = 0.5,
    n: int = 5,
) -> List[str]:
    """Save up to ``n`` 1×3 figures: image | ground-truth | prediction overlay."""
    os.makedirs(save_dir, exist_ok=True)
    n = min(n, images.shape[0])
    paths: List[str] = []

    for i in range(n):
        img = denormalize_image(images[i])
        gt = masks[i].detach().cpu().numpy()
        if gt.ndim == 3:
            gt = gt[0]
        pr_logits = preds[i]
        if pr_logits.ndim == 3:
            pr_logits = pr_logits[0]
        pr = (torch.sigmoid(pr_logits).detach().cpu().numpy() > threshold).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Image"); axes[0].axis("off")

        axes[1].imshow(img)
        axes[1].imshow(gt, cmap="Reds", alpha=0.5)
        axes[1].set_title("Ground truth"); axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].imshow(pr, cmap="Blues", alpha=0.5)
        axes[2].set_title("Prediction"); axes[2].axis("off")

        out_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)

    return paths


# ---------------------------------------------------------------------------
def save_attention_heatmaps(
    image: torch.Tensor,
    gates: List[Optional[torch.Tensor]],
    save_path: str,
) -> str:
    """Visualise DAGM gate ``α`` maps from each encoder scale.

    Each gate is mean-pooled over channels then resized to the image size.
    """
    img = denormalize_image(image)
    H, W = img.shape[:2]

    valid = [(i, g) for i, g in enumerate(gates) if g is not None]
    if not valid:
        return ""

    fig, axes = plt.subplots(1, len(valid) + 1, figsize=(4 * (len(valid) + 1), 4))
    axes[0].imshow(img); axes[0].set_title("Image"); axes[0].axis("off")

    for col, (lvl, g) in enumerate(valid, start=1):
        gate = g.mean(dim=1, keepdim=True)            # (B,1,h,w)
        gate = F.interpolate(gate, size=(H, W), mode="bilinear", align_corners=False)
        gate_np = gate[0, 0].cpu().numpy()
        axes[col].imshow(img)
        axes[col].imshow(gate_np, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[col].set_title(f"DAGM α — scale {lvl}")
        axes[col].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path
