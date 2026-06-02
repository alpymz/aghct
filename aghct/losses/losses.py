"""Segmentation loss functions.

* :class:`DiceLoss`     — soft Dice (1 - DSC)
* :class:`BCEDiceLoss`  — equally-weighted BCE-with-logits + Dice

Inputs are *logits* (no sigmoid applied yet).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss (binary)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sig = torch.sigmoid(pred)
        target = target.float()
        # (B, 1, H, W) — sum over spatial dims, mean over batch
        dims = tuple(range(2, pred_sig.dim()))
        intersection = (pred_sig * target).sum(dim=dims)
        cardinality = pred_sig.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """``BCE_with_logits + Dice`` (equal weights), per proposal spec."""

    def __init__(self, smooth: float = 1.0, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = self.dice(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice
