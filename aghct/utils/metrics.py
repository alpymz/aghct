"""Segmentation evaluation metrics.

All functions accept *logits* ``pred`` and binary ``target`` tensors of the
same shape (``(1, H, W)`` or ``(B, 1, H, W)``) and apply ``sigmoid`` + threshold
internally.
"""
from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
def _binarize(pred: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(pred) > threshold).float()


# ---------------------------------------------------------------------------
def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8
) -> torch.Tensor:
    """DSC = 2|P ∩ G| / (|P| + |G|)."""
    pred_bin = _binarize(pred, threshold)
    target = target.float()
    intersection = (pred_bin * target).sum()
    cardinality = pred_bin.sum() + target.sum()
    return (2 * intersection + eps) / (cardinality + eps)


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8
) -> torch.Tensor:
    """IoU = |P ∩ G| / |P ∪ G|."""
    pred_bin = _binarize(pred, threshold)
    target = target.float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)


def sensitivity(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8
) -> torch.Tensor:
    """Recall / Sensitivity = TP / (TP + FN)."""
    pred_bin = _binarize(pred, threshold)
    target = target.float()
    tp = (pred_bin * target).sum()
    fn = ((1 - pred_bin) * target).sum()
    return (tp + eps) / (tp + fn + eps)


def specificity(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8
) -> torch.Tensor:
    """Specificity = TN / (TN + FP)."""
    pred_bin = _binarize(pred, threshold)
    target = target.float()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    fp = (pred_bin * (1 - target)).sum()
    return (tn + eps) / (tn + fp + eps)


def hausdorff_95(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    """95-th percentile Hausdorff Distance (medpy).

    Returns ``inf`` if either mask is empty.
    """
    # Lazy import — medpy can fail to import on some environments
    try:
        from medpy.metric.binary import hd95 as medpy_hd95
    except Exception:  # pragma: no cover
        return float("nan")

    pred_np = _binarize(pred, threshold).detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)

    if pred_np.ndim == 4:
        pred_np = pred_np[0, 0]
    elif pred_np.ndim == 3:
        pred_np = pred_np[0]
    if target_np.ndim == 4:
        target_np = target_np[0, 0]
    elif target_np.ndim == 3:
        target_np = target_np[0]

    if pred_np.sum() == 0 or target_np.sum() == 0:
        return float("inf")

    try:
        return float(medpy_hd95(pred_np, target_np))
    except Exception:
        return float("inf")


def compute_all_metrics(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Dict[str, Union[float, int]]:
    """Return all 5 metrics as a plain Python ``dict``."""
    return {
        "dice": dice_coefficient(pred, target, threshold).item(),
        "iou": iou_score(pred, target, threshold).item(),
        "sensitivity": sensitivity(pred, target, threshold).item(),
        "specificity": specificity(pred, target, threshold).item(),
        "hd95": hausdorff_95(pred, target, threshold),
    }


def count_parameters(model: torch.nn.Module) -> float:
    """Trainable parameter count in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
