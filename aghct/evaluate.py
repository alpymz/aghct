"""Evaluation script — produces metrics, qualitative figures, attention maps.

Examples::

    python evaluate.py --config configs/config.yaml --model aghct \
        --checkpoint /kaggle/working/checkpoints/aghct_isic_fold0_frac1.0_best.pth \
        --dataset isic

Outputs (in ``--results-dir`` which defaults to ``logging.results_dir``)::

    metrics_<run>.json        Mean ± std of DSC, IoU, Sens, Spec, HD95
    qualitative/<run>/        Up to ``--n-samples`` overlay figures
    attention/<run>/          DAGM ``α`` heat-maps (AGHCT only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from data.isic_dataset import ISICDataset  # noqa: E402
from data.drive_dataset import DRIVEDataset  # noqa: E402
from data.augmentations import (  # noqa: E402
    get_val_transforms,
    get_drive_val_transforms,
)
from models.aghct import AGHCT  # noqa: E402
from models.unet import UNet  # noqa: E402
from utils.metrics import compute_all_metrics, count_parameters  # noqa: E402
from utils.visualization import save_qualitative_grid, save_attention_heatmaps  # noqa: E402


# ---------------------------------------------------------------------------
def build_model(model_name: str, config: dict) -> torch.nn.Module:
    if model_name == "aghct":
        return AGHCT(config["model"])
    if model_name == "unet":
        return UNet(base_channels=config["model"].get("base_channels", 64))
    if model_name == "transunet":
        from models.transunet import TransUNet  # lazy: requires timm

        return TransUNet()
    raise ValueError(f"Unknown model: {model_name}")


def build_dataset(dataset_name: str, config: dict, split: str, fold: int, transform):
    if dataset_name == "isic":
        cfg = config["dataset"]["isic"]
        return ISICDataset(
            root_dir=cfg["root"],
            split=split,
            transform=transform,
            data_fraction=1.0,
            seed=int(config.get("seed", 42)),
            fold=fold,
            n_folds=int(config.get("cv_folds", 5)),
        )
    if dataset_name == "drive":
        cfg = config["dataset"]["drive"]
        return DRIVEDataset(
            root_dir=cfg["root"],
            split="test" if split == "val" else split,
            transform=transform,
            use_green_channel=bool(cfg.get("use_green_channel", True)),
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_val_transform(dataset_name: str, img_size: int):
    if dataset_name == "isic":
        return get_val_transforms(img_size)
    if dataset_name == "drive":
        return get_drive_val_transforms(img_size)
    raise ValueError(dataset_name)


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Return per-sample metric lists in a dict."""
    model.eval()
    all_metrics: Dict[str, List[float]] = {
        "dice": [], "iou": [], "sensitivity": [], "specificity": [], "hd95": []
    }
    for images, masks in tqdm(loader, desc="evaluate"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        preds = model(images)
        for i in range(images.size(0)):
            m = compute_all_metrics(preds[i].unsqueeze(0), masks[i].unsqueeze(0))
            for k, v in m.items():
                all_metrics[k].append(v)
    return all_metrics


def aggregate(all_metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    results = {}
    for k, values in all_metrics.items():
        if k == "hd95":
            finite = [v for v in values if np.isfinite(v)]
            mean = float(np.mean(finite)) if finite else float("inf")
            std = float(np.std(finite)) if finite else 0.0
            results[k] = {
                "mean": mean, "std": std,
                "n_valid": len(finite), "n_total": len(values),
            }
        else:
            arr = np.array(values, dtype=np.float64)
            results[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                          "n_valid": len(arr), "n_total": len(arr)}
    return results


# ---------------------------------------------------------------------------
def collect_qualitative_batch(loader: DataLoader, n: int):
    images, masks = [], []
    for imgs, msks in loader:
        images.append(imgs)
        masks.append(msks if msks.dim() == 4 else msks.unsqueeze(1))
        if sum(b.shape[0] for b in images) >= n:
            break
    images = torch.cat(images, dim=0)[:n]
    masks = torch.cat(masks, dim=0)[:n]
    return images, masks


@torch.no_grad()
def save_attention_visuals(model, sample_images, device, save_dir):
    """For AGHCT only — store per-image DAGM gate heat-maps."""
    if not hasattr(model, "get_dagm_gates"):
        return []
    os.makedirs(save_dir, exist_ok=True)
    out_paths = []
    model.eval()
    for i in range(sample_images.size(0)):
        x = sample_images[i:i + 1].to(device)
        _ = model(x)
        gates = model.get_dagm_gates()
        path = os.path.join(save_dir, f"attention_{i:03d}.png")
        save_attention_heatmaps(sample_images[i], gates, path)
        out_paths.append(path)
    return out_paths


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["aghct", "unet", "transunet"], required=True)
    parser.add_argument("--dataset", choices=["isic", "drive"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--no-qualitative", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    results_dir = args.results_dir or config["logging"].get("results_dir", "/kaggle/working/results")
    os.makedirs(results_dir, exist_ok=True)

    # -------------------- Data --------------------
    img_size = int(config["dataset"][args.dataset]["img_size"])
    val_tf = build_val_transform(args.dataset, img_size)
    val_ds = build_dataset(args.dataset, config, args.split, args.fold, val_tf)
    batch_size = int(config["dataset"][args.dataset]["batch_size"])
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["kaggle"]["num_workers"]),
        pin_memory=True,
    )

    # -------------------- Model --------------------
    model = build_model(args.model, config).to(device)
    # Our checkpoints embed a small config dict (incl. numpy scalars); the
    # weights_only=True default introduced in PyTorch 2.6 can't unpickle that.
    # We trust our own checkpoints, so opt out.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    n_params = count_parameters(model)
    print(f"[evaluate] {args.model}: {n_params:.2f} M params | checkpoint = {args.checkpoint}")

    # -------------------- Metrics --------------------
    all_metrics = evaluate_model(model, val_loader, device)
    summary = aggregate(all_metrics)

    print("\n" + "=" * 60)
    print(f"Results — model={args.model} dataset={args.dataset} fold={args.fold} split={args.split}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<12}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n_valid']}/{v['n_total']})")
    print("=" * 60)

    run_name = (
        f"{args.model}_{args.dataset}_fold{args.fold}_{args.split}"
    )
    metrics_path = os.path.join(results_dir, f"metrics_{run_name}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset,
                "fold": args.fold,
                "split": args.split,
                "checkpoint": args.checkpoint,
                "n_params_M": n_params,
                "metrics": summary,
                "per_sample": all_metrics,
            },
            f,
            indent=2,
        )
    print(f"[evaluate] metrics → {metrics_path}")

    # -------------------- Qualitative --------------------
    if not args.no_qualitative and args.n_samples > 0:
        qual_dir = os.path.join(results_dir, "qualitative", run_name)
        sample_imgs, sample_masks = collect_qualitative_batch(val_loader, args.n_samples)
        model.eval()
        with torch.no_grad():
            sample_preds = model(sample_imgs.to(device)).cpu()
        save_qualitative_grid(
            sample_imgs, sample_masks, sample_preds,
            save_dir=qual_dir, prefix="sample", n=args.n_samples,
        )
        print(f"[evaluate] qualitative → {qual_dir}")
    else:
        sample_imgs = None

    # -------------------- Attention --------------------
    if (
        not args.no_attention
        and args.model == "aghct"
        and sample_imgs is not None
    ):
        att_dir = os.path.join(results_dir, "attention", run_name)
        save_attention_visuals(model, sample_imgs, device, att_dir)
        print(f"[evaluate] attention → {att_dir}")


if __name__ == "__main__":
    main()
