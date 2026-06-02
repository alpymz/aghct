"""ISIC 2018 Skin Lesion Segmentation Dataset (Kaggle-friendly).

The expected Kaggle layout is::

    /kaggle/input/isic-2018/
        ISIC2018_Task1-2_Training_Input/      (.jpg)
        ISIC2018_Task1_Training_GroundTruth/  (_segmentation.png)
        ISIC2018_Task1-2_Test_Input/          (.jpg)

Because Kaggle datasets sometimes have slightly different folder names, the
class discovers the actual sub-directories with ``os.listdir`` and prints them
once so the user can verify.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Folder discovery helpers
# ---------------------------------------------------------------------------
def _find_subdir(root: str, must_contain: List[str]) -> Optional[str]:
    """Return first sub-directory of ``root`` whose name contains all keywords."""
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        lower = name.lower()
        if all(k.lower() in lower for k in must_contain):
            return full
    return None


def _list_image_files(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(exts) and not f.startswith(".")
    )


def _match_mask(images_dir: str, masks_dir: str, image_name: str) -> Optional[str]:
    """ISIC mask name = ``<image_id>_segmentation.png``."""
    stem = os.path.splitext(image_name)[0]
    candidates = [
        f"{stem}_segmentation.png",
        f"{stem}_Segmentation.png",
        f"{stem}.png",
    ]
    for c in candidates:
        path = os.path.join(masks_dir, c)
        if os.path.exists(path):
            return path
    return None


# ===========================================================================
# Labeled dataset (segmentation)
# ===========================================================================
class ISICDataset(Dataset):
    """ISIC 2018 segmentation dataset with k-fold + data-fraction support."""

    _printed_paths = False  # class-level flag — print structure once per run

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        data_fraction: float = 1.0,
        seed: int = 42,
        fold: int = 0,
        n_folds: int = 5,
    ) -> None:
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        assert 0 < data_fraction <= 1.0, "data_fraction must be in (0, 1]"
        assert 0 <= fold < n_folds, f"fold must be in [0, {n_folds})"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data_fraction = data_fraction
        self.seed = seed
        self.fold = fold
        self.n_folds = n_folds

        images_dir = _find_subdir(root_dir, ["training", "input"])
        masks_dir = _find_subdir(root_dir, ["training", "groundtruth"])

        if images_dir is None or masks_dir is None:
            # Fallback: try generic "input"/"mask" naming
            images_dir = images_dir or _find_subdir(root_dir, ["input"])
            masks_dir = masks_dir or _find_subdir(root_dir, ["groundtruth"]) \
                or _find_subdir(root_dir, ["mask"])

        if images_dir is None or masks_dir is None:
            raise FileNotFoundError(
                f"Could not locate ISIC train images/masks under {root_dir}. "
                f"Listed entries: {os.listdir(root_dir) if os.path.isdir(root_dir) else 'N/A'}"
            )

        if not ISICDataset._printed_paths:
            print(f"[ISICDataset] images dir : {images_dir}")
            print(f"[ISICDataset] masks  dir : {masks_dir}")
            ISICDataset._printed_paths = True

        self.images_dir = images_dir
        self.masks_dir = masks_dir

        all_images = _list_image_files(images_dir)
        # Drop images without a matching mask
        self.samples: List[Tuple[str, str]] = []
        for name in all_images:
            mask_path = _match_mask(images_dir, masks_dir, name)
            if mask_path is not None:
                self.samples.append((os.path.join(images_dir, name), mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No (image, mask) pairs found in {images_dir} ↔ {masks_dir}"
            )

        # ------------------------------------------------------------------
        # Deterministic shuffle + k-fold split + data_fraction
        # ------------------------------------------------------------------
        rng = np.random.RandomState(seed)
        order = np.arange(len(self.samples))
        rng.shuffle(order)
        ordered_samples = [self.samples[i] for i in order]

        kf = KFold(n_splits=n_folds, shuffle=False)
        splits = list(kf.split(ordered_samples))
        train_idx, val_idx = splits[fold]

        if split == "train":
            chosen_idx = train_idx
            if data_fraction < 1.0:
                n_keep = max(1, int(len(chosen_idx) * data_fraction))
                # Deterministic subset, seeded by (seed, fold, fraction)
                sub_rng = np.random.RandomState(
                    seed + fold * 1000 + int(data_fraction * 10000)
                )
                chosen_idx = sub_rng.permutation(chosen_idx)[:n_keep]
        else:
            chosen_idx = val_idx  # validation always full

        self.indices = list(chosen_idx)
        self.samples = [ordered_samples[i] for i in self.indices]

        print(
            f"[ISICDataset] split={split} fold={fold}/{n_folds} "
            f"fraction={data_fraction} -> {len(self.samples)} samples"
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 127).astype(np.float32)  # binarize → {0,1}

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"].float()
        return image, mask


# ===========================================================================
# Unlabeled dataset for SimCLR pre-training
# ===========================================================================
class ISICUnlabeledDataset(Dataset):
    """All ISIC images (train + test) without masks, for SimCLR.

    Returns two augmented views ``(view_i, view_j)`` per sample.
    """

    _printed_paths = False

    def __init__(self, root_dir: str, augmentation, include_test: bool = True):
        self.root_dir = root_dir
        self.augmentation = augmentation

        train_dir = _find_subdir(root_dir, ["training", "input"])
        test_dir = _find_subdir(root_dir, ["test", "input"]) if include_test else None

        if train_dir is None:
            train_dir = _find_subdir(root_dir, ["input"])
        if train_dir is None:
            raise FileNotFoundError(
                f"Could not locate ISIC input images under {root_dir}"
            )

        if not ISICUnlabeledDataset._printed_paths:
            print(f"[ISICUnlabeledDataset] train images: {train_dir}")
            if test_dir:
                print(f"[ISICUnlabeledDataset] test  images: {test_dir}")
            ISICUnlabeledDataset._printed_paths = True

        self.image_paths: List[str] = [
            os.path.join(train_dir, n) for n in _list_image_files(train_dir)
        ]
        if test_dir is not None and os.path.isdir(test_dir):
            self.image_paths += [
                os.path.join(test_dir, n) for n in _list_image_files(test_dir)
            ]

        if not self.image_paths:
            raise RuntimeError(f"No images found for SimCLR under {root_dir}")
        print(f"[ISICUnlabeledDataset] total images = {len(self.image_paths)}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        view_i, view_j = self.augmentation(image)
        return view_i, view_j
