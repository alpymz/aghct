"""DRIVE retinal vessel segmentation dataset (Kaggle-friendly).

Typical layout (varies across Kaggle uploads)::

    /kaggle/input/drive-digital-retinal-images/
        DRIVE/
            training/
                images/   *.tif
                1st_manual/  *.gif   (vessel masks)
                mask/        *.gif   (FOV masks)
            test/
                images/   *.tif
                1st_manual/  *.gif
                2nd_manual/  *.gif
                mask/        *.gif
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _walk_find_dir(root: str, must_contain: List[str]) -> Optional[str]:
    """Recursively search for a directory whose path contains all keywords."""
    if not os.path.isdir(root):
        return None
    keywords = [k.lower() for k in must_contain]
    for dirpath, dirnames, _ in os.walk(root):
        lower = dirpath.lower().replace("\\", "/")
        if all(k in lower for k in keywords):
            return dirpath
    return None


def _list_image_files(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(exts) and not f.startswith(".")
    )


def _drive_id(filename: str) -> str:
    """Extract the numeric DRIVE id (e.g. '21_training.tif' -> '21')."""
    stem = os.path.splitext(filename)[0]
    return stem.split("_")[0]


def _match_by_id(folder: str, image_id: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    for f in _list_image_files(folder):
        if _drive_id(f) == image_id:
            return os.path.join(folder, f)
    return None


# ===========================================================================
# Labeled DRIVE dataset
# ===========================================================================
class DRIVEDataset(Dataset):
    """DRIVE retinal vessel segmentation.

    Args:
        root_dir: Kaggle DRIVE root (auto-detects ``training``/``test``).
        split:    ``"train"`` or ``"test"``.
        transform: Albumentations transform.
        use_green_channel: If True, use only the green channel (replicated to 3 channels).
        fold: ignored except for low-data ablation determinism.
    """

    _printed_paths = False

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        use_green_channel: bool = True,
        data_fraction: float = 1.0,
        seed: int = 42,
        fold: int = 0,
        n_folds: int = 5,
    ) -> None:
        assert split in ("train", "test", "val"), "split must be train/val/test"
        # DRIVE has no official train/val split → treat 'val' the same as 'test'
        kaggle_split = "training" if split == "train" else "test"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_green_channel = use_green_channel

        images_dir = _walk_find_dir(root_dir, [kaggle_split, "images"])
        masks_dir = _walk_find_dir(root_dir, [kaggle_split, "1st_manual"])
        if masks_dir is None:
            masks_dir = _walk_find_dir(root_dir, [kaggle_split, "manual"])
        fov_dir = _walk_find_dir(root_dir, [kaggle_split, "mask"])

        if images_dir is None or masks_dir is None:
            raise FileNotFoundError(
                f"DRIVE: could not find images/masks for split='{kaggle_split}' under {root_dir}"
            )

        if not DRIVEDataset._printed_paths:
            print(f"[DRIVEDataset] images: {images_dir}")
            print(f"[DRIVEDataset] masks : {masks_dir}")
            print(f"[DRIVEDataset] fov   : {fov_dir}")
            DRIVEDataset._printed_paths = True

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.fov_dir = fov_dir

        samples: List[Tuple[str, str, Optional[str]]] = []
        for fname in _list_image_files(images_dir):
            img_id = _drive_id(fname)
            mask_path = _match_by_id(masks_dir, img_id)
            fov_path = _match_by_id(fov_dir, img_id) if fov_dir else None
            if mask_path is not None:
                samples.append((os.path.join(images_dir, fname), mask_path, fov_path))

        if not samples:
            raise RuntimeError(f"No DRIVE samples found in {images_dir}")

        # Deterministic ordering + optional sub-sampling for low-data
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(samples))
        samples = [samples[i] for i in order]
        if split == "train" and data_fraction < 1.0:
            n_keep = max(1, int(len(samples) * data_fraction))
            sub_rng = np.random.RandomState(
                seed + fold * 1000 + int(data_fraction * 10000)
            )
            keep_idx = sub_rng.permutation(len(samples))[:n_keep]
            samples = [samples[i] for i in sorted(keep_idx)]

        self.samples = samples
        print(
            f"[DRIVEDataset] split={split} fold={fold} fraction={data_fraction} "
            f"-> {len(self.samples)} samples"
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, _fov_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        if self.use_green_channel:
            green = image[:, :, 1]
            image = np.stack([green, green, green], axis=-1)

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"].float()
        return image, mask


# ===========================================================================
# Unlabeled DRIVE / multi-source for SimCLR (DRIVE only by default)
# ===========================================================================
class DRIVEUnlabeledDataset(Dataset):
    """Unlabeled retinal images for SimCLR pre-training.

    Crops dense ``patch_size`` patches with stride ``stride`` from every image.
    """

    def __init__(
        self,
        root_dir: str,
        augmentation,
        patch_size: int = 128,
        stride: int = 64,
        include_test: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.stride = stride

        img_dirs = []
        train_dir = _walk_find_dir(root_dir, ["training", "images"])
        if train_dir:
            img_dirs.append(train_dir)
        if include_test:
            test_dir = _walk_find_dir(root_dir, ["test", "images"])
            if test_dir:
                img_dirs.append(test_dir)

        if not img_dirs:
            raise FileNotFoundError(f"No DRIVE image directories under {root_dir}")

        self.image_paths: List[str] = []
        for d in img_dirs:
            for f in _list_image_files(d):
                self.image_paths.append(os.path.join(d, f))

        # Pre-compute (path, y, x) patch index for deterministic length
        self.patches: List[Tuple[str, int, int]] = []
        for path in self.image_paths:
            with Image.open(path) as im:
                W, H = im.size
            for y in range(0, max(1, H - patch_size + 1), stride):
                for x in range(0, max(1, W - patch_size + 1), stride):
                    self.patches.append((path, y, x))
        if not self.patches:
            raise RuntimeError("No DRIVE patches generated")
        print(
            f"[DRIVEUnlabeledDataset] images={len(self.image_paths)} "
            f"patches={len(self.patches)} (patch={patch_size}, stride={stride})"
        )

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        path, y, x = self.patches[idx]
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        H, W = image.shape[:2]
        y2 = min(y + self.patch_size, H)
        x2 = min(x + self.patch_size, W)
        patch = image[y:y2, x:x2]
        view_i, view_j = self.augmentation(patch)
        return view_i, view_j
