"""Albumentations transforms for ISIC and DRIVE.

Albumentations otomatik olarak mask'i da transform eder; sadece
``transform(image=img, mask=mask)`` çağırmak yeterlidir.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# ISIC (RGB skin lesion, 256×256)
# ---------------------------------------------------------------------------
def get_train_transforms(img_size: int = 256) -> A.Compose:
    """Training augmentations for ISIC 2018."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """Deterministic transforms for ISIC validation / test."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# DRIVE (Retinal vessels, 512×512)
# ---------------------------------------------------------------------------
def get_drive_train_transforms(img_size: int = 512) -> A.Compose:
    """Training augmentations for DRIVE retinal vessel segmentation."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_drive_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
