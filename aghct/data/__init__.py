# Lazy imports — datasets need PIL/sklearn (cheap) but the augmentation module
# requires albumentations. Importing this package therefore never *forces*
# albumentations to be installed, e.g. when only the model code is needed.
from .isic_dataset import ISICDataset, ISICUnlabeledDataset
from .drive_dataset import DRIVEDataset, DRIVEUnlabeledDataset

__all__ = [
    "ISICDataset",
    "ISICUnlabeledDataset",
    "DRIVEDataset",
    "DRIVEUnlabeledDataset",
    "get_train_transforms",
    "get_val_transforms",
    "get_drive_train_transforms",
    "get_drive_val_transforms",
]


def __getattr__(name):
    if name in {
        "get_train_transforms",
        "get_val_transforms",
        "get_drive_train_transforms",
        "get_drive_val_transforms",
    }:
        from . import augmentations as _aug  # requires albumentations

        return getattr(_aug, name)
    raise AttributeError(f"module 'aghct.data' has no attribute {name!r}")
