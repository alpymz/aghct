__all__ = [
    "SimCLRAugmentation",
    "ProjectionHead",
    "NTXentLoss",
    "SimCLRTrainer",
]


# Lazy import — `simclr` pulls in albumentations and tqdm which may not be
# available in lightweight (e.g. CI / local Windows) environments.
def __getattr__(name):
    if name in __all__:
        from . import simclr as _s

        return getattr(_s, name)
    raise AttributeError(f"module 'aghct.pretrain' has no attribute {name!r}")
