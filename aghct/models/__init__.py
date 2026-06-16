from .cnn_encoder import CNNEncoder, DepthwiseSeparableConv
from .dagm import DAGM, SEBranch, LWTBranch, LocalWindowTransformerBlock
from .decoder import Decoder, DecoderBlock
from .aghct import AGHCT
from .unet import UNet
from .attention_unet import AttentionUNet

__all__ = [
    "CNNEncoder",
    "DepthwiseSeparableConv",
    "DAGM",
    "SEBranch",
    "LWTBranch",
    "LocalWindowTransformerBlock",
    "Decoder",
    "DecoderBlock",
    "AGHCT",
    "UNet",
    "TransUNet",
    "AttentionUNet",
]


def __getattr__(name):
    # Lazy-import TransUNet so that `timm` is only required when this baseline
    # is actually used (e.g. on Kaggle where the user installs it explicitly).
    if name == "TransUNet":
        from .transunet import TransUNet as _TransUNet
        return _TransUNet
    raise AttributeError(f"module 'aghct.models' has no attribute {name!r}")
