"""TransUNet baseline (Chen et al., 2021), implemented with ``timm``'s ViT.

ViT operates on 224×224 inputs. We resize on-the-fly, run ViT, reshape the
196 tokens back into a 14×14 grid and upsample 4× with a U-Net-style decoder
to recover the original resolution.

This implementation aims to be a *fair, competitive baseline* rather than a
1-to-1 reproduction of the original paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("TransUNet baseline requires `timm` (pip install timm)") from e


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _conv_block(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class TransUNet(nn.Module):
    """ViT-B/16 (ImageNet pre-trained) + U-Net decoder.

    Args:
        in_channels: 3 for RGB.
        num_classes: 1 for binary segmentation.
        img_size: ViT input size (must match a pretrained model, 224 by default).
        pretrained: download/use ImageNet pre-trained ViT weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 224,
        pretrained: bool = True,
    ):
        super().__init__()
        self.img_size = img_size

        # Load ViT-B/16 backbone — features_only to expose intermediate maps
        # is awkward for ViTs, so we use the standard model and extract tokens.
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,           # remove classifier head
            global_pool="",         # keep tokens
        )
        # ViT-B/16 → 768-dim tokens, 14×14 grid (for 224×224)
        self.embed_dim = self.vit.embed_dim  # 768
        self.grid = img_size // 16            # 14

        # Decoder: 14×14 → 28 → 56 → 112 → 224
        self.dec1 = _UpBlock(self.embed_dim, 256)   # 14 → 28
        self.dec2 = _UpBlock(256, 128)              # 28 → 56
        self.dec3 = _UpBlock(128, 64)               # 56 → 112
        self.dec4 = _UpBlock(64, 32)                # 112 → 224

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    def _vit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens reshaped to ``(B, embed_dim, grid, grid)``."""
        # forward_features outputs the full token sequence including CLS
        tokens = self.vit.forward_features(x)  # (B, 1+N, D) or (B, N, D)
        if tokens.dim() == 3 and tokens.size(1) == self.grid * self.grid + 1:
            tokens = tokens[:, 1:, :]          # drop CLS
        # (B, N, D) -> (B, D, gH, gW)
        B, N, D = tokens.shape
        gH = gW = int(N**0.5)
        tokens = tokens.transpose(1, 2).reshape(B, D, gH, gW)
        return tokens

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[-2:]

        if in_size != (self.img_size, self.img_size):
            x_in = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        else:
            x_in = x

        feat = self._vit_tokens(x_in)  # (B, 768, 14, 14)

        x = self.dec1(feat)            # 28x28
        x = self.dec2(x)               # 56x56
        x = self.dec3(x)               # 112x112
        x = self.dec4(x)               # 224x224
        logits = self.classifier(x)

        if logits.shape[-2:] != in_size:
            logits = F.interpolate(
                logits, size=in_size, mode="bilinear", align_corners=False
            )
        return logits
