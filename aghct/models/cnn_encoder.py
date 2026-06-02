"""Lightweight CNN encoder using Depthwise-Separable convolutions.

A 4-stage hierarchy whose output channels follow the standard U-Net doubling
pattern ``[C, 2C, 4C, 8C]`` so it composes naturally with the DAGM/decoder.

Shapes (input ``(B, 3, H, W)``)::

    f1: (B, C,   H/2,  W/2 )
    f2: (B, 2C,  H/4,  W/4 )
    f3: (B, 4C,  H/8,  W/8 )
    f4: (B, 8C,  H/16, W/16)

``C = base_channels`` (default 64).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 + Pointwise 1x1 + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


# ---------------------------------------------------------------------------
class _EncoderStage(nn.Module):
    """One encoder stage: 2x DSConv (no stride) followed by 2x2 max-pool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.block(x))


# ---------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    """4-stage depthwise-separable encoder."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        # Stem: lift input to ``c`` channels (no spatial downsample)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        # 4 down-sampling stages, doubling channels each time
        self.stage1 = _EncoderStage(c, c)        # → C  @ H/2
        self.stage2 = _EncoderStage(c, c * 2)    # → 2C @ H/4
        self.stage3 = _EncoderStage(c * 2, c * 4)  # → 4C @ H/8
        self.stage4 = _EncoderStage(c * 4, c * 8)  # → 8C @ H/16

        self.out_channels: List[int] = [c, c * 2, c * 4, c * 8]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]
