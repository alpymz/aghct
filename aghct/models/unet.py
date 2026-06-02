"""Classic U-Net baseline (Ronneberger et al., 2015).

No pre-training. Same interface as AGHCT: ``forward(x) -> (B, C, H, W)`` logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = _double_conv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with ``base_channels`` = 64 (~31 M params)."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
    ):
        super().__init__()
        c = base_channels
        self.inc = _double_conv(in_channels, c)
        self.down1 = _Down(c, c * 2)
        self.down2 = _Down(c * 2, c * 4)
        self.down3 = _Down(c * 4, c * 8)
        self.down4 = _Down(c * 8, c * 16)

        self.up4 = _Up(c * 16, c * 8, c * 8)
        self.up3 = _Up(c * 8, c * 4, c * 4)
        self.up2 = _Up(c * 4, c * 2, c * 2)
        self.up1 = _Up(c * 2, c, c)
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.out(x)
