"""FPN-style segmentation decoder used by AGHCT.

Receives the 4 gated encoder feature maps
``[g1, g2, g3, g4]`` with channels ``[C, 2C, 4C, 8C]`` and progressively
upsamples them via ``upsample → concat(skip) → conv → DAGM``.

Final 1×1 convolution emits raw logits ``(B, num_classes, H, W)``. Sigmoid is
**not** applied here — :class:`losses.BCEDiceLoss` handles that.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dagm import DAGM


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """One level of the decoder: upsample → concat skip → 2× ConvBN → DAGM."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_dagm: bool = True,
        dagm_config: Optional[dict] = None,
    ):
        super().__init__()
        self.conv1 = _ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = _ConvBNReLU(out_ch, out_ch)
        self.use_dagm = use_dagm
        if use_dagm:
            cfg = dagm_config or {}
            self.dagm = DAGM(
                channels=out_ch,
                window_size=cfg.get("window_size", 8),
                num_heads=cfg.get("num_heads", 8),
                num_layers=max(1, cfg.get("num_transformer_layers", 4) // 2),
                se_reduction=cfg.get("se_reduction_ratio", 16),
            )
        else:
            self.dagm = nn.Identity()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Align spatial dims (rounding tolerance for odd inputs)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        x = self.dagm(x)
        return x


class Decoder(nn.Module):
    """4-level FPN decoder.

    Args:
        encoder_channels: ``[C, 2C, 4C, 8C]`` from the encoder.
        base_channels:    Output channel of the very last conv (== ``C``).
        dagm_config:      DAGM kwargs (window_size, num_heads, ...).
        num_classes:      Output channels of the final 1×1 conv.
        use_dagm_in_decoder: Toggle DAGM blocks inside decoder stages.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        base_channels: int = 64,
        dagm_config: Optional[dict] = None,
        num_classes: int = 1,
        use_dagm_in_decoder: bool = True,
    ):
        super().__init__()
        assert len(encoder_channels) == 4, "expecting 4 encoder scales"
        c1, c2, c3, c4 = encoder_channels  # [C, 2C, 4C, 8C]

        # Top-down path: start from the deepest feature (no skip)
        self.up3 = DecoderBlock(
            in_ch=c4,
            skip_ch=c3,
            out_ch=c3,
            use_dagm=use_dagm_in_decoder,
            dagm_config=dagm_config,
        )
        self.up2 = DecoderBlock(
            in_ch=c3,
            skip_ch=c2,
            out_ch=c2,
            use_dagm=use_dagm_in_decoder,
            dagm_config=dagm_config,
        )
        self.up1 = DecoderBlock(
            in_ch=c2,
            skip_ch=c1,
            out_ch=c1,
            use_dagm=use_dagm_in_decoder,
            dagm_config=dagm_config,
        )
        # Final upsample back to input resolution (×2): no skip here.
        self.up0 = DecoderBlock(
            in_ch=c1,
            skip_ch=0,
            out_ch=base_channels,
            use_dagm=False,
            dagm_config=None,
        )

        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features  # increasing depth

        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.up0(x, None)

        logits = self.classifier(x)
        return logits
