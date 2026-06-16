"""
Attention U-Net (Oktay et al., 2018)
Based on: https://arxiv.org/abs/1804.03999

Adds attention gates on skip connections to standard U-Net.
Used as second baseline for AGHCT comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv-BN-ReLU x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Additive attention gate from Oktay et al. 2018.

    Args:
        F_g: channels of gating signal (from coarser scale)
        F_l: channels of skip-connection feature (from encoder)
        F_int: intermediate channels (typically F_l // 2)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (B, F_g, Hg, Wg) — coarser
        # x: skip feature  (B, F_l, Hx, Wx) — finer

        # Resize gating to spatial size of x
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # gated skip


class UpBlock(nn.Module):
    """Upsample + AttentionGate + DoubleConv"""
    def __init__(self, in_ch_up, skip_ch, out_ch):
        super().__init__()
        # bilinear upsample (no learnable up-conv to keep params low)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attn = AttentionGate(F_g=in_ch_up, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = DoubleConv(in_ch_up + skip_ch, out_ch)

    def forward(self, x, skip):
        x_up = self.up(x)
        skip_gated = self.attn(g=x_up, x=skip)
        out = torch.cat([x_up, skip_gated], dim=1)
        return self.conv(out)


class AttentionUNet(nn.Module):
    """
    Attention U-Net baseline.
    Base channels = 64 → matches our U-Net baseline for fair comparison.
    """
    def __init__(self, in_channels=3, num_classes=1, base_channels=64):
        super().__init__()
        C = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, C)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(C, C * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(C * 2, C * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(C * 4, C * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(C * 8, C * 16)

        # Decoder with attention gates
        self.up4 = UpBlock(C * 16, C * 8, C * 8)
        self.up3 = UpBlock(C * 8, C * 4, C * 4)
        self.up2 = UpBlock(C * 4, C * 2, C * 2)
        self.up1 = UpBlock(C * 2, C, C)

        # Output
        self.out_conv = nn.Conv2d(C, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out_conv(d1)  # logits (B, 1, H, W)


if __name__ == "__main__":
    model = AttentionUNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"AttentionUNet: input {x.shape} -> output {y.shape}")
    print(f"Parameters: {n_params:.2f}M")
