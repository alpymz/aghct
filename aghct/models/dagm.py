"""Dual-Attention Gate Module (DAGM).

Two parallel branches refine a feature map ``F ∈ R^{B×C×H×W}``:

1. **SE branch** — channel attention via global pooling + bottleneck MLP::

       z_c   = (1/HW) Σ F_c(i, j)
       s     = σ(W_2 δ(W_1 z))
       F_SE  = s ⊙ F

2. **Local-Window Transformer (LWT) branch** — spatial attention applied
   inside non-overlapping ``w × w`` windows, with a learnable relative
   positional bias inside each window::

       Attn = softmax(Q K^T / √C + B) V

3. **Gated fusion** — a learned per-pixel gate adaptively blends them::

       α     = σ(Conv_{1×1}([F_SE ; F_LWT]))
       F̂    = α ⊙ F_SE + (1 - α) ⊙ F_LWT

The gate tensor ``α`` is stored on ``self.last_gate`` whenever the module is
in eval mode, so it can be visualised by ``utils.visualization``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ===========================================================================
# 1. Squeeze-and-Excitation branch
# ===========================================================================
class SEBranch(nn.Module):
    """Standard Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = self.pool(x).view(b, c)
        s = self.fc(z).view(b, c, 1, 1)
        return s * x


# ===========================================================================
# 2. Window partition / unpartition (with padding to handle odd sizes)
# ===========================================================================
def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Split ``(B, C, H, W)`` into windows of shape ``(B*nW, C, w, w)``.

    If H or W is not divisible by ``window_size``, the tensor is right/bottom
    padded with zeros. Returns the partitioned tensor and the padding info
    (pad_h, pad_w, H_padded, W_padded) needed by :func:`window_unpartition`.
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad last dim first
    Hp, Wp = H + pad_h, W + pad_w

    windows = rearrange(
        x, "b c (h wh) (w ww) -> (b h w) c wh ww", wh=window_size, ww=window_size
    )
    return windows, (pad_h, pad_w, Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_info: Tuple[int, int, int, int],
    H: int,
    W: int,
    batch_size: int,
) -> torch.Tensor:
    """Inverse of :func:`window_partition` (also strips the padding)."""
    pad_h, pad_w, Hp, Wp = pad_info
    nH = Hp // window_size
    nW = Wp // window_size
    x = rearrange(
        windows,
        "(b h w) c wh ww -> b c (h wh) (w ww)",
        b=batch_size,
        h=nH,
        w=nW,
    )
    if pad_h or pad_w:
        x = x[:, :, :H, :W]
    return x


# ===========================================================================
# 3. One local-window transformer block
# ===========================================================================
class LocalWindowTransformerBlock(nn.Module):
    """Pre-norm Transformer block operating on flattened window tokens.

    ``forward`` expects ``(B*nW, w*w, C)``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Relative positional bias inside each window — one scalar per
        # (head, query-pos, key-pos) triple.
        n = window_size * window_size
        self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, n, n))
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*nW, N=w*w, C)
        h = self.norm1(x)

        # Build (B*nW, num_heads, N, N) attention bias by broadcasting.
        bias = self.relative_position_bias.unsqueeze(0)  # (1, H, N, N)
        bias = bias.expand(h.size(0), -1, -1, -1)
        attn_mask = bias.reshape(h.size(0) * self.num_heads, h.size(1), h.size(1))

        attn_out, _ = self.attn(
            h, h, h, need_weights=False, attn_mask=attn_mask
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ===========================================================================
# 4. LWT branch — stack of windowed transformer blocks
# ===========================================================================
class LWTBranch(nn.Module):
    """Apply ``num_layers`` window-local Transformer blocks then unwindow."""

    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        num_heads: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.window_size = window_size

        # Pick a head count that actually divides ``channels`` (Multi-head
        # attention requires ``dim % num_heads == 0``).
        if channels % num_heads != 0:
            for cand in (num_heads, 8, 4, 2, 1):
                if channels % cand == 0:
                    num_heads = cand
                    break
        self.num_heads = num_heads

        self.layers = nn.ModuleList(
            [
                LocalWindowTransformerBlock(
                    dim=channels, num_heads=num_heads, window_size=window_size
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.window_size

        windows, pad_info = window_partition(x, w)         # (B*nW, C, w, w)
        tokens = rearrange(windows, "n c h w -> n (h w) c")  # (B*nW, w*w, C)

        for blk in self.layers:
            tokens = blk(tokens)

        windows = rearrange(
            tokens, "n (h w) c -> n c h w", h=w, w=w
        )
        out = window_unpartition(windows, w, pad_info, H, W, B)
        return out


# ===========================================================================
# 5. DAGM — gated fusion of SE + LWT
# ===========================================================================
class DAGM(nn.Module):
    """Dual-Attention Gate Module: SE ⊕ LWT with a learned spatial gate."""

    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        num_heads: int = 8,
        num_layers: int = 4,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.se_branch = SEBranch(channels, se_reduction)
        self.lwt_branch = LWTBranch(
            channels, window_size=window_size, num_heads=num_heads, num_layers=num_layers
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # For visualisation: holds the most recent gate map (eval mode only).
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_se = self.se_branch(x)
        f_lwt = self.lwt_branch(x)
        alpha = self.gate_conv(torch.cat([f_se, f_lwt], dim=1))

        if not self.training:
            # Store a detached copy for downstream visualisation
            self.last_gate = alpha.detach()

        return alpha * f_se + (1.0 - alpha) * f_lwt
