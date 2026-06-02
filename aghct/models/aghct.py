"""AGHCT — Attention-Guided Hybrid CNN-Transformer.

Pipeline::

    Input (B,3,H,W) → CNNEncoder → [f1,f2,f3,f4]
                  ↘ DAGM_i applied per-scale → [g1,g2,g3,g4]
                  ↘ FPN-style Decoder → (B,1,H,W) logits
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import CNNEncoder
from .dagm import DAGM
from .decoder import Decoder


class AGHCT(nn.Module):
    """Attention-Guided Hybrid CNN-Transformer.

    Args:
        config: dict-like, the ``model`` section of ``config.yaml``.
        in_channels: input channels (3 for RGB).
        num_classes: output channels (1 for binary).
        use_dagm_in_decoder: also place DAGM inside decoder stages.
    """

    def __init__(
        self,
        config: dict,
        in_channels: int = 3,
        num_classes: int = 1,
        use_dagm_in_decoder: bool = True,
    ):
        super().__init__()
        base_ch = int(config["base_channels"])
        dagm_cfg = config["dagm"]
        dropout_p = float(config.get("dropout", 0.0))

        self.encoder = CNNEncoder(in_channels=in_channels, base_channels=base_ch)
        enc_channels: List[int] = self.encoder.out_channels  # [C, 2C, 4C, 8C]

        # One DAGM per encoder scale
        self.dagm_modules = nn.ModuleList(
            [
                DAGM(
                    channels=ch,
                    window_size=int(dagm_cfg["window_size"]),
                    num_heads=int(dagm_cfg["num_heads"]),
                    num_layers=int(dagm_cfg["num_transformer_layers"]),
                    se_reduction=int(dagm_cfg["se_reduction_ratio"]),
                )
                for ch in enc_channels
            ]
        )

        self.decoder = Decoder(
            encoder_channels=enc_channels,
            base_channels=base_ch,
            dagm_config=dagm_cfg,
            num_classes=num_classes,
            use_dagm_in_decoder=use_dagm_in_decoder,
        )

        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        gated = [dagm(f) for f, dagm in zip(features, self.dagm_modules)]
        gated[-1] = self.dropout(gated[-1])  # drop only at the deepest map
        logits = self.decoder(gated)
        # Ensure output matches the input resolution
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits

    # ------------------------------------------------------------------
    def load_pretrained_encoder(
        self, checkpoint_path: str, strict: bool = False, verbose: bool = True
    ) -> None:
        """Load SimCLR-pre-trained encoder weights into ``self.encoder``."""
        # Pretrained encoder weights are pure tensors, but old checkpoints
        # may include extra metadata — opt out of weights_only for safety.
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Allow checkpoints saved either as raw state_dict or {"encoder": ...}
        if isinstance(state_dict, dict) and "encoder" in state_dict:
            state_dict = state_dict["encoder"]
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=strict)
        if verbose:
            print(
                f"[AGHCT] Loaded pre-trained encoder from {checkpoint_path}\n"
                f"        missing={len(missing)}, unexpected={len(unexpected)}"
            )

    # ------------------------------------------------------------------
    def get_dagm_gates(self) -> List[Optional[torch.Tensor]]:
        """Return the per-scale ``α`` gate maps cached during the last eval pass."""
        return [m.last_gate for m in self.dagm_modules]
