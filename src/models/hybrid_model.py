"""Hybrid CNN (InceptionTime) + Transformer encoder for trajectory-aware
classification.

The model accepts:
    * trajectory  : (B, T, 2) — raw (x, y)
    * velocity    : (B, T, 2)
    * mask        : (B, T)    — 1 valid / 0 padded
    * confidence  : (B,)      — tracking confidence (proposed term)
    * casa        : (B, 9)    — VCL/VSL/VAP/LIN/STR/WOB/ALH/BCF/PAW
    * density     : (B, T)    — per-frame detection count (proposed term)

Set ``use_tracking_confidence=False`` for the *baseline* configuration
(Shahali-style: trajectory only, no tracking-aware terms).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .inception_time import _InceptionModule
from .multi_task_head import MultiTaskHead


class _PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class HybridTrajectoryModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 4,                # x, y, vx, vy
        cnn_filters: int = 32,
        cnn_kernels: Sequence[int] = (9, 19, 39),
        cnn_depth: int = 4,
        transformer_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        casa_dim: int = 9,
        use_tracking_confidence: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_tracking_confidence = use_tracking_confidence

        # ---- CNN branch ------------------------------------------------ #
        modules = []
        ch = in_channels
        for _ in range(cnn_depth):
            mod = _InceptionModule(ch, cnn_filters, cnn_kernels)
            modules.append(mod)
            ch = mod.out_channels
        self.cnn = nn.ModuleList(modules)
        self.cnn_proj = nn.Conv1d(ch, transformer_dim, 1)

        # ---- Transformer branch -------------------------------------- #
        self.pos_enc = _PositionalEncoding(transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ---- side branches ------------------------------------------- #
        self.casa_proj = nn.Sequential(
            nn.LayerNorm(casa_dim),
            nn.Linear(casa_dim, transformer_dim),
            nn.GELU(),
        )
        if use_tracking_confidence:
            self.tracking_proj = nn.Sequential(
                nn.Linear(2, transformer_dim),  # (confidence, mean_density)
                nn.GELU(),
            )
        # ---- head ----------------------------------------------------- #
        fused_dim = transformer_dim * (3 if use_tracking_confidence else 2)
        self.head = MultiTaskHead(fused_dim, num_classes=num_classes, kinematic_dim=casa_dim, dropout=dropout)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        trajectory: torch.Tensor,        # (B, T, 2)
        velocity: torch.Tensor,          # (B, T, 2)
        mask: torch.Tensor,              # (B, T)
        confidence: torch.Tensor,        # (B,)
        casa: torch.Tensor,              # (B, 9)
        density: torch.Tensor,           # (B, T)
    ) -> dict:
        B, T, _ = trajectory.shape
        # input channels: x, y, vx, vy → (B, 4, T)
        x = torch.cat([trajectory, velocity], dim=-1).transpose(1, 2)

        for block in self.cnn:
            x = block(x)
        x = self.cnn_proj(x)                       # (B, D, T)
        x = x.transpose(1, 2)                      # (B, T, D)
        x = self.pos_enc(x)

        # Transformer mask: True = pad
        key_padding_mask = mask < 0.5
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Masked global average pool
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        traj_feat = (x * m).sum(dim=1) / denom

        casa_feat = self.casa_proj(casa)
        feats = [traj_feat, casa_feat]

        if self.use_tracking_confidence:
            mean_density = (density * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            tcfg = torch.stack([confidence, mean_density], dim=-1)
            feats.append(self.tracking_proj(tcfg))

        h = torch.cat(feats, dim=-1)
        return self.head(h)
