"""Minimal InceptionTime baseline (Shahali et al. 2026 reference architecture).

Adapted from Fawaz et al. 2020 for 1-D time-series classification.
Used as the *baseline* for the experiments.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class _InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes: Sequence[int] = (9, 19, 39),
        bottleneck_channels: int = 32,
    ) -> None:
        super().__init__()
        bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
            if in_channels > 1
            else nn.Identity()
        )
        self.bottleneck = bottleneck
        ch = bottleneck_channels if in_channels > 1 else in_channels
        self.convs = nn.ModuleList(
            [nn.Conv1d(ch, n_filters, k, padding=k // 2, bias=False) for k in kernel_sizes]
        )
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, n_filters, 1, bias=False)
        out_ch = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        outs = [c(z) for c in self.convs]
        outs.append(self.pool_conv(self.maxpool(x)))
        return self.act(self.bn(torch.cat(outs, dim=1)))


class InceptionTime(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        n_filters: int = 32,
        kernel_sizes: Sequence[int] = (9, 19, 39),
        depth: int = 6,
    ) -> None:
        super().__init__()
        modules = []
        ch = in_channels
        for i in range(depth):
            mod = _InceptionModule(ch, n_filters, kernel_sizes)
            modules.append(mod)
            ch = mod.out_channels
        self.blocks = nn.ModuleList(modules)
        # Residual every 3 blocks
        self.residuals = nn.ModuleList(
            [nn.Conv1d(in_channels if i == 0 else self.blocks[i - 1].out_channels,
                        self.blocks[i + 2].out_channels, 1, bias=False)
             for i in range(0, depth, 3)]
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = ch
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % 3 == 0 and (i // 3) < len(self.residuals):
                x = x + self.residuals[i // 3](residual)
                residual = x
        feat = self.gap(x).squeeze(-1)
        return self.head(feat)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % 3 == 0 and (i // 3) < len(self.residuals):
                x = x + self.residuals[i // 3](residual)
                residual = x
        return self.gap(x).squeeze(-1)
