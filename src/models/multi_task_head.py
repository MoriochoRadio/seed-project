"""Multi-task heads:
    1) WHO motility classification (3-class)
    2) Hyperactivation detection    (binary)
    3) Kinematic regression         (continuous CASA-style scores)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 3,
        kinematic_dim: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(in_dim, num_classes)
        self.hyper_head = nn.Linear(in_dim, 2)
        self.reg_head = nn.Linear(in_dim, kinematic_dim)

    def forward(self, h: torch.Tensor) -> dict:
        z = self.shared(h)
        return {
            "logits": self.cls_head(z),
            "hyper_logits": self.hyper_head(z),
            "kinematic_pred": self.reg_head(z),
        }
