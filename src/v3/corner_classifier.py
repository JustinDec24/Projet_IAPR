"""Phase 4 V3 — Classifieur de coins (CNN ultra-léger, 54 classes).

Input  : crop 80×80 RGB (le digit-coin + marge couleur)
Archi  : 4 blocs Conv+BN+ReLU [32, 64, 96, 128] + GAP + Linear(128, 54)
Cible  : 1-1.5M params (vérifié par count_parameters)

Pas de flip dans les augmentations (6 vs 9 distincts !) — géré côté dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import NUM_CLASSES

INPUT_SIZE = 80


def _block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CornerClassifier(nn.Module):
    """80×80 → 54 logits. ~1.2M params avec [32,64,96,128]."""

    def __init__(self, num_classes: int = NUM_CLASSES,
                 channels: tuple[int, int, int, int] = (32, 64, 96, 128)
                 ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            _block(3, c1),    # 80 -> 40
            _block(c1, c2),   # 40 -> 20
            _block(c2, c3),   # 20 -> 10
            _block(c3, c4),   # 10 -> 5
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
