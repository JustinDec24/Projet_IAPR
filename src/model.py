"""CNN custom (style ResNet-18) pour classifier des crops de cartes UNO en 54 classes.

Architecture : stem 3×3, 4 stages de 2 BasicBlocks (skip connections), GAP,
classifier linéaire. ~11.2M params — sous le plafond de 12M imposé par le
règlement, entièrement entraîné from scratch sur la dataset synthétique.

Le saut résiduel facilite l'optimisation et permet d'aller plus profond sans
gradient vanishing — utile pour distinguer des classes visuellement proches
(ex : y_8 vs y_9, b_skip vs b_reverse).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES


class BasicBlock(nn.Module):
    """Bloc résiduel à 2 convolutions 3×3, optionnel downsampling sur le shortcut."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


def _make_stage(in_c: int, out_c: int, n_blocks: int, downsample: bool) -> nn.Sequential:
    blocks = [BasicBlock(in_c, out_c, stride=2 if downsample else 1)]
    for _ in range(n_blocks - 1):
        blocks.append(BasicBlock(out_c, out_c, stride=1))
    return nn.Sequential(*blocks)


class UnoCNN(nn.Module):
    """ResNet-18-like classifier dont les channels sont configurables.

    Default (channels=[64, 128, 256, 512]) -> ~11.2M params (le "teacher" pour
    la distillation).
    Compact (channels=[48, 96, 192, 384]) -> ~6.2M params (le "student", pour
    libérer du budget côté détecteur).
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 channels: tuple[int, int, int, int] = (64, 128, 256, 512)) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels

        # Stem : on garde la résolution spatiale pleine au début (convs 3×3, no
        # stride agressif) car les chiffres/symboles UNO sont des features
        # localisées dont on ne veut pas perdre la définition trop tôt.
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(c1, c1, n_blocks=2, downsample=False)
        self.stage2 = _make_stage(c1, c2, n_blocks=2, downsample=True)
        self.stage3 = _make_stage(c2, c3, n_blocks=2, downsample=True)
        self.stage4 = _make_stage(c3, c4, n_blocks=2, downsample=True)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
