"""Phase 3 V3 — Détecteur de coins de cartes (point detection style CenterNet).

Détecte les coins-digit (pas de bbox rotée). Sortie :
  - heatmap : (B, 1, H/4, W/4) sigmoid — pic = position d'un coin
  - offsets : (B, 2, H/4, W/4) — raffinement sous-pixel (dx, dy ∈ [0,1])

Backbone léger : channels [32, 64, 128, 256], pas de FPN (les coins ont une
taille assez homogène d'une carte à l'autre dans ces images contrôlées).

Cible : 3-4M params (vérifié par count_parameters).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DOWNSAMPLE = 4  # heatmap à H/4, W/4


def _conv_bn(in_c: int, out_c: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _Block(nn.Module):
    """2 convs 3×3 + skip (résiduel léger)."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_c)
        self.short: nn.Module = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_c))
            if (stride != 1 or in_c != out_c) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = F.relu(self.b1(self.c1(x)), inplace=True)
        o = self.b2(self.c2(o))
        return F.relu(o + self.short(x), inplace=True)


class CornerDetector(nn.Module):
    """Entrée (B,3,H,W) RGB normalisé [0,1]. Sortie heatmap + offsets à H/4."""

    def __init__(self, channels: tuple[int, int, int, int] = (32, 64, 128, 256)
                 ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        # Stem ÷2
        self.stem = _conv_bn(3, c1, k=7, s=2)
        # Stage 1 : ÷2 -> ÷4 cumul
        self.stage1 = nn.Sequential(_Block(c1, c1), _Block(c1, c2, stride=2))
        # Stage 2 : reste à ÷4 (stride 1) — préserve la résolution pour les coins
        self.stage2 = nn.Sequential(_Block(c2, c2), _Block(c2, c3))
        # Stage 3 : ÷4 toujours, plus de capacité
        self.stage3 = nn.Sequential(_Block(c3, c3), _Block(c3, c4))
        # Tête
        self.head = nn.Sequential(_conv_bn(c4, c3), _conv_bn(c3, c3))
        self.hm = nn.Conv2d(c3, 1, 1)      # heatmap logits
        self.off = nn.Conv2d(c3, 2, 1)     # offsets
        nn.init.constant_(self.hm.bias, -4.6)  # prior bas (peu de pixels positifs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)        # ÷2
        x = self.stage1(x)      # ÷4
        x = self.stage2(x)      # ÷4
        x = self.stage3(x)      # ÷4
        x = self.head(x)
        return {
            "heatmap": self.hm(x),               # logits (B,1,H/4,W/4)
            "offset": torch.sigmoid(self.off(x)),  # (B,2,H/4,W/4) ∈ [0,1]
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
