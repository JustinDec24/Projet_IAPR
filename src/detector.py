"""Détecteur de cartes anchor-free (style CenterNet), entraîné from scratch
sur les bboxes annotées manuellement.

Architecture :
- Backbone : 4 stages convolutifs (downsample 16× au total) ~ 3M params
- Head : 5 channels par cellule (objectness + cx_norm + cy_norm + w_norm + h_norm)

Cible d'entraînement :
- Heatmap gaussienne 2D centrée sur le centre de chaque bbox GT
- Régression L1 des coordonnées (cx, cy, w, h) au pixel central uniquement

Inference :
- Seuillage de la heatmap d'objectness
- Pour chaque pixel positif : lire les 4 valeurs de bbox → bbox absolue
- NMS pour dédupliquer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_c: int, out_c: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=kernel // 2,
                  stride=stride, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
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
        return F.relu(out + self.shortcut(x), inplace=True)


class CardDetector(nn.Module):
    """Anchor-free 1-class detector (objectness heatmap + bbox regression).

    Input : (B, 3, H, W) image normalisée [0, 1] BGR→RGB
    Output : dict avec
        - 'obj'  : (B, 1, H/16, W/16)   logits d'objectness (sigmoid au sample)
        - 'bbox' : (B, 4, H/16, W/16)   (cx_norm, cy_norm, w_norm, h_norm)
                                          tous normalisés à [0, 1] de l'image
    """

    def __init__(self,
                 channels: tuple[int, int, int, int] = (16, 32, 64, 96)) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        # Stem 2×
        self.stem = nn.Sequential(_conv_bn(3, c1, kernel=3, stride=2))
        # Stage 1 : c1 → c2, downsample 2× (total 4×)
        self.stage1 = nn.Sequential(BasicBlock(c1, c2, stride=2), BasicBlock(c2, c2))
        # Stage 2 : c2 → c3, downsample 2× (total 8×)
        self.stage2 = nn.Sequential(BasicBlock(c2, c3, stride=2), BasicBlock(c3, c3))
        # Stage 3 : c3 → c4, downsample 2× (total 16×)
        self.stage3 = nn.Sequential(BasicBlock(c3, c4, stride=2), BasicBlock(c4, c4))

        # Head : conv 3×3 commun puis 1×1 pour objectness et bbox
        head_c = max(64, c4 // 2)
        self.head_shared = _conv_bn(c4, head_c)
        self.head_obj = nn.Conv2d(head_c, 1, kernel_size=1)
        self.head_bbox = nn.Conv2d(head_c, 4, kernel_size=1)

        # Init du biais d'objectness pour partir bas (prior = beaucoup de neg)
        nn.init.constant_(self.head_obj.bias, -4.6)  # sigmoid(-4.6) ≈ 0.01

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head_shared(x)
        return {
            "obj": self.head_obj(x),         # logits
            "bbox": self.head_bbox(x).sigmoid(),  # bbox dans [0, 1]
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
