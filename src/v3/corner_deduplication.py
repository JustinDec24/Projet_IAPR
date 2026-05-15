"""Phase 5 V3 — Déduplication géométrique des coins → cartes.

Une carte UNO a 2 coins-digit diagonalement opposés à distance ≈ diagonale de
la carte. Deux coins de MÊME classe à cette distance = la même carte.

Algo (greedy) :
  - trie les coins par score décroissant
  - pour chaque coin non assigné, cherche un partenaire compatible
    (même classe, distance ∈ [d_min, d_max]) → forme une carte 2-coins
  - les coins isolés (occlusion forte, 1 seul visible) → carte 1-coin
  - confiance carte = moyenne des scores des coins associés
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Corner:
    x: float
    y: float
    label: str
    score: float


@dataclass
class Card:
    cx: float
    cy: float
    label: str
    confidence: float
    n_corners: int
    corners: list[Corner] = field(default_factory=list)


def deduplicate_corners(corners: list[Corner],
                        diag_min: float = 130.0,
                        diag_max: float = 300.0,
                        require_same_label: bool = False) -> list[Card]:
    """Regroupe les coins en cartes.

    Args:
        corners : liste de Corner (x,y px, label classe, score)
        diag_min/max : bornes de la distance entre les 2 coins-digit d'une carte
                       (≈ diagonale carte en pixels image).
        require_same_label : si False (défaut V3-C), pairing GÉOMÉTRIQUE pur
                       (distance seule). Le classifieur par-coin étant bruité
                       (~76 % réel), exiger l'accord de label fait échouer le
                       pairing → cartes mal localisées. En géométrique pur, la
                       carte est bien centrée (milieu des 2 coins) et le label
                       = celui du coin le plus confiant.
    """
    order = sorted(range(len(corners)), key=lambda i: -corners[i].score)
    used = [False] * len(corners)
    cards: list[Card] = []
    mid = (diag_min + diag_max) / 2

    for i in order:
        if used[i]:
            continue
        ci = corners[i]
        best_j, best_d = -1, None
        for j in order:
            if j == i or used[j]:
                continue
            cj = corners[j]
            if require_same_label and cj.label != ci.label:
                continue
            d = math.hypot(ci.x - cj.x, ci.y - cj.y)
            if diag_min <= d <= diag_max:
                if best_d is None or abs(d - mid) < abs(best_d - mid):
                    best_j, best_d = j, d
        if best_j >= 0:
            cj = corners[best_j]
            used[i] = used[best_j] = True
            # label = coin le plus confiant des 2 (max, pas accord requis)
            top = ci if ci.score >= cj.score else cj
            cards.append(Card(
                cx=(ci.x + cj.x) / 2, cy=(ci.y + cj.y) / 2,
                label=top.label,
                confidence=(ci.score + cj.score) / 2,
                n_corners=2, corners=[ci, cj]))
        else:
            used[i] = True
            cards.append(Card(cx=ci.x, cy=ci.y, label=ci.label,
                              confidence=ci.score * 0.85,
                              n_corners=1, corners=[ci]))
    return cards
