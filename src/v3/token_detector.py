"""Phase 5 V3 — Détection du token (HSV classique, 0 paramètre).

Token = jeton jaune rond OU noir rectangulaire posé près du joueur actif.
Stratégie : double seuillage HSV (jaune saturé / noir), morpho, composantes
connexes filtrées par aire et compacité. Renvoie (x, y) du centre ou None.

Réutilise la logique éprouvée du V2 (src.detection.detect_active_token) qui
gère déjà le seuil adaptatif fond clair/bruité, plus fiable qu'un seuil fixe.
"""
from __future__ import annotations

import numpy as np

from src.detection import detect_active_token


def detect_token(image: np.ndarray) -> tuple[float, float] | None:
    """Renvoie (x, y) pixels du centre du token, ou None.

    Wrappe `src.detection.detect_active_token` (V2) qui essaie noir puis jaune
    (ou l'inverse) selon la médiane V — robuste fond blanc ET bruité.
    """
    return detect_active_token(image)
