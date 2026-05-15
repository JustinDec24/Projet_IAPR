"""Matcher template pour la carte centrale (0 paramètre, classique).

La carte du milieu est le cas idéal : isolée, entière, nette, grande. Un
nearest-neighbor sur les 54 templates de référence y est très fiable, alors
qu'il échoue dans les zones joueurs (occlusion/chevauchement).

Méthode :
  1. couleur dominante du crop → restreint aux ~10-14 templates de cette couleur
  2. dans ce sous-ensemble, similarité structurelle (NCC sur le canal de
     luminance égalisé) entre crop warpé et template, aux rotations 0°/180°
     (le gros glyphe UNO est lisible dans les 2 sens) + flips
  3. meilleur match = label, avec un score de confiance [0,1]

Les templates sont des *données de référence* (assets internes), pas des
poids de modèle → conforme à la contrainte 12M params.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import CARD_CLASSES
from src.templates import CARD_H, CARD_W, _color_mask

_TPL_CACHE: dict[str, np.ndarray] | None = None
_TPL_GRAY: dict[str, list[np.ndarray]] | None = None


def _color_of(label: str) -> str:
    return "k" if label in ("wild", "draw_4") else label[0]


def _prep_gray(bgr: np.ndarray) -> np.ndarray:
    """Luminance égalisée, taille canonique, focalisée sur la zone centrale
    (le grand glyphe), insensible aux bords/usure."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    g = cv2.equalizeHist(g)
    h, w = g.shape
    # zone centrale 60% (le gros chiffre/symbole discriminant)
    return g[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]


def _load_templates(tpl_dir: Path) -> None:
    global _TPL_CACHE, _TPL_GRAY
    if _TPL_CACHE is not None:
        return
    _TPL_CACHE, _TPL_GRAY = {}, {}
    for label in CARD_CLASSES:
        p = tpl_dir / f"{label}.png"
        img = cv2.imread(str(p))
        if img is None:
            continue
        _TPL_CACHE[label] = img
        base = _prep_gray(img)
        # variantes : 0° et 180° (symétrie de lecture UNO) + flip horizontal
        variants = [base,
                    cv2.rotate(base, cv2.ROTATE_180),
                    cv2.flip(base, 1),
                    cv2.flip(cv2.rotate(base, cv2.ROTATE_180), 1)]
        _TPL_GRAY[label] = variants


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation ∈ [-1,1]."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def match_center_card(warped_bgr: np.ndarray,
                      tpl_dir: Path) -> tuple[str, float]:
    """Renvoie (label, confiance∈[0,1]) pour un crop de carte centrale warpé."""
    _load_templates(tpl_dir)
    if not _TPL_CACHE:
        return "", 0.0

    hsv = cv2.cvtColor(cv2.resize(warped_bgr, (CARD_W, CARD_H)),
                       cv2.COLOR_BGR2HSV)
    counts = {c: int(_color_mask(hsv, c).sum() // 255)
              for c in ("r", "g", "b", "y", "k")}
    dom = max(counts, key=counts.get)

    cand = [lbl for lbl in _TPL_CACHE if _color_of(lbl) == dom]
    if not cand:  # couleur ambiguë → tous
        cand = list(_TPL_CACHE)

    q = _prep_gray(warped_bgr)
    best_lbl, best_s = "", -2.0
    second_s = -2.0
    for lbl in cand:
        s = max(_ncc(q, v) for v in _TPL_GRAY[lbl])
        if s > best_s:
            second_s = best_s
            best_s, best_lbl = s, lbl
        elif s > second_s:
            second_s = s
    # confiance = NCC du meilleur, modulée par la marge sur le 2e
    margin = max(0.0, best_s - second_s)
    conf = float(np.clip(0.5 * (best_s + 1.0) + 0.5 * margin, 0.0, 1.0))
    return best_lbl, conf
