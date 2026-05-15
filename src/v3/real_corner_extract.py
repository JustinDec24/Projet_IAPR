"""Phase 3 V3 — Extraction de coins-digit *rotation-aware* depuis les bboxes
YOLO axis-aligned des vraies images.

Problème : les annotations manuelles sont des bboxes axis-aligned de la carte
ENTIÈRE. Les cartes sont souvent tournées (±20°) → un coin dérivé
géométriquement de la bbox tombe à côté du vrai digit.

Solution : dans la région de la bbox (+marge), on fitte un rotated rect via le
masque de couleur dominante UNO (cv2.minAreaRect), puis on place les 2
coins-digit (TL et BR de la carte) ~14 % vers l'intérieur des coins du rect.

Fallback si la couleur ne donne rien (cartes noires wild/draw_4 ou détection
ratée) : approximation axis-aligned d'origine.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.templates import COLOR_RANGES, _color_mask, _order_corners

# Inset du digit depuis le coin de la carte (fraction de la diagonale demi-côté)
DIGIT_INSET = 0.16


def _axis_aligned_corners(cx: float, cy: float, bw: float, bh: float,
                          w: int, h: int) -> list[tuple[float, float]]:
    """Approximation d'origine (fallback)."""
    x_tl = (cx - bw / 2 * (1 - 2 * DIGIT_INSET)) * w
    y_tl = (cy - bh / 2 * (1 - 2 * DIGIT_INSET)) * h
    x_br = (cx + bw / 2 * (1 - 2 * DIGIT_INSET)) * w
    y_br = (cy + bh / 2 * (1 - 2 * DIGIT_INSET)) * h
    return [(x_tl, y_tl), (x_br, y_br)]


def corners_from_bbox(image: np.ndarray, cx: float, cy: float,
                      bw: float, bh: float) -> list[tuple[float, float]]:
    """Renvoie les 2 coins-digit (TL, BR) en pixels image.

    Args sont normalisés [0,1] (format YOLO). `image` est BGR.
    """
    h, w = image.shape[:2]
    # Région de la bbox + marge 12 %
    x0 = int(np.clip((cx - bw / 2) * w - 0.12 * bw * w, 0, w - 1))
    y0 = int(np.clip((cy - bh / 2) * h - 0.12 * bh * h, 0, h - 1))
    x1 = int(np.clip((cx + bw / 2) * w + 0.12 * bw * w, 0, w))
    y1 = int(np.clip((cy + bh / 2) * h + 0.12 * bh * h, 0, h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return _axis_aligned_corners(cx, cy, bw, bh, w, h)

    roi = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Couleur dominante UNO dans la ROI (hors blanc / fond)
    best_color, best_count, best_mask = None, 0, None
    for color in COLOR_RANGES:
        m = _color_mask(hsv, color)
        c = int(m.sum() // 255)
        if c > best_count:
            best_count, best_color, best_mask = c, color, m

    roi_area = (x1 - x0) * (y1 - y0)
    if best_mask is None or best_count < 0.06 * roi_area:
        # Carte noire (wild/draw_4) ou détection couleur trop faible
        return _axis_aligned_corners(cx, cy, bw, bh, w, h)

    # Plus gros contour de la couleur dominante → rotated rect
    mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE,
                            np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _axis_aligned_corners(cx, cy, bw, bh, w, h)
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.05 * roi_area:
        return _axis_aligned_corners(cx, cy, bw, bh, w, h)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)              # 4 coins (repère ROI)
    ordered = _order_corners(box)          # TL, TR, BR, BL portrait
    tl, _tr, br, _bl = ordered
    center = ordered.mean(axis=0)
    # Coins-digit = coins de carte tirés vers le centre de DIGIT_INSET
    d_tl = tl + (center - tl) * DIGIT_INSET
    d_br = br + (center - br) * DIGIT_INSET
    return [(float(d_tl[0] + x0), float(d_tl[1] + y0)),
            (float(d_br[0] + x0), float(d_br[1] + y0))]


def load_real_corners(image: np.ndarray, annot_lines: list[str]
                      ) -> list[tuple[float, float]]:
    """Pour toutes les bboxes YOLO d'une image, renvoie tous les coins (px)."""
    pts: list[tuple[float, float]] = []
    for line in annot_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        pts.extend(corners_from_bbox(image, cx, cy, bw, bh))
    return pts
