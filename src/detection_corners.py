"""Détection de cartes UNO par méthode classique anchrée sur les couleurs et
mini-digits aux coins de chaque carte.

Stratégie :
1. Pour chaque couleur UNO (R/Y/G/B/K), on isole les blobs de cette couleur dans
   l'image (avec morphologie pour fermer l'ovale blanc central).
2. Chaque blob de bonne taille → rotated rect candidat.
3. Pour les blobs surdimensionnés (stacks même couleur), on détecte les
   mini-digits aux coins et on split.
4. On ajoute une pass "coin occlus" : digits blancs non rattachés à un blob
   colorié → hypothèse de carte cachée par-dessus.

Avantages vs détecteur appris :
- 0 params (libère 0.53M de budget)
- Robuste sur fonds bruités (les couleurs feuillage ne matchent pas exactement
  les ranges HSV des couleurs UNO saturées)
- Sépare naturellement les stacks de couleurs différentes
- Rotated rect → crops plus serrés que axis-aligned
"""
from __future__ import annotations

import cv2
import numpy as np

from .templates import (
    ASPECT_MAX,
    ASPECT_MIN,
    COLOR_RANGES,
    MIN_CARD_AREA,
    _color_mask,
    _warp_from_rect,
    dominant_color,
)

# Tailles attendues à 4000×2662 (carte ~250-400 px de large)
# Aire min = MIN_CARD_AREA (50k) hérité de templates.py
CARD_AREA_MAX = 250_000      # au-delà : suspicion de stack
STACK_AREA_RATIO = 1.45       # > 1.45× la médiane = stack

# Morphologie pour fermer l'ovale blanc au milieu d'une carte de couleur
COLOR_CLOSE_KERNEL = 45       # fermeture pour reconstituer la carte entière
COLOR_OPEN_KERNEL = 7         # ouverture pour nettoyer petits bruits

# Mini-digit aux coins
DIGIT_AREA_MIN = 250
DIGIT_AREA_MAX = 4_000
DIGIT_ASPECT_MAX = 2.8         # digits sont assez compacts
DIGIT_SURROUND_COLOR_DENSITY = 0.30  # >= 30% de pixels UNO autour du digit
DIGIT_CARD_W_RATIO = 8.0       # carte ≈ 8× largeur du digit (calibration approx)


def _color_mask_closed(hsv: np.ndarray, color: str,
                       close_k: int = COLOR_CLOSE_KERNEL,
                       open_k: int = COLOR_OPEN_KERNEL) -> np.ndarray:
    """Mask d'une couleur UNO + morphologie pour reconstituer le rect plein."""
    m = _color_mask(hsv, color)
    if close_k > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    if open_k > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    return m


def _find_color_card_rects(image: np.ndarray) -> list[dict]:
    """Pour chaque couleur, extrait les rotated rects candidats de cartes."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]

    candidates: list[dict] = []
    areas_single: list[float] = []

    for color in COLOR_RANGES:
        m = _color_mask_closed(hsv, color)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CARD_AREA:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            if min(rw, rh) < 80:
                continue
            ar = max(rw, rh) / min(rw, rh)
            if not (1.05 <= ar <= 2.2):
                continue
            candidates.append({
                "color": color, "rect": rect, "area": float(area),
                "cx": cx, "cy": cy, "contour": c,
            })
            if ASPECT_MIN <= ar <= ASPECT_MAX:
                areas_single.append(float(area))

    median_area = float(np.median(areas_single)) if areas_single else 150_000.0
    for c in candidates:
        c["median_area_ref"] = median_area
    return candidates


def _find_digit_candidates(image: np.ndarray) -> list[dict]:
    """Petits blobs blancs entourés majoritairement de couleur UNO → mini-digits.

    Chaque digit est un point d'ancrage qui prouve l'existence d'une carte
    (au moins son coin) à proximité.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]

    white_mask = ((hsv[..., 2] > 175) & (hsv[..., 1] < 55)).astype(np.uint8) * 255
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    color_mask_any = np.zeros((h, w), np.uint8)
    for color in COLOR_RANGES:
        color_mask_any = cv2.bitwise_or(color_mask_any, _color_mask(hsv, color))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask)
    digits: list[dict] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (DIGIT_AREA_MIN <= area <= DIGIT_AREA_MAX):
            continue
        ar = max(bw, bh) / max(min(bw, bh), 1)
        if ar > DIGIT_ASPECT_MAX:
            continue
        # Surroundings : dilate around blob and check UNO color density
        pad = max(8, int(0.4 * max(bw, bh)))
        y0, y1 = max(0, y - pad), min(h, y + bh + pad)
        x0, x1 = max(0, x - pad), min(w, x + bw + pad)
        local_color = color_mask_any[y0:y1, x0:x1]
        if local_color.size == 0:
            continue
        density = float(local_color.mean()) / 255.0
        if density < DIGIT_SURROUND_COLOR_DENSITY:
            continue
        digits.append({
            "cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "area": int(area), "color_density": density,
        })
    return digits


def _split_color_blob_by_digits(card: dict, digits: list[dict]) -> list[tuple]:
    """Pour un blob colorié surdimensionné, sépare en plusieurs cartes via les
    digits qu'il contient.

    Algorithme :
    - On garde les digits qui tombent dans la bbox du card.
    - Si 2 digits → potentiel 1 carte (digits diagonaux) → on garde le rect global.
    - Si ≥ 3 digits → potentiel multi-cartes → on cluster les digits par
      proximité 2-à-2 et on construit un rect par paire.
    - Si ≥ 4 digits → 2 cartes (2 paires diagonales).
    """
    rect = card["rect"]
    box = cv2.boxPoints(rect)
    x_min, y_min = box.min(axis=0)
    x_max, y_max = box.max(axis=0)
    inside = [d for d in digits
              if x_min <= d["cx"] <= x_max and y_min <= d["cy"] <= y_max]
    n = len(inside)
    if n < 4:
        return [rect]

    # Plusieurs cartes : on prend les digits 2 à 2 par paires proches
    # Stratégie simple : cluster K-means en 2 groupes spatialement
    pts = np.array([[d["cx"], d["cy"]] for d in inside], dtype=np.float32)
    _, labels, _ = cv2.kmeans(
        pts, K=2, bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
        attempts=4, flags=cv2.KMEANS_PP_CENTERS,
    )
    rects: list[tuple] = []
    for k in range(2):
        cluster = pts[labels.ravel() == k]
        if len(cluster) < 2:
            continue
        c_x_min, c_y_min = cluster.min(axis=0)
        c_x_max, c_y_max = cluster.max(axis=0)
        # Construire rotated rect à partir des 4 points (les digits sont aux
        # coins opposés en diagonale typiquement)
        cluster_center = cluster.mean(axis=0)
        # Pour estimer l'angle de la carte, prendre la direction du vecteur
        # entre les 2 digits les plus distants
        if len(cluster) >= 2:
            dists = np.linalg.norm(cluster[:, None] - cluster[None, :], axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            diag = cluster[j] - cluster[i]
            # Carte = 2:3, donc diag fait ~sqrt(4+9)/2 = 1.8× la moitié de la largeur
            diag_len = float(np.linalg.norm(diag))
            # Largeur ≈ diag_len / sqrt(13)/2 * 2 = diag_len * 2/sqrt(13)
            card_w = diag_len * 2 / np.sqrt(13)
            card_h = card_w * 1.5
            angle = float(np.degrees(np.arctan2(diag[1], diag[0])))
            # L'angle de la diagonale = angle_carte + atan(2/3) → soustraire ce offset
            angle -= float(np.degrees(np.arctan2(card_h, card_w)))
            rects.append(((float(cluster_center[0]), float(cluster_center[1])),
                          (card_w, card_h), angle))
    return rects or [rect]


def _hypothesize_card_from_digit(digit: dict, image_shape: tuple[int, int, int]
                                 ) -> tuple:
    """Hypothèse de carte autour d'un digit isolé (cas occlus).

    Le digit est typiquement au coin top-left ou bottom-right d'une carte.
    On hypothétise les 2 positions (sens 1 et sens 2 puisque la carte est
    symétrique 180°). On choisira la meilleure au stade de validation.
    """
    digit_w = max(digit["bbox"][2], digit["bbox"][3])
    card_w = digit_w * DIGIT_CARD_W_RATIO
    card_h = card_w * 1.5
    # Offset : le digit est dans le coin TL ou BR, donc le centre est
    # vers l'intérieur de la carte
    dx = card_w * 0.5 - digit_w * 0.8
    dy = card_h * 0.5 - digit_w * 0.8
    # Sens A : centre = digit + (dx, dy)
    center_a = (digit["cx"] + dx, digit["cy"] + dy)
    return ((float(center_a[0]), float(center_a[1])),
            (float(card_w), float(card_h)), 0.0)


def _looks_like_card_loose(warped: np.ndarray) -> bool:
    """Variante plus permissive de templates._looks_like_card pour les cartes
    occluses (on demande juste une présence d'un peu de blanc et/ou couleur)."""
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    central = hsv[int(h * 0.20):int(h * 0.80), int(w * 0.30):int(w * 0.70)]
    if central.size == 0:
        return False
    white_frac = float(((central[..., 2] > 180) & (central[..., 1] < 60)).mean())
    if white_frac > 0.05:
        return True
    dark_frac = float((central[..., 2] < 70).mean())
    if dark_frac > 0.20:
        return True
    return False


def detect_cards_classical(image: np.ndarray,
                           exclude_xy: tuple[float, float] | None = None,
                           exclude_radius: float = 220.0) -> list[dict]:
    """Pipeline complet de détection classique sans détecteur appris.

    Renvoie une liste de cards dans le même format que `detect_cards_in_scene`,
    plug-and-play avec `predict_scene`.
    """
    # 1. Cartes via blobs de couleur UNO
    color_cards = _find_color_card_rects(image)
    if not color_cards:
        return []
    median_area = color_cards[0]["median_area_ref"]

    # 2. Mini-digits aux coins (ancres)
    digits = _find_digit_candidates(image)

    # 3. Pour chaque blob colorié, sub-divise si stack
    final_rects: list[tuple] = []
    used_digit_ids: set[int] = set()
    for card in color_cards:
        if card["area"] > STACK_AREA_RATIO * median_area:
            rects = _split_color_blob_by_digits(card, digits)
            final_rects.extend(rects)
        else:
            final_rects.append(card["rect"])
        # Marque les digits utilisés (inside le blob)
        rect = card["rect"]
        box = cv2.boxPoints(rect)
        x_min, y_min = box.min(axis=0)
        x_max, y_max = box.max(axis=0)
        for i, d in enumerate(digits):
            if x_min <= d["cx"] <= x_max and y_min <= d["cy"] <= y_max:
                used_digit_ids.add(i)

    # 4. Digits orphelins → cartes potentiellement occluses
    for i, d in enumerate(digits):
        if i in used_digit_ids:
            continue
        # Validation extra : autour du digit doit y avoir une carte plausible
        rect = _hypothesize_card_from_digit(d, image.shape)
        warped = _warp_from_rect(image, rect)
        if _looks_like_card_loose(warped):
            final_rects.append(rect)

    # 5. Assemblage final + dédup
    cards: list[dict] = []
    for rect in final_rects:
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) < 80:
            continue
        # Exclusion zone token
        if exclude_xy is not None:
            ex, ey = exclude_xy
            if (cx - ex) ** 2 + (cy - ey) ** 2 < exclude_radius ** 2:
                continue
        warped = _warp_from_rect(image, rect)
        if warped.size == 0:
            continue
        cards.append({
            "cx": float(cx), "cy": float(cy), "rect": rect, "warped": warped,
            "color": dominant_color(warped), "source": "classical",
        })

    # Dédup par proximité (min_dist ~150 px)
    cards.sort(key=lambda c: -c["rect"][1][0] * c["rect"][1][1])  # plus gros d'abord
    kept: list[dict] = []
    for card in cards:
        too_close = any(
            (card["cx"] - k["cx"]) ** 2 + (card["cy"] - k["cy"]) ** 2 < 150 ** 2
            for k in kept
        )
        if not too_close:
            kept.append(card)
    return kept
