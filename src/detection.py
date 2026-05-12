"""Détection des cartes et du jeton actif dans les images de jeu UNO.

Stratégies combinées :
- **Saturation** (fond blanc) : carte = blob saturé après gaussian blur. Pour les
  stacks de cartes empilées, on split d'abord par couleur (si couleurs distinctes
  dans le blob), sinon par splitting géométrique le long du grand axe.
- **Ovales blancs** (fond bruité) : la saturation est inutilisable, mais l'ovale
  blanc au centre de chaque carte forme un trou enclosed dans le mask saturé.

Toutes les détections sont dédupliquées par proximité, et la zone autour du
jeton actif est exclue en aval (`predict_scene`).
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
    _split_rect,
    _warp_from_rect,
    card_mask_from_image,
    dominant_color,
)

# Ovales blancs : seuils relâchés pour rattraper plus de cartes (recall) sur fond bruité.
OVAL_AREA_MIN = 6_500
OVAL_AREA_MAX = 40_000
OVAL_MIN_SIDE = 60
OVAL_ASPECT_MIN, OVAL_ASPECT_MAX = 1.10, 2.0
OVAL_SOLIDITY_MIN = 0.60
OVAL_TO_CARD_SCALE = 1.50

# Split par couleur (stacks) : si un blob a une aire suspecte ET contient ≥2 couleurs
# distinctes en quantité substantielle, on split par couleur.
STACK_AREA_RATIO = 1.20  # seuil d'aire au-dessus duquel on suspecte un stack
COLOR_CLUSTER_MIN_PIXELS = 3_000
COLOR_SPLIT_SIGMA = 18.0
COLOR_SPLIT_THRESHOLD = 30


def _enclosed_holes(sat_mask: np.ndarray) -> np.ndarray:
    """Régions claires enclosed dans le sat_mask (= ovales blancs des cartes)."""
    h, w = sat_mask.shape
    inv = cv2.bitwise_not(sat_mask)
    flood = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[sy, sx] == 255:
            cv2.floodFill(flood, ff_mask, (sx, sy), 0)
    return flood


def _looks_like_card(warped: np.ndarray) -> bool:
    """Vérifie qu'un crop a la signature géométrique d'une carte UNO.

    Une vraie carte a un ovale blanc connexe au centre (cartes colorées) OU
    un fond très sombre dominant (wild/draw_4). Les patches de feuillage ont
    des pixels blancs/sombres mais éparpillés.
    """
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    central = hsv[int(h * 0.20):int(h * 0.80), int(w * 0.30):int(w * 0.70)]
    central_area = central.shape[0] * central.shape[1]

    white_mask = ((central[..., 2] > 200) & (central[..., 1] < 60)).astype(np.uint8) * 255
    if int(white_mask.sum()) > 0:
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(cv2.contourArea(c) for c in contours)
            if largest > 0.10 * central_area:
                return True

    if float((central[..., 2] < 70).mean()) > 0.35:
        return True
    return False


def _split_by_color_clusters(contour: np.ndarray, image: np.ndarray,
                             hsv: np.ndarray, median_area: float
                             ) -> list[tuple]:
    """Si le blob contient plusieurs clusters de couleurs distinctes, renvoie un
    rotated rect par cluster. Sinon liste vide.

    Algo : pour chaque couleur UNO, on prend les pixels de cette couleur DANS le
    contour, on les floute pour fusionner en blob de taille de carte, puis on
    extrait des rotated rects par composante connexe. Cela permet de séparer
    proprement un stack jaune+rouge en deux rects distincts.
    """
    h, w = image.shape[:2]
    contour_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    found_rects: list[tuple] = []
    for color in COLOR_RANGES:
        color_pixels = cv2.bitwise_and(_color_mask(hsv, color), contour_mask)
        if int(color_pixels.sum() // 255) < COLOR_CLUSTER_MIN_PIXELS:
            continue
        blurred = cv2.GaussianBlur(color_pixels.astype(np.float32), (0, 0),
                                   sigmaX=COLOR_SPLIT_SIGMA)
        cluster_mask = (blurred > COLOR_SPLIT_THRESHOLD).astype(np.uint8) * 255
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN,
                                        np.ones((11, 11), np.uint8))
        for cc in cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(cc) < median_area * 0.55:
                continue
            r = cv2.minAreaRect(cc)
            (_, _), (rw, rh), _ = r
            if min(rw, rh) < 50:
                continue
            ar = max(rw, rh) / min(rw, rh)
            if not (1.05 <= ar <= 2.2):
                continue
            found_rects.append(r)
    return found_rects


def _detect_via_saturation(image: np.ndarray) -> list[dict]:
    """Approche fond blanc : cartes = blobs saturés directs.

    Pour les blobs surdimensionnés (stacks de cartes empilées) :
    1. on tente d'abord un split par couleur (si couleurs distinctes)
    2. sinon split géométrique le long du grand axe (cas dernier recours).
    """
    mask = card_mask_from_image(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    candidates: list[tuple] = []
    single_areas: list[float] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CARD_AREA:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if min(w, h) < 50:
            continue
        ar = max(w, h) / min(w, h)
        candidates.append((c, rect, area, ar))
        if ASPECT_MIN <= ar <= ASPECT_MAX:
            single_areas.append(area)
    median_area = float(np.median(single_areas)) if single_areas else 150_000

    cards: list[dict] = []
    for c, rect, area, ar in candidates:
        # Si l'aire est suspecte (potentiel stack), on essaie d'abord per-color
        sub_rects: list[tuple] = []
        if area > STACK_AREA_RATIO * median_area:
            color_split = _split_by_color_clusters(c, image, hsv, median_area)
            if len(color_split) >= 2:
                sub_rects = color_split
            else:
                n = max(1, round(area / median_area))
                sub_rects = _split_rect(rect, n)
        else:
            sub_rects = [rect]

        for sub in sub_rects:
            (cx, cy), _, _ = sub
            warped = _warp_from_rect(image, sub)
            if not _looks_like_card(warped):
                continue
            cards.append(
                {"cx": cx, "cy": cy, "rect": sub, "warped": warped,
                 "color": dominant_color(warped), "source": "sat"}
            )
    return cards


def _detect_via_ovals(image: np.ndarray) -> list[dict]:
    """Approche fond bruité : ovales blancs enclosed → rect de carte par ×scale."""
    sat_mask = card_mask_from_image(image)
    holes = _enclosed_holes(sat_mask)
    holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, np.ones((81, 81), np.uint8))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards: list[dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (OVAL_AREA_MIN < area < OVAL_AREA_MAX):
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w_o, h_o), angle = rect
        if min(w_o, h_o) < OVAL_MIN_SIDE:
            continue
        ar = max(w_o, h_o) / min(w_o, h_o)
        if not (OVAL_ASPECT_MIN <= ar <= OVAL_ASPECT_MAX):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area > 0 and area / hull_area < OVAL_SOLIDITY_MIN:
            continue
        card_rect = ((cx, cy), (w_o * OVAL_TO_CARD_SCALE, h_o * OVAL_TO_CARD_SCALE), angle)
        warped = _warp_from_rect(image, card_rect)
        if not _looks_like_card(warped):
            continue
        cards.append(
            {"cx": cx, "cy": cy, "rect": card_rect, "warped": warped,
             "color": dominant_color(warped), "source": "oval"}
        )
    return cards


def _deduplicate(cards: list[dict], min_dist: float = 200) -> list[dict]:
    """Si deux détections sont à < min_dist, garde celle qui vient du sat_mask."""
    cards_sorted = sorted(cards, key=lambda c: 0 if c["source"] == "sat" else 1)
    kept: list[dict] = []
    for card in cards_sorted:
        too_close = any(
            (card["cx"] - k["cx"]) ** 2 + (card["cy"] - k["cy"]) ** 2 < min_dist ** 2
            for k in kept
        )
        if not too_close:
            kept.append(card)
    return kept


def detect_cards_in_scene(image: np.ndarray,
                          exclude_xy: tuple[float, float] | None = None,
                          exclude_radius: float = 220) -> list[dict]:
    """Cartes détectées (combinaison des deux stratégies, dédupliquée).

    Si `exclude_xy` est fourni, on filtre les détections trop proches de ce point
    (typiquement la position du jeton actif, pour ne pas le classer comme carte).
    """
    cards = _detect_via_saturation(image) + _detect_via_ovals(image)
    cards = _deduplicate(cards)
    if exclude_xy is not None:
        ex, ey = exclude_xy
        cards = [c for c in cards
                 if (c["cx"] - ex) ** 2 + (c["cy"] - ey) ** 2 > exclude_radius ** 2]
    return cards


def detect_active_token(image: np.ndarray) -> tuple[float, float] | None:
    """Détecte le jeton actif (noir rectangulaire sur fond blanc OU jaune rond sur fond bruité)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    val = hsv[..., 2]
    median_v = float(np.median(val))
    if median_v > 180:
        return _detect_black_token(val, median_v)
    return _detect_yellow_token(hsv)


def _detect_black_token(val: np.ndarray, median_v: float) -> tuple[float, float] | None:
    threshold = max(20, int(median_v - 80))
    val_smooth = cv2.medianBlur(val, 7)
    black = (val_smooth < threshold).astype(np.uint8) * 255
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    best = None
    for c in cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if not (1_500 < area < 70_000):
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        if min(w, h) < 30:
            continue
        ar = max(w, h) / min(w, h)
        if ar > 3.0:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.85:
            continue
        if best is None or area > best[2]:
            best = (float(cx), float(cy), area)
    return (best[0], best[1]) if best else None


def _detect_yellow_token(hsv: np.ndarray) -> tuple[float, float] | None:
    yellow = cv2.inRange(hsv, np.array([20, 130, 130]), np.array([38, 255, 255]))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    best = None
    for c in cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if not (5_000 < area < 60_000):
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        circle_area = np.pi * r * r
        if circle_area == 0 or area / circle_area < 0.7:
            continue
        circ = area / circle_area
        if best is None or circ > best[3]:
            best = (float(cx), float(cy), area, circ)
    return (best[0], best[1]) if best else None


def assign_player(cx: float, cy: float, image_shape: tuple[int, int, int],
                  center_radius_frac: float = 0.18) -> str:
    """Assigne (cx, cy) à center / p1 / p2 / p3 / p4 selon la position dans l'image."""
    h, w = image_shape[:2]
    dx = cx - w / 2
    dy = cy - h / 2
    diag = float(np.hypot(w, h))
    if np.hypot(dx, dy) < center_radius_frac * diag / 2:
        return "center"
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if -45 <= angle < 45:
        return "p2"
    if 45 <= angle < 135:
        return "p1"
    if -135 <= angle < -45:
        return "p3"
    return "p4"


def assign_token_to_player(token_xy: tuple[float, float], cards: list[dict],
                           image_shape: tuple[int, int, int]) -> str | None:
    """Assigne un jeton au joueur dont la carte la plus proche appartient."""
    tx, ty = token_xy
    best_player: str | None = None
    best_dist = float("inf")
    for card in cards:
        player = assign_player(card["cx"], card["cy"], image_shape)
        if player == "center":
            continue
        d2 = (card["cx"] - tx) ** 2 + (card["cy"] - ty) ** 2
        if d2 < best_dist:
            best_dist = d2
            best_player = player
    return best_player
