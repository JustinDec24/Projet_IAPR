"""Extrait des crops propres pour les 54 cartes UNO depuis reference_images/.

Stratégie : segmentation globale par saturation HSV + un Gaussian blur agressif
pour propager la saturation des cadres colorés vers l'intérieur de chaque
carte. Les blobs trop gros (cartes mergées) sont splittés selon le ratio
aire/aire-médiane. Chaque crop warpé est ensuite validé par sa couleur
dominante, et étiqueté selon le layout connu de chaque image de référence.
"""

from pathlib import Path

import cv2
import numpy as np

CARD_W, CARD_H = 200, 300
MIN_CARD_AREA = 50_000
ASPECT_MIN, ASPECT_MAX = 1.2, 1.8

LAYOUTS: dict[str, dict] = {
    "L1000765": {
        "rows": [
            ["y_3", "r_3", "b_3", "g_3"],
            ["y_2", "r_2", "b_2", "g_2"],
            ["y_1", "r_1", "b_1", "g_1"],
        ],
    },
    "L1000766": {
        "rows": [
            ["y_6", "r_6", "b_6", "g_6"],
            ["y_5", "r_5", "b_5", "g_5"],
            ["y_4", "r_4", "b_4", "g_4"],
        ],
        # Trois cartes (y_5, y_4, r_4) prises dans un blob L-shape mergé que
        # le split géométrique global ne récupère pas. On donne juste un point
        # de départ (cx, cy) ; la détection locale par couleur récupère le bon
        # rotated rect (avec le vrai angle de la carte).
        "fallbacks": {
            "y_5": (1719, 1571),
            "y_4": (1755, 2174),
            "r_4": (2235, 2174),
        },
    },
    "L1000767": {
        "rows": [
            ["y_9", "r_9", "b_9", "g_9"],
            ["y_8", "r_8", "b_8", "g_8"],
            ["y_7", "r_7", "b_7", "g_7"],
        ],
        "black_top_to_bottom": ["draw_4", "wild"],
    },
    "L1000768": {
        "rows": [
            ["y_reverse", "r_reverse", "b_reverse", "g_reverse"],
            ["y_skip", "r_skip", "b_skip", "g_skip"],
            ["y_0", "r_0", "b_0", "g_0"],
            ["y_draw_2", "b_draw_2", "r_draw_2", "g_draw_2"],
        ],
    },
}

COLOR_RANGES: dict[str, list[tuple]] = {
    "y": [((20, 80, 80), (40, 255, 255))],
    "g": [((40, 60, 60), (85, 255, 255))],
    "b": [((90, 80, 60), (135, 255, 255))],
    "r": [((0, 80, 60), (12, 255, 255)), ((165, 80, 60), (180, 255, 255))],
    "k": [((0, 0, 0), (180, 80, 70))],
}


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Réordonne 4 coins en TL, TR, BR, BL avec orientation portrait (h ≥ w)."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect = np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )
    w = np.linalg.norm(rect[1] - rect[0])
    h = np.linalg.norm(rect[3] - rect[0])
    if w > h:
        rect = np.array([rect[1], rect[2], rect[3], rect[0]], dtype=np.float32)
    return rect


def _warp_from_rect(image: np.ndarray, rect: tuple) -> np.ndarray:
    box = cv2.boxPoints(rect)
    src = _order_corners(box)
    dst = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
        dtype=np.float32,
    )
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (CARD_W, CARD_H))


def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    parts = [cv2.inRange(hsv, np.array(low), np.array(high)) for low, high in COLOR_RANGES[color]]
    return np.bitwise_or.reduce(parts)


def dominant_color(warped: np.ndarray) -> str:
    """Renvoie la couleur dominante d'un crop de carte parmi y/g/b/r/k."""
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    counts = {c: int(_color_mask(hsv, c).sum() // 255) for c in COLOR_RANGES}
    return max(counts, key=counts.get)


def card_mask_from_image(image: np.ndarray) -> np.ndarray:
    """Masque global des cartes (saturation floue + composante sombre pour wild/draw_4)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    sat_blur = cv2.GaussianBlur(sat, (0, 0), sigmaX=25)
    val_blur = cv2.GaussianBlur(val, (0, 0), sigmaX=25)
    mask = ((sat_blur > 25) | (val_blur < 100)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask


def _split_rect(rect: tuple, n: int) -> list[tuple]:
    """Coupe un rotated rect en n sous-rectangles le long de son grand axe."""
    if n <= 1:
        return [rect]
    (cx, cy), (w, h), angle = rect
    rad = np.radians(angle)
    if w >= h:
        ux, uy = np.cos(rad), np.sin(rad)
        new_w = w / n
        return [
            (
                (cx + (i - (n - 1) / 2) * new_w * ux, cy + (i - (n - 1) / 2) * new_w * uy),
                (new_w, h),
                angle,
            )
            for i in range(n)
        ]
    ux, uy = -np.sin(rad), np.cos(rad)
    new_h = h / n
    return [
        (
            (cx + (i - (n - 1) / 2) * new_h * ux, cy + (i - (n - 1) / 2) * new_h * uy),
            (w, new_h),
            angle,
        )
        for i in range(n)
    ]


def detect_cards(image: np.ndarray) -> list[dict]:
    """Segmente les cartes via le masque global ; split géométriquement les blobs surdimensionnés."""
    mask = card_mask_from_image(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        candidates.append((rect, area, ar))
        if ASPECT_MIN <= ar <= ASPECT_MAX:
            single_areas.append(area)
    median_area = float(np.median(single_areas)) if single_areas else 150_000

    cards: list[dict] = []
    for rect, area, _ in candidates:
        n = max(1, round(area / median_area)) if area > 1.5 * median_area else 1
        for sub in _split_rect(rect, n):
            warped = _warp_from_rect(image, sub)
            (cx, cy), (w, h), angle = sub
            cards.append(
                {
                    "cx": cx, "cy": cy, "rect_size": (w, h), "rect_angle": angle,
                    "warped": warped, "color": dominant_color(warped),
                }
            )
    return cards


def _cluster_rows(cards: list[dict], n_rows: int) -> list[list[dict]]:
    """Coupe les cartes en n_rows lignes via les (n_rows-1) plus grands gaps de cy."""
    if not cards:
        return [[] for _ in range(n_rows)]
    cards = sorted(cards, key=lambda c: c["cy"])
    if len(cards) <= n_rows:
        rows = [[c] for c in cards]
        rows.extend([[] for _ in range(n_rows - len(cards))])
        return rows
    cys = np.array([c["cy"] for c in cards])
    gaps = np.diff(cys)
    boundary_idx = sorted(np.argsort(gaps)[-(n_rows - 1):].tolist())
    rows: list[list[dict]] = []
    start = 0
    for idx in boundary_idx:
        rows.append(cards[start : idx + 1])
        start = idx + 1
    rows.append(cards[start:])
    return [sorted(r, key=lambda c: c["cx"]) for r in rows]


def extract_from_image(image_path: Path) -> dict[str, np.ndarray]:
    """Étiquette chaque carte détectée par sa couleur + position en y dans la couleur.

    Chaque image de référence place chaque couleur ≤ 1 fois par rangée. Donc pour
    chaque couleur, trier les cartes détectées par cy donne directement
    l'assignation rangée 0 → 1 → 2 → ... — sans avoir à clusteriser globalement.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    layout = LAYOUTS[image_path.stem]
    cards = detect_cards(image)

    by_color: dict[str, list[dict]] = {c: [] for c in COLOR_RANGES}
    for card in cards:
        by_color[card["color"]].append(card)
    for color in by_color:
        by_color[color].sort(key=lambda c: c["cy"])

    result: dict[str, np.ndarray] = {}
    for row_idx, row_labels in enumerate(layout["rows"]):
        for label in row_labels:
            color = label[0]
            if color in by_color and row_idx < len(by_color[color]):
                result[label] = by_color[color][row_idx]["warped"]

    for label, card in zip(layout.get("black_top_to_bottom", []), by_color.get("k", [])):
        result[label] = card["warped"]

    # Fallbacks : ROI rectangulaire serrée (juste plus grande qu'une carte
    # mais plus petite que l'inter-card spacing) autour de (cx, cy). On
    # détecte la couleur attendue dans la ROI et on warpe le rotated rect
    # trouvé — angle correct, voisins exclus.
    for label, (cx, cy) in layout.get("fallbacks", {}).items():
        color = label[0] if label not in ("wild", "draw_4") else "k"
        warped = _fallback_warp(image, cx, cy, color)
        if warped is not None:
            result[label] = warped
    return result


def _fallback_warp(
    image: np.ndarray, cx: float, cy: float, color: str,
    radius_x: int = 240, radius_y: int = 300,
) -> np.ndarray | None:
    """Warpe la carte de la couleur attendue dans une ROI serrée autour de (cx, cy)."""
    h, w = image.shape[:2]
    x0 = max(0, int(cx - radius_x))
    y0 = max(0, int(cy - radius_y))
    x1 = min(w, int(cx + radius_x))
    y1 = min(h, int(cy + radius_y))
    roi_hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)

    raw = _color_mask(roi_hsv, color)
    blurred = cv2.GaussianBlur(raw.astype(np.float32), (0, 0), sigmaX=20)
    mask = (blurred > 25).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CARD_AREA // 2]
    if not valid:
        return None
    best = max(valid, key=cv2.contourArea)
    (lcx, lcy), size, angle = cv2.minAreaRect(best)
    return _warp_from_rect(image, ((lcx + x0, lcy + y0), size, angle))
