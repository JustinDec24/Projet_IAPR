"""Hybride : corner-detector V3 (localisation, 0.985 réel, robuste occlusion)
+ classifieur full-card V2 (0.925 val, fort sur carte entière).

On contourne le maillon faible (corner-classifier 0.76 réel) :
  1. corner-detector → coins (x,y,score)  [excellent : R0.99]
  2. classification grossière des coins → label provisoire (juste pour pairer
     les coins de MÊME carte par accord+distance)
  3. dédup → pour chaque carte, on reconstruit le rotated rect depuis les 2
     coins-digit diagonaux (modèle géométrique carte 2:3) ; pour les coins
     seuls, rect estimé par taille médiane
  4. warp 200×300 → classifieur full-card V2 (le vrai label vient de là)
  5. assignation géométrique V2 (distance centre + angle)

Budget : corner_detector 3.19M + classifier student 6.31M = 9.5M < 12M.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config import IDX_TO_CLASS
from src.detection import (assign_player, assign_token_to_player,
                           detect_active_token)
from src.inference import Classifier
from src.templates import _warp_from_rect
from src.v3.corner_classifier_dataset import _crop_around
from src.v3.corner_detector_runtime import CornerDetectorRuntime
from src.v3.predict_scene_v3 import CornerClassifierRuntime

# Carte UNO 2:3. Coins-digit à 14 % → distance digit-corner = 1.298·w.
DIGIT_INSET = 0.14
_K = math.sqrt((1 - 2 * DIGIT_INSET) ** 2 * (1 + 1.5 ** 2))  # ≈1.298
_DIAG_ANGLE = math.degrees(math.atan2(1.5, 1.0))              # 56.3°


@dataclass
class ScenePredHybrid:
    image_id: str
    center_card: str = "EMPTY"
    active_player: str = "EMPTY"
    player_cards: dict[str, list[str]] = field(
        default_factory=lambda: {p: [] for p in ("p1", "p2", "p3", "p4")})

    def to_csv_row(self) -> dict[str, str]:
        def hand(c: list[str]) -> str:
            return ";".join(c) if c else "EMPTY"
        return {
            "image_id": self.image_id,
            "center_card": self.center_card or "EMPTY",
            "active_player": self.active_player or "EMPTY",
            "player_1_cards": hand(self.player_cards["p1"]),
            "player_2_cards": hand(self.player_cards["p2"]),
            "player_3_cards": hand(self.player_cards["p3"]),
            "player_4_cards": hand(self.player_cards["p4"]),
        }


_AXIS_ALIGNED = True  # True = bbox axis-aligned (pas de reconstruction d'angle)


def _rect_from_two_corners(p1: tuple[float, float], p2: tuple[float, float]
                           ) -> tuple:
    """2 coins-digit diagonaux → rect carte.

    Mode axis-aligned : on englobe les 2 coins + l'inset (les vrais coins
    physiques sont 14 % au-delà), angle=0. Le TTA rotation du classifieur V2
    gère l'orientation → on évite l'ambiguïté d'ordre des coins (±180°) qui
    polluait la reconstruction d'angle.
    """
    cx = (p1[0] + p2[0]) / 2
    cy = (p1[1] + p2[1]) / 2
    d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    card_w = d / _K
    card_h = 1.5 * card_w
    if _AXIS_ALIGNED:
        # span des 2 coins + extension inset → boîte couvrant la carte entière
        span_x = abs(p2[0] - p1[0]) / (1 - 2 * DIGIT_INSET)
        span_y = abs(p2[1] - p1[1]) / (1 - 2 * DIGIT_INSET)
        bw = max(span_x, card_w) * 1.05
        bh = max(span_y, card_h) * 1.05
        return ((float(cx), float(cy)), (float(bw), float(bh)), 0.0)
    vec_ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    angle = vec_ang - _DIAG_ANGLE
    return ((float(cx), float(cy)), (float(card_w), float(card_h)), float(angle))


def _rect_from_single(corner: tuple[float, float], med_w: float,
                      img_w: int, img_h: int) -> tuple:
    """Coin seul (occlus) : carte de taille médiane, centre estimé en
    décalant le coin vers le centre image (les cartes pointent vers le centre).
    """
    cx0, cy0 = corner
    half_diag = (med_w * _K) / 2 * (1 - 2 * DIGIT_INSET) / (1 - 2 * DIGIT_INSET)
    # direction coin -> centre image
    dx, dy = img_w / 2 - cx0, img_h / 2 - cy0
    n = math.hypot(dx, dy) or 1.0
    off = (med_w * 0.72)  # ~ demi-diagonale digit
    cx = cx0 + dx / n * off
    cy = cy0 + dy / n * off
    return ((float(cx), float(cy)), (float(med_w), float(1.5 * med_w)), 0.0)


def predict_scene_hybrid(image: np.ndarray, image_id: str,
                         detector: CornerDetectorRuntime,
                         corner_clf: CornerClassifierRuntime,
                         full_clf: Classifier,
                         conf_threshold: float = 0.45,
                         diag_min: float = 200.0,
                         diag_max: float = 520.0,
                         use_tta: bool = True) -> ScenePredHybrid:
    h, w = image.shape[:2]
    pred = ScenePredHybrid(image_id=image_id)
    token = detect_active_token(image)

    raw = detector.detect(image)            # [(x,y,score)]  R~0.99
    if not raw:
        if token is not None:
            z = assign_player(token[0], token[1], image.shape)
            pred.active_player = "p1" if z == "center" else z
        return pred

    # Label grossier des coins (juste pour aider le pairing)
    pts = np.array([[x, y] for x, y, _ in raw])
    if len(pts) >= 2:
        dd = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        np.fill_diagonal(dd, np.inf)
        med_nn = float(np.median(dd.min(axis=1)))
        crop_sz = int(np.clip(med_nn * 1.1, 36, 160))
    else:
        crop_sz = 80
    corner_crops = [_crop_around(image, x, y, crop_sz) for x, y, _ in raw]
    clab = corner_clf.classify(corner_crops)  # [(label,conf)]

    # Pairing : même label grossier + distance ∈ [diag_min,diag_max] ;
    # fallback distance seule si pas de partenaire de même label.
    n = len(raw)
    order = sorted(range(n), key=lambda i: -raw[i][2])
    used = [False] * n
    rects: list[tuple] = []
    mid = (diag_min + diag_max) / 2
    widths: list[float] = []
    for i in order:
        if used[i]:
            continue
        xi, yi, _ = raw[i]
        best_j, best_d, best_same = -1, None, False
        for j in order:
            if j == i or used[j]:
                continue
            xj, yj, _ = raw[j]
            d = math.hypot(xi - xj, yi - yj)
            if not (diag_min <= d <= diag_max):
                continue
            same = clab[i][0] == clab[j][0]
            better = (best_d is None
                      or (same and not best_same)
                      or (same == best_same and abs(d - mid) < abs(best_d - mid)))
            if better:
                best_j, best_d, best_same = j, d, same
        if best_j >= 0:
            used[i] = used[best_j] = True
            r = _rect_from_two_corners((xi, yi),
                                       (raw[best_j][0], raw[best_j][1]))
            rects.append(r)
            widths.append(r[1][0])
        else:
            used[i] = True
            rects.append(("single", (xi, yi)))

    med_w = float(np.median(widths)) if widths else max(h, w) * 0.07
    final_rects = []
    for r in rects:
        if r[0] == "single":
            final_rects.append(_rect_from_single(r[1], med_w, w, h))
        else:
            final_rects.append(r)

    # Warp + classification full-card V2
    crops = []
    centers = []
    for rect in final_rects:
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) < 25:
            continue
        warped = _warp_from_rect(image, rect)
        if warped.size == 0:
            continue
        crops.append(warped)
        centers.append((cx, cy))
    if not crops:
        if token is not None:
            z = assign_player(token[0], token[1], image.shape)
            pred.active_player = "p1" if z == "center" else z
        return pred

    if use_tta:
        probs = full_clf.classify_full_tta(crops)
    else:
        probs = full_clf.classify_full(crops)
    labels = [(IDX_TO_CLASS[int(probs[i].argmax())],
               float(probs[i].max())) for i in range(len(crops))]

    # Carte centrale (plus proche du centre image, hors filtre confiance)
    cc = [(centers[i], labels[i]) for i in range(len(crops))
          if assign_player(centers[i][0], centers[i][1], image.shape) == "center"]
    if cc:
        bx = min(cc, key=lambda t: (t[0][0] - w / 2) ** 2 + (t[0][1] - h / 2) ** 2)
        pred.center_card = bx[1][0]

    kept_cards = []
    for (cx, cy), (lab, cf) in zip(centers, labels):
        if cf < conf_threshold:
            continue
        z = assign_player(cx, cy, image.shape)
        kept_cards.append({"cx": cx, "cy": cy, "label": lab})
        if z != "center" and z in pred.player_cards:
            pred.player_cards[z].append(lab)

    # Joueur actif : carte la plus proche du token (robuste, comme V2)
    if token is not None:
        active = assign_token_to_player(token, kept_cards, image.shape)
        if active is not None:
            pred.active_player = active
        else:
            z = assign_player(token[0], token[1], image.shape)
            pred.active_player = "p1" if z == "center" else z
    return pred
