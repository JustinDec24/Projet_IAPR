"""Auto-labeling itératif V3 : localise les cartes avec le corner-detector
(R=0.99, robuste occlusion) → matche au GT par couleur+zone → crops réels
labellisés plus nombreux et mieux localisés que real_crops d'origine (237).

Sortie : data/real_crops_v3/<label>/<id>.png  (warps 200×300)
"""
from __future__ import annotations

import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.detection import assign_player  # noqa: E402
from src.templates import _warp_from_rect, dominant_color  # noqa: E402
from src.v3.corner_detector_runtime import CornerDetectorRuntime  # noqa: E402
from src.v3.predict_scene_hybrid import _rect_from_two_corners  # noqa: E402
from scripts.autolabel_train import color_of, match_slot, parse_hand  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("autolabel_v3")

OUT = ROOT / "data" / "real_crops_v3"
DIAG_MIN, DIAG_MAX = 200.0, 520.0


def localize_cards(image: np.ndarray, det: CornerDetectorRuntime) -> list[dict]:
    raw = det.detect(image)
    if not raw:
        return []
    n = len(raw)
    order = sorted(range(n), key=lambda i: -raw[i][2])
    used = [False] * n
    mid = (DIAG_MIN + DIAG_MAX) / 2
    rects: list[tuple] = []
    for i in order:
        if used[i]:
            continue
        xi, yi, _ = raw[i]
        best_j, best_d = -1, None
        for j in order:
            if j == i or used[j]:
                continue
            xj, yj, _ = raw[j]
            d = math.hypot(xi - xj, yi - yj)
            if DIAG_MIN <= d <= DIAG_MAX and (best_d is None
                                              or abs(d - mid) < abs(best_d - mid)):
                best_j, best_d = j, d
        if best_j >= 0:
            used[i] = used[best_j] = True
            rects.append(_rect_from_two_corners(
                (xi, yi), (raw[best_j][0], raw[best_j][1])))
        else:
            used[i] = True  # coin seul ignoré (pas de rect fiable)
    cards = []
    for rect in rects:
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) < 25:
            continue
        warped = _warp_from_rect(image, rect)
        if warped.size == 0:
            continue
        cards.append({"cx": cx, "cy": cy, "warped": warped,
                      "color": dominant_color(warped)})
    return cards


def main() -> int:
    det = CornerDetectorRuntime(ROOT / "checkpoints" / "v3_corner_detector.pth")
    df = pd.read_csv(TRAIN_CSV)
    OUT.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    by_class: dict[str, int] = defaultdict(int)
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="autolabel_v3"):
        p = TRAIN_DIR / f"{row.image_id}.jpg"
        if not p.exists():
            continue
        image = cv2.imread(str(p))
        cards = localize_cards(image, det)
        if not cards:
            continue
        by_slot: dict[str, list[dict]] = defaultdict(list)
        for c in cards:
            by_slot[assign_player(c["cx"], c["cy"], image.shape)].append(c)
        gt_per_slot = {
            "p1": parse_hand(row.player_1_cards),
            "p2": parse_hand(row.player_2_cards),
            "p3": parse_hand(row.player_3_cards),
            "p4": parse_hand(row.player_4_cards),
            "center": ([row.center_card]
                       if row.center_card and row.center_card != "EMPTY" else []),
        }
        for slot, scs in by_slot.items():
            gtl = gt_per_slot.get(slot, [])
            if not gtl:
                continue
            for ci, lab in match_slot(scs, gtl).items():
                d = OUT / lab
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{row.image_id}_{slot}_{ci:02d}.png"),
                            scs[ci]["warped"])
                n_saved += 1
                by_class[lab] += 1

    logger.info("V3 auto-label : %d crops, %d classes", n_saved, len(by_class))
    print(f"\nreal_crops_v3 : {n_saved} crops (origine real_crops = 237)")
    print(f"classes couvertes : {len(by_class)}/54")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
