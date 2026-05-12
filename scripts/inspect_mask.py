"""Sauve le mask actuel + signaux candidats pour comprendre les échecs sur fond bruité."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DIR  # noqa: E402
from src.detection import _enclosed_holes  # noqa: E402
from src.templates import card_mask_from_image  # noqa: E402

OUT = ROOT / "outputs" / "debug_scenes"
OUT.mkdir(parents=True, exist_ok=True)


for image_id in ["L1000770", "L1000909", "L1000972"]:
    img = cv2.imread(str(TRAIN_DIR / f"{image_id}.jpg"))
    sat_mask = card_mask_from_image(img)
    holes_raw = _enclosed_holes(sat_mask)
    holes = cv2.morphologyEx(holes_raw, cv2.MORPH_CLOSE, np.ones((81, 81), np.uint8))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cv2.imwrite(str(OUT / f"{image_id}_mask_sat.png"), sat_mask)
    cv2.imwrite(str(OUT / f"{image_id}_mask_holes_raw.png"), holes_raw)
    cv2.imwrite(str(OUT / f"{image_id}_mask_holes_closed.png"), holes)

    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    info = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        ar = max(w, h) / max(min(w, h), 1)
        hull = cv2.contourArea(cv2.convexHull(c))
        sol = area / hull if hull > 0 else 0
        info.append((int(area), round(ar, 2), round(sol, 2)))
    info.sort(reverse=True)
    print(f"{image_id}: {len(info)} candidate ovals (area>=5k):")
    for a, ar, s in info[:15]:
        print(f"    area={a:>6d} aspect={ar} solidity={s}")
