"""Debug: visualise tous les candidats noir token et pourquoi ils sont rejetés."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DIR  # noqa: E402

OUT = ROOT / "outputs" / "debug_scenes"


for image_id in ["L1000771", "L1000770", "L1000775"]:
    img = cv2.imread(str(TRAIN_DIR / f"{image_id}.jpg"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    black = ((hsv[..., 1] < 80) & (hsv[..., 2] < 80)).astype(np.uint8) * 255
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cv2.imwrite(str(OUT / f"{image_id}_black_mask.png"), black)

    contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\n=== {image_id} ===")
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        if min(w, h) < 5:
            continue
        ar = max(w, h) / max(min(w, h), 1)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        sol = area / hull_area if hull_area > 0 else 0
        in_filter = (1000 < area < 40000 and min(w, h) >= 15 and ar < 4.0 and sol > 0.8)
        print(f"  area={int(area):>6d} cx={cx:7.0f} cy={cy:7.0f} "
              f"wh=({w:.0f},{h:.0f}) ar={ar:.2f} sol={sol:.2f} {'KEEP' if in_filter else 'REJECT'}")
