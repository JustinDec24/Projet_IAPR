"""Find the darkest regions in the image - should reveal where the token is."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DIR  # noqa: E402

OUT = ROOT / "outputs" / "debug_scenes"
OUT.mkdir(parents=True, exist_ok=True)


for image_id in ["L1000771", "L1000775"]:
    img = cv2.imread(str(TRAIN_DIR / f"{image_id}.jpg"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    val = hsv[..., 2]
    # Smooth then find the very darkest blobs
    val_smooth = cv2.medianBlur(val, 7)
    print(f"\n=== {image_id} ===")
    print(f"  Image V: min={val_smooth.min()}  max={val_smooth.max()}  "
          f"mean={val_smooth.mean():.1f}")
    for v_th in (50, 80, 110, 140, 170):
        n = int((val_smooth < v_th).sum())
        print(f"  V<{v_th}: {n:,} pixels ({n / val.size:.2%})")

    # Save the very-dark mask (V < median - 50)
    threshold = max(20, int(np.median(val) - 80))
    print(f"  Threshold used : V < {threshold}")
    mask = (val_smooth < threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cv2.imwrite(str(OUT / f"{image_id}_v_dark.png"), mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        if min(w, h) < 10:
            continue
        ar = max(w, h) / max(min(w, h), 1)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        sol = area / hull_area if hull_area > 0 else 0
        blobs.append((int(area), int(cx), int(cy), int(w), int(h), round(ar, 2), round(sol, 2)))
    blobs.sort(reverse=True)
    print(f"  Top dark blobs (area, cx, cy, w, h, ar, solidity):")
    for b in blobs[:10]:
        print(f"    {b}")
