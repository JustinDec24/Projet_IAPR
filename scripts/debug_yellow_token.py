"""Debug: visualise tous les blobs jaunes sur images fond bruité."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DIR  # noqa: E402

OUT = ROOT / "outputs" / "debug_scenes"

for image_id in ["L1000909", "L1000972"]:
    img = cv2.imread(str(TRAIN_DIR / f"{image_id}.jpg"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv, np.array([20, 130, 130]), np.array([38, 255, 255]))
    yellow_open = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cv2.imwrite(str(OUT / f"{image_id}_yellow_mask.png"), yellow_open)

    contours, _ = cv2.findContours(yellow_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\n=== {image_id} ===")
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        circle_area = np.pi * r * r
        circ = area / circle_area if circle_area > 0 else 0
        blobs.append((int(area), int(cx), int(cy), int(r), round(circ, 2)))
    blobs.sort(reverse=True)
    for b in blobs[:15]:
        print(f"  area={b[0]:>6d} cx={b[1]:>5d} cy={b[2]:>5d} r={b[3]:>3d} circularity={b[4]}")
