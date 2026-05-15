"""Overlay des annotations coins sur les scènes synthétiques pour QA."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "synthetic" / "images"
LBL = ROOT / "data" / "synthetic" / "labels"
OUT = ROOT / "reports" / "v3_synthetic_qa"
OUT.mkdir(parents=True, exist_ok=True)

ids = [f"scene_{i:05d}" for i in [0, 1, 2, 3, 4]]
mosaic = []
for sid in ids:
    img = cv2.imread(str(IMG / f"{sid}.jpg"))
    if img is None:
        continue
    txt = (LBL / f"{sid}.txt").read_text().strip()
    n_vis = n_occ = 0
    for line in txt.split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        lbl, x, y, o = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        cs = int(parts[4]) if len(parts) > 4 else 50
        half = cs // 2
        color = (0, 140, 255) if o else (0, 220, 0)  # orange si occlus, vert sinon
        n_occ += o
        n_vis += 1
        # Boîte de crop réelle (ce que verra le corner-classifier)
        cv2.rectangle(img, (x - half, y - half), (x + half, y + half), color, 2)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(img, lbl, (x - half, y - half - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    cv2.putText(img, f"{sid}: {n_vis} coins ({n_occ} occlus)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / f"{sid}_qa.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    mosaic.append(cv2.resize(img, (512, 384)))

if mosaic:
    top = np.hstack(mosaic[:3])
    pad = 512 * 3 - 512 * len(mosaic[3:])
    bot_imgs = mosaic[3:] + [np.zeros((384, 512, 3), np.uint8)] * (3 - len(mosaic[3:]))
    bot = np.hstack(bot_imgs)
    cv2.imwrite(str(OUT / "mosaic.jpg"), np.vstack([top, bot]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"QA: {len(mosaic)} scenes -> {OUT}/mosaic.jpg")
