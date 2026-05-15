"""Compare coins rotation-aware vs axis-aligned naïf sur des vraies images."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.v3.real_corner_extract import corners_from_bbox, _axis_aligned_corners

IMGDIR = ROOT / "data" / "train_images"
ANNDIR = ROOT / "data" / "yolo_annotations"
OUT = ROOT / "reports" / "v3_real_corners"
OUT.mkdir(parents=True, exist_ok=True)

ids = ["L1000770", "L1000771", "L1000843", "L1000909", "L1000967", "L1000975"]
mosaic = []
for sid in ids:
    img = cv2.imread(str(IMGDIR / f"{sid}.jpg"))
    if img is None:
        continue
    h, w = img.shape[:2]
    lines = (ANNDIR / f"{sid}.txt").read_text().strip().split("\n")
    for ln in lines:
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        # naïf = rouge, rotation-aware = vert
        for (x, y) in _axis_aligned_corners(cx, cy, bw, bh, w, h):
            cv2.circle(img, (int(x), int(y)), 16, (0, 0, 255), 3)
        for (x, y) in corners_from_bbox(img, cx, cy, bw, bh):
            cv2.circle(img, (int(x), int(y)), 22, (0, 220, 0), 3)
    cv2.putText(img, f"{sid}  rouge=naif  vert=rotation-aware", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    cv2.imwrite(str(OUT / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    mosaic.append(cv2.resize(img, (640, 426)))

if mosaic:
    rows = [np.hstack(mosaic[i:i + 3]) for i in range(0, len(mosaic), 3)]
    while rows and rows[-1].shape[1] < rows[0].shape[1]:
        pad = rows[0].shape[1] - rows[-1].shape[1]
        rows[-1] = np.hstack([rows[-1], np.zeros((426, pad, 3), np.uint8)])
    cv2.imwrite(str(OUT / "mosaic.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"-> {OUT}/mosaic.jpg ({len(mosaic)} images)")
