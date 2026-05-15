import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "reports" / "v3_eval.csv")
print("n_gt total :", d.n_gt.sum(), " n_pred total :", d.n_pred.sum())
print("moy n_gt/img :", round(d.n_gt.mean(), 1),
      " moy n_pred/img :", round(d.n_pred.mean(), 1))
print("center_ok :", int(d.center_ok.sum()), "/", len(d))
print("images n_pred==0 :", int((d.n_pred == 0).sum()))
print("images sur-detect (n_pred>1.5*n_gt) :",
      int((d.n_pred > 1.5 * d.n_gt).sum()))
print(d[["image_id", "n_gt", "n_pred", "f1", "center_ok"]].head(10).to_string(index=False))

# Mesure la vraie distance entre 2 coins-digit d'une carte sur image réelle
sys.path.insert(0, str(ROOT))
import cv2
from src.v3.real_corner_extract import corners_from_bbox

img = cv2.imread(str(ROOT / "data" / "train_images" / "L1000770.jpg"))
lines = (ROOT / "data" / "yolo_annotations" / "L1000770.txt").read_text().strip().split("\n")
dists = []
for ln in lines:
    p = ln.split()
    if len(p) < 5:
        continue
    cx, cy, bw, bh = (float(v) for v in p[1:5])
    pts = corners_from_bbox(img, cx, cy, bw, bh)
    if len(pts) == 2:
        dd = np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
        dists.append(dd)
print(f"\nL1000770 image {img.shape[1]}x{img.shape[0]}")
print(f"distance coin-TL<->coin-BR (px) : min={min(dists):.0f} "
      f"max={max(dists):.0f} median={np.median(dists):.0f}")
print("=> diag_min/max actuels = [130, 320] -> COMPLETEMENT hors echelle")
