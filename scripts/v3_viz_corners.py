"""Visualise un échantillon de coins extraits pour QA visuel."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
root = ROOT / "data" / "corners_test"
samples, labels = [], []
for lbl in ["g_4", "r_9", "b_5", "y_7", "r_skip", "g_reverse", "b_0", "y_3"]:
    d = root / lbl
    if d.exists():
        fs = list(d.glob("*.png"))
        if fs:
            img = cv2.imread(str(fs[0]))
            img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, lbl, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            samples.append(img)
            labels.append(lbl)
if samples:
    out = ROOT / "reports" / "v3_corner_samples.png"
    cv2.imwrite(str(out), np.hstack(samples))
    print(f"saved {len(samples)} samples: {labels} -> {out}")
