"""Sample HSV values of the black token to calibrate detection thresholds."""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DIR  # noqa: E402

# Token approximate positions (visually identified)
SAMPLES = {
    "L1000770": (1012, 364, 80, 130),  # cx, cy, half_w, half_h (well-detected)
    "L1000771": (3690, 2300, 50, 80),  # bottom-right of image
    "L1000775": (180, 1000, 100, 150), # left
}

for image_id, (cx, cy, hw, hh) in SAMPLES.items():
    img = cv2.imread(str(TRAIN_DIR / f"{image_id}.jpg"))
    h, w = img.shape[:2]
    x0, x1 = max(0, cx - hw), min(w, cx + hw)
    y0, y1 = max(0, cy - hh), min(h, cy + hh)
    roi = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    print(f"\n=== {image_id} ROI [{x0}:{x1}, {y0}:{y1}] ===")
    print(f"  V mean={hsv[..., 2].mean():.1f}  median={int(hsv[..., 2].mean())}  "
          f"min={int(hsv[..., 2].min())}  max={int(hsv[..., 2].max())}")
    print(f"  S mean={hsv[..., 1].mean():.1f}  min={int(hsv[..., 1].min())}  "
          f"max={int(hsv[..., 1].max())}")
    # Quel pourcentage des pixels passe V<80 / V<100 / V<130 ?
    for v_th in (80, 100, 130, 150):
        frac = (hsv[..., 2] < v_th).mean()
        print(f"  V<{v_th}: {frac:.2%} of pixels")
