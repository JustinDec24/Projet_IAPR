"""Test le détecteur Phase 1 sur des images synthétiques (vérif sanity)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector_inference import CardDetectorRuntime

CHECKPOINT = ROOT / "outputs" / "models" / "detector_pretrain.pt"
OUT_DIR = ROOT / "outputs" / "viz_obb_synth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

runtime = CardDetectorRuntime(CHECKPOINT)

# Test on 4 synth images
for i in [0, 100, 1000, 3000]:
    path = ROOT / "data" / "synth_images" / f"synth_{i:06d}.jpg"
    if not path.exists():
        continue
    image = cv2.imread(str(path))
    detections = runtime.predict_obb(image, obj_threshold=0.3)
    print(f"synth_{i:06d}: {len(detections)} detections")

    vis = image.copy()
    for rect, score in detections:
        box = cv2.boxPoints(rect).astype(np.intp)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 3)
        (cx, cy), (w, h), angle = rect
        cv2.putText(vis, f"{score:.2f} a={angle:.0f}",
                    (int(cx) - 40, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out = OUT_DIR / f"synth_{i:06d}_obb.jpg"
    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
