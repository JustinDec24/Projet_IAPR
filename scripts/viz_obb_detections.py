"""Visualise les détections OBB du nouveau détecteur sur quelques images train."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector_inference import CardDetectorRuntime

CHECKPOINT = ROOT / "outputs" / "models" / "detector_pretrain.pt"
OUT_DIR = ROOT / "outputs" / "viz_obb_pretrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES = ["L1000770", "L1000843", "L1000909", "L1000794", "L1000857", "L1000967"]
THRESHOLD = 0.10


runtime = CardDetectorRuntime(CHECKPOINT)
print(f"Detector loaded from {CHECKPOINT.name}")

for img_id in IMAGES:
    path = ROOT / "data" / "train_images" / f"{img_id}.jpg"
    if not path.exists():
        path = ROOT / "data" / "test_images" / f"{img_id}.jpg"
    if not path.exists():
        print(f"Not found: {img_id}")
        continue
    image = cv2.imread(str(path))
    detections = runtime.predict_obb(image, obj_threshold=THRESHOLD)
    print(f"{img_id}: {len(detections)} detections (threshold {THRESHOLD})")

    vis = image.copy()
    for rect, score in detections:
        box = cv2.boxPoints(rect).astype(np.intp)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 5)
        (cx, cy), (w, h), angle = rect
        cv2.putText(vis, f"{score:.2f} a={angle:.0f}",
                    (int(cx) - 80, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    out = OUT_DIR / f"{img_id}_obb.jpg"
    vis_small = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
    cv2.imwrite(str(out), vis_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"  -> {out.name}")
