"""Probe detector outputs to debug 0-score eval."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from src.detector_inference import CardDetectorRuntime
from src.detector_dataset import INPUT_W, INPUT_H

runtime = CardDetectorRuntime("outputs/models/detector.pt", device="cuda")
# Use cv2.imread (BGR) to match the eval pipeline
img_np = cv2.imread("data/train_images/L1000770.jpg")
print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")

for thr in [0.01, 0.05, 0.1, 0.2, 0.35, 0.5]:
    bboxes = runtime.predict_bboxes(img_np, obj_threshold=thr, iou_threshold=0.45)
    print(f"thr={thr}: {len(bboxes)} bboxes")
    if bboxes and thr == 0.01:
        scores = [b[4] for b in bboxes[:10]]
        print(f"  top10 scores: {scores}")

# Inspect raw heatmap directly
img_resized = cv2.resize(img_np, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
tensor = tensor.to(runtime.device)
with torch.no_grad():
    out = runtime.model(tensor)
obj = torch.sigmoid(out["obj"])[0, 0].cpu().numpy()
print(f"\nRaw obj heatmap shape: {obj.shape}")
print(f"  min={obj.min():.6f} max={obj.max():.6f} mean={obj.mean():.6f}")
print(f"  >0.01: {(obj > 0.01).sum()} / {obj.size}")
print(f"  >0.1:  {(obj > 0.1).sum()} / {obj.size}")
print(f"  >0.35: {(obj > 0.35).sum()} / {obj.size}")
print(f"\nbbox preds (mean of cx,cy,w,h channels):")
bbox = out["bbox"][0].cpu().numpy()
for i, name in enumerate(["cx", "cy", "w", "h"]):
    print(f"  {name}: mean={bbox[i].mean():.4f}, min={bbox[i].min():.4f}, max={bbox[i].max():.4f}")
