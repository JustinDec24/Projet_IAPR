"""Diagnostic : pour une image synth, montre les détections par niveau FPN +
les raw bbox values pour comprendre où ça casse.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector_dataset import GRID_H, GRID_W, INPUT_H, INPUT_W, LEVELS
from src.detector_inference import CardDetectorRuntime

CHECKPOINT = ROOT / "outputs" / "models" / "detector_pretrain_v2.pt"
runtime = CardDetectorRuntime(CHECKPOINT)

# Pick one synth image with GT
img_id = "synth_000000"
img_path = ROOT / "data" / "synth_images" / f"{img_id}.jpg"
ann_path = ROOT / "data" / "synth_annotations" / f"{img_id}.txt"

image = cv2.imread(str(img_path))
h0, w0 = image.shape[:2]
print(f"Image : {w0}x{h0}")

# GT
gt_lines = ann_path.read_text().strip().split("\n")
print(f"\nGT ({len(gt_lines)} cards) :")
for line in gt_lines:
    parts = line.split()
    cx, cy, w, h, ang = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
    print(f"  cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f} ang={ang:.0f}")

# Forward
img_resized = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
tensor = tensor.to(runtime.device)

with torch.no_grad():
    out = runtime.model(tensor)

print("\n--- Heatmap stats per level ---")
for lvl in LEVELS:
    obj = torch.sigmoid(out[f"obj_{lvl}"])[0, 0].cpu().numpy()
    bbox = out[f"bbox_{lvl}"][0].cpu().numpy()
    gh, gw = obj.shape
    n_above_30 = int((obj > 0.30).sum())
    n_above_50 = int((obj > 0.50).sum())
    n_above_70 = int((obj > 0.70).sum())
    print(f"\n{lvl} (grid {gh}x{gw}):")
    print(f"  obj: min={obj.min():.4f} max={obj.max():.4f} mean={obj.mean():.4f}")
    print(f"  >0.30: {n_above_30}, >0.50: {n_above_50}, >0.70: {n_above_70}")
    if n_above_50 > 0:
        ys, xs = np.where(obj > 0.5)
        for yi, xi in zip(ys[:5], xs[:5]):
            cx_n = bbox[0, yi, xi]
            cy_n = bbox[1, yi, xi]
            w_n = bbox[2, yi, xi]
            h_n = bbox[3, yi, xi]
            print(f"    peak@({yi},{xi}) obj={obj[yi,xi]:.3f} bbox=({cx_n:.3f},{cy_n:.3f},{w_n:.3f},{h_n:.3f})")

print("\n--- Compare GT pixel coordinates ---")
# For first GT card, compute which level it'd be assigned to and the expected pixel
for line in gt_lines[:3]:
    parts = line.split()
    cx, cy, w, h, ang = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
    if w < 0.10:
        lvl = "p3"
    elif w < 0.20:
        lvl = "p4"
    else:
        lvl = "p5"
    gw, gh = GRID_W[lvl], GRID_H[lvl]
    gxi = int(cx * gw)
    gyi = int(cy * gh)
    obj = torch.sigmoid(out[f"obj_{lvl}"])[0, 0].cpu().numpy()
    bbox = out[f"bbox_{lvl}"][0].cpu().numpy()
    pred_at_gt = (bbox[0, gyi, gxi], bbox[1, gyi, gxi], bbox[2, gyi, gxi], bbox[3, gyi, gxi])
    print(f"  GT cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f} -> {lvl} pixel ({gyi},{gxi})")
    print(f"    obj@GT_pixel = {obj[gyi,gxi]:.3f}")
    print(f"    pred bbox@GT_pixel = {pred_at_gt}")
