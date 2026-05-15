import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.v3.corner_dataset import CornerDataset
from src.v3.corner_detector import CornerDetector, count_parameters
from scripts.train_corner_detector import focal_loss, off_loss

D = ROOT / "data"
ds = CornerDataset(D / "synthetic" / "images", D / "synthetic" / "labels",
                   D / "train_images", D / "yolo_annotations",
                   real_ratio=0.3, augment=True, n_samples=8, seed=42)
print("dataset len(virtual):", len(ds))
nsynth = nreal = 0
for i in range(8):
    s = ds[i]
    pos = int(s["mask"].sum())
    print(f"  sample {i}: img {tuple(s['image'].shape)} hm {tuple(s['heatmap'].shape)} pos={pos}")

m = CornerDetector(channels=(40, 80, 160, 320))
print(f"params {count_parameters(m)/1e6:.2f}M")
b = {k: s.unsqueeze(0) for k, s in ds[0].items()}
o = m(b["image"])
lh = focal_loss(o["heatmap"], b["heatmap"])
lo = off_loss(o["offset"], b["offset"], b["mask"])
print(f"loss hm={lh.item():.4f} off={lo.item():.4f}")
(lh + lo).backward()
print("backward OK -> smoke PASS")
