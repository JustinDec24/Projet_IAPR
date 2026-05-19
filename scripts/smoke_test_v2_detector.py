"""Smoke test du détecteur 5M axis-aligned + dataset multi-source."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import CardDetector, count_parameters
from src.detector_dataset import CardDetectionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Param counts pour 3 tailles
for ch in [(16, 32, 64, 96), (40, 80, 160, 320), (48, 96, 192, 384)]:
    m = CardDetector(channels=ch)
    n = count_parameters(m)
    print(f"channels={ch}: {n/1e6:.2f}M params")

# Test dataset multi-source
ds = CardDetectionDataset(
    [ROOT / "data" / "synth_images", ROOT / "data" / "train_images"],
    [ROOT / "data" / "synth_annotations", ROOT / "data" / "yolo_annotations"],
    weights=[0.8, 0.2],
    augment=True, n_samples=10, seed=0,
)
print(f"\nDataset OK : {[len(s[0]) for s in ds.sources]} files per source, weights={ds.weights}")

# Forward
model = CardDetector(channels=(40, 80, 160, 320)).to(device)
sample = ds[0]
batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
out = model(batch["image"])
print(f"\nForward OK:")
for k, v in out.items():
    print(f"  {k}: {tuple(v.shape)}")

# Loss
from scripts.train_detector import focal_loss, bbox_loss
loss_obj = focal_loss(out["obj"], batch["obj_target"])
loss_bb = bbox_loss(out["bbox"], batch["bbox_target"], batch["bbox_mask"])
print(f"\nlosses : obj={loss_obj.item():.4f} bb={loss_bb.item():.4f}")
(loss_obj + 5 * loss_bb).backward()
print("Backward OK")
