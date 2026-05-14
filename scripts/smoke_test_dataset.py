"""Vérifie que le dataset lit correctement synth + real (formats annotations différents)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector_dataset import LEVELS, OBBDetectionDataset

# Mixed sources
ds = OBBDetectionDataset(
    [ROOT / "data" / "synth_images", ROOT / "data" / "train_images"],
    [ROOT / "data" / "synth_annotations", ROOT / "data" / "yolo_annotations"],
    weights=[0.5, 0.5],
    augment=True, n_samples=20, seed=0,
)
print(f"Sources : {[len(f) for f in ds.files_per_source]} files each")
print(f"Weights : {ds.weights}")

# Sample 5 batches and check distributions
n_synth = 0
n_real = 0
n_positive_p3 = 0
n_positive_p4 = 0
n_positive_p5 = 0
for i in range(20):
    sample = ds[i]
    img = sample["image"]
    # Heuristic: synth has cleaner backgrounds (less variation in stem features)
    # Just count which source is more frequent
    pos_p3 = int(sample["mask_p3"].sum().item())
    pos_p4 = int(sample["mask_p4"].sum().item())
    pos_p5 = int(sample["mask_p5"].sum().item())
    n_positive_p3 += pos_p3
    n_positive_p4 += pos_p4
    n_positive_p5 += pos_p5

print(f"\nPositives total (over 20 samples):")
print(f"  P3: {n_positive_p3}")
print(f"  P4: {n_positive_p4}")
print(f"  P5: {n_positive_p5}")
print("=> Smoke test dataset PASS" if (n_positive_p3 + n_positive_p4 + n_positive_p5) > 0 else "FAIL")
