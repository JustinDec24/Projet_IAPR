"""Quick sanity check: load dataset and inspect a few samples."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import IDX_TO_CLASS  # noqa: E402
from src.dataset import UnoTemplateDataset  # noqa: E402

ds = UnoTemplateDataset(
    ROOT / "data" / "card_templates",
    real_crops_dir=ROOT / "data" / "real_crops",
    real_crop_prob=0.5,
    n_samples=100,
)
print(f"Templates: {len(ds.templates)}")
print(f"Real crops total: {ds.n_real_crops_total}")
print(f"Bg patches: {len(ds.bg_patches)}")
print(f"n_classes: {ds.n_classes}")
print(f"has_non_card: {ds.has_non_card}")

# Sample a few items
class_counts = {}
for i in range(50):
    tensor, label = ds[i]
    class_counts[label] = class_counts.get(label, 0) + 1
print(f"\nSample 50 items, classes seen: {len(class_counts)} distinct")
print(f"First sample: shape={tensor.shape} dtype={tensor.dtype} range=[{tensor.min():.2f}, {tensor.max():.2f}]")
