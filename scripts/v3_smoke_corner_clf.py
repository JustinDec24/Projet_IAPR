import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.v3.corner_classifier import CornerClassifier, count_parameters
from src.v3.corner_classifier_dataset import CornerClassifierDataset

ds = CornerClassifierDataset(ROOT, augment=True, n_samples=20, seed=42)
print("classes couvertes:", len(ds.classes), "/ 54")
src_counts = Counter()
for v in ds.by_class.values():
    for kind, _ in v:
        src_counts[kind] += 1
print("items par source:", dict(src_counts))
xs, ys = [], []
for i in range(12):
    x, y = ds[i]
    xs.append(x); ys.append(y)
b = torch.stack(xs)
print("batch", tuple(b.shape), "labels sample", ys[:6])
m = CornerClassifier(channels=(48, 96, 160, 224))
print(f"params {count_parameters(m)/1e6:.2f}M")
out = m(b)
loss = torch.nn.functional.cross_entropy(out, torch.tensor(ys))
loss.backward()
print(f"out {tuple(out.shape)} loss {loss.item():.3f} -> smoke PASS")
