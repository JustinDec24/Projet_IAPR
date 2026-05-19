"""Smoke test : 1 forward + 1 backward pour vérifier que train converge."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import CardDetector, count_parameters
from src.detector_dataset import LEVELS, OBBDetectionDataset
from scripts.train_detector import compute_loss

DATA_DIR = ROOT / "data"
SYNTH_IMG_DIR = DATA_DIR / "synth_images"
SYNTH_ANN_DIR = DATA_DIR / "synth_annotations"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# Vérifier qu'on a quelques synth images
ds = OBBDetectionDataset([SYNTH_IMG_DIR], [SYNTH_ANN_DIR],
                          augment=True, n_samples=4, seed=0)
print(f"Dataset OK : {len(ds.files_per_source[0])} files in source")

# Build model
model = CardDetector().to(device)
n = count_parameters(model)
print(f"Detector : {n / 1e6:.2f}M params")

# Get a batch
batch = ds[0]
batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
print("Batch shapes :")
for k, v in batch.items():
    print(f"  {k}: {tuple(v.shape)}")

# Forward
out = model(batch["image"])
print("\nForward OK. Outputs :")
for k, v in out.items():
    print(f"  {k}: {tuple(v.shape)}")

# Loss
loss, breakdown = compute_loss(out, batch)
print(f"\nLoss = {loss.item():.4f}")
for k, v in breakdown.items():
    print(f"  {k}: {v:.4f}")

# Backward
loss.backward()
print("\nBackward OK. Grad sample :")
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"  {name[:50]}: grad_norm={p.grad.norm().item():.6f}")
        break

print("\n=> Smoke test PASS")
