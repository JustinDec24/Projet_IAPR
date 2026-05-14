"""Vérifie le compte de params des nouvelles architectures."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from src.detector import CardDetector, count_parameters as cd_count
from src.model import UnoCNN

# Classifier teacher (11.2M)
m_big = UnoCNN(num_classes=54, channels=(64, 128, 256, 512))
n_big = sum(p.numel() for p in m_big.parameters() if p.requires_grad)
print(f"Classifier teacher [64,128,256,512] : {n_big / 1e6:.2f}M params")

# Classifier student (target ~6M)
m_small = UnoCNN(num_classes=54, channels=(48, 96, 192, 384))
n_small = sum(p.numel() for p in m_small.parameters() if p.requires_grad)
print(f"Classifier student [48,96,192,384]  : {n_small / 1e6:.2f}M params")

# Détecteur (target ~5M)
d = CardDetector(channels=(40, 80, 160, 320), fpn_channels=128)
n_d = cd_count(d)
print(f"Detector [40,80,160,320] FPN=128    : {n_d / 1e6:.2f}M params")

# Forward test
x = torch.randn(2, 3, 352, 512)
out = d(x)
for k, v in out.items():
    print(f"  {k}: {tuple(v.shape)}")

print(f"\nTotal student+detector : {(n_small + n_d) / 1e6:.2f}M / 12M")
print(f"Total teacher+detector : {(n_big + n_d) / 1e6:.2f}M / 12M")
