"""Smoke test du détecteur."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import CardDetector, count_parameters  # noqa: E402

m = CardDetector()
n = count_parameters(m)
print(f"Params : {n:,} ({n / 1e6:.2f} M)")

x = torch.randn(2, 3, 512, 352)
y = m(x)
print(f'obj  : {tuple(y["obj"].shape)}')
print(f'bbox : {tuple(y["bbox"].shape)}')
print(f"Input  : (B, 3, 512, 352)")
print(f"Output downsample : 16× → grid 32×22")
