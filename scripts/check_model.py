"""Vérifie taille et forward pass du modèle."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import UnoCNN, count_parameters  # noqa: E402

m = UnoCNN()
n = count_parameters(m)
print(f"Params       : {n:,} ({n / 1e6:.2f} M)")
print(f"Sous 12M     : {n < 12_000_000}")
x = torch.randn(2, 3, 144, 96)
y = m(x)
print(f"Forward pass : in {tuple(x.shape)} -> out {tuple(y.shape)}")
