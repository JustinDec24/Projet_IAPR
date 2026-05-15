import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.v3.corner_classifier import CornerClassifier, count_parameters

for ch in [(32, 64, 96, 128), (40, 80, 128, 160), (48, 96, 160, 224)]:
    m = CornerClassifier(channels=ch)
    print(ch, f"{count_parameters(m) / 1e6:.2f}M")
m = CornerClassifier()
x = torch.randn(4, 3, 80, 80)
print("out", tuple(m(x).shape))
