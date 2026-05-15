import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.v3.corner_detector import CornerDetector, count_parameters

for ch in [(32, 64, 128, 256), (40, 80, 160, 320), (24, 48, 96, 192)]:
    m = CornerDetector(channels=ch)
    n = count_parameters(m)
    print(f"channels={ch}: {n/1e6:.2f}M params")

m = CornerDetector()
x = torch.randn(2, 3, 384, 512)
o = m(x)
print("input", tuple(x.shape))
for k, v in o.items():
    print(f"  {k}: {tuple(v.shape)}")
