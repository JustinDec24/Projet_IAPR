import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
t = cv2.imread(str(ROOT / "data" / "card_templates" / "g_4.png"))
print("template g_4 shape:", t.shape, "dtype", t.dtype)
print("corner TL:", t[0, 0], "corner BR:", t[-1, -1], "center:", t[150, 100])
n = len(list((ROOT / "data" / "card_templates").glob("*.png")))
print("n templates:", n)
# check train image size
img = cv2.imread(str(next((ROOT / "data" / "train_images").glob("*.jpg"))))
print("train image shape:", img.shape)
