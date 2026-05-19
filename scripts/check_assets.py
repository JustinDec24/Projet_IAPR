import cv2
import os
from collections import Counter

sizes = Counter()
for f in os.listdir("data/bg_patches"):
    img = cv2.imread("data/bg_patches/" + f)
    if img is not None:
        sizes[img.shape] += 1
print("bg_patches sizes:")
for sz, n in sizes.most_common(5):
    print(f"  {sz}: {n}")

img = cv2.imread("data/card_templates/b_0.png")
print(f"card template size: {img.shape if img is not None else None}")
print(f"total bg patches: {sum(sizes.values())}")
