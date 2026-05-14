"""Mesure visuellement la taille d'un coin de carte UNO dans une image train.

Affiche le mask des blobs blancs petits-moyens dans l'image pour calibrer les
seuils de la détection ancrée sur les coins.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "probe_corners"
OUT.mkdir(parents=True, exist_ok=True)

# Test on a mix of bg conditions
IMAGES = [
    "L1000770",   # clean white bg, fully visible cards
    "L1000843",   # clean white bg, stacked cards
    "L1000909",   # foliage bg, hard case
]


def analyze(image_id: str) -> None:
    path = ROOT / "data" / "train_images" / f"{image_id}.jpg"
    image = cv2.imread(str(path))
    if image is None:
        print(f"Not found: {path}")
        return

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White pixels: high V, low S
    white_mask = ((hsv[..., 2] > 180) & (hsv[..., 1] < 50)).astype(np.uint8) * 255
    # Close small gaps
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask)
    print(f"\n=== {image_id} ({w}x{h}) ===")
    print(f"  total blanks blobs: {n - 1}")

    # Bucket by area
    buckets = {
        "tiny (<400)": 0,
        "corner-like (400-5k)": 0,
        "medium (5k-20k)": 0,
        "oval-like (20k-100k)": 0,
        "huge (>100k)": 0,
    }
    corner_examples = []  # for visualization
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 400:
            buckets["tiny (<400)"] += 1
        elif area < 5000:
            buckets["corner-like (400-5k)"] += 1
            if 0.5 < bw / max(bh, 1) < 2.0 and len(corner_examples) < 30:
                corner_examples.append((x, y, bw, bh, area))
        elif area < 20000:
            buckets["medium (5k-20k)"] += 1
        elif area < 100000:
            buckets["oval-like (20k-100k)"] += 1
        else:
            buckets["huge (>100k)"] += 1
    for k, v in buckets.items():
        print(f"  {k}: {v}")

    # Visualize the white mask + highlight corner candidates
    vis = image.copy()
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[white_mask > 0] = (255, 255, 255)
    vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
    for (x, y, bw, bh, area) in corner_examples:
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.putText(vis, f"{int(area)}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = OUT / f"{image_id}_corners.jpg"
    # Downscale for output size
    vis_small = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
    cv2.imwrite(str(out_path), vis_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"  -> {out_path}")


for img_id in IMAGES:
    analyze(img_id)
