"""Test la nouvelle détection classique sur quelques images variées."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection_corners import detect_cards_classical  # noqa: E402
from src.detection import assign_player  # noqa: E402

PLAYER_COLORS = {
    "p1": (220, 100, 80), "p2": (80, 180, 80),
    "p3": (60, 140, 230), "p4": (200, 80, 200),
    "center": (60, 220, 220),
}

OUT = ROOT / "outputs" / "test_corners"
OUT.mkdir(parents=True, exist_ok=True)

IMAGES = ["L1000770", "L1000843", "L1000909", "L1000794", "L1000773"]


def visualize(image: np.ndarray, cards: list[dict]) -> np.ndarray:
    vis = image.copy()
    for c in cards:
        player = assign_player(c["cx"], c["cy"], image.shape)
        color = PLAYER_COLORS[player]
        box = cv2.boxPoints(c["rect"]).astype(np.intp)
        cv2.drawContours(vis, [box], 0, color, thickness=8)
        cv2.putText(vis, c.get("color", "?"),
                    (int(c["cx"]) - 50, int(c["cy"]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    return vis


for img_id in IMAGES:
    path = ROOT / "data" / "train_images" / f"{img_id}.jpg"
    if not path.exists():
        path = ROOT / "data" / "test_images" / f"{img_id}.jpg"
    if not path.exists():
        print(f"Not found: {img_id}")
        continue
    image = cv2.imread(str(path))
    cards = detect_cards_classical(image)
    print(f"{img_id}: {len(cards)} cards detected")
    vis = visualize(image, cards)
    out = OUT / f"{img_id}_classical.jpg"
    vis_small = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
    cv2.imwrite(str(out), vis_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"  -> {out}")
