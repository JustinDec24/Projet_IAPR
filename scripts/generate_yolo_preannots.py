"""Pré-annote chaque train_image au format YOLO (1 fichier .txt par image)
à partir de la détection actuelle. Format YOLO :

    class_idx cx_norm cy_norm w_norm h_norm

avec une seule classe 'card' (class_idx=0). L'utilisateur reviewse ensuite ces
pré-annotations dans LabelImg : ajoute les cartes manquées, corrige les bboxes
imparfaites, supprime les faux positifs.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DIR  # noqa: E402
from src.detection import detect_active_token, detect_cards_in_scene  # noqa: E402

OUT_DIR = ROOT / "data" / "yolo_annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fichier classes.txt requis par LabelImg en mode YOLO
(OUT_DIR / "classes.txt").write_text("card\n")


def rect_to_aabb(rect: tuple, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convertit un rotated rect OpenCV en bbox axis-aligned (x_min, y_min, x_max, y_max)."""
    box = cv2.boxPoints(rect).astype(np.intp)
    x_min = int(max(0, box[:, 0].min()))
    y_min = int(max(0, box[:, 1].min()))
    x_max = int(min(img_w - 1, box[:, 0].max()))
    y_max = int(min(img_h - 1, box[:, 1].max()))
    return x_min, y_min, x_max, y_max


paths = sorted(TRAIN_DIR.glob("*.jpg"))
n_total = 0
for path in tqdm(paths, desc="Pre-annot"):
    image = cv2.imread(str(path))
    if image is None:
        continue
    h, w = image.shape[:2]
    token = detect_active_token(image)
    cards = detect_cards_in_scene(image, exclude_xy=token)

    lines = []
    for c in cards:
        x_min, y_min, x_max, y_max = rect_to_aabb(c["rect"], w, h)
        if x_max <= x_min or y_max <= y_min:
            continue
        cx = (x_min + x_max) / 2.0 / w
        cy = (y_min + y_max) / 2.0 / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        n_total += 1

    (OUT_DIR / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

print(f"\n=> {n_total} bboxes pré-annotées sur {len(paths)} images")
print(f"   -> {OUT_DIR}")
print(f"\n   Classes : 'card' (class_idx=0)")
print(f"   Format : YOLO normalisé (cx, cy, w, h)")
