"""Génère un dataset synthétique de scènes UNO depuis assets internes.

Usage :
    python scripts/generate_synth_dataset.py --n 5000 --preview 8

Sortie :
    data/synth_images/<id>.jpg
    data/synth_annotations/<id>.txt    (YOLO étendu : class cx cy w h angle_deg)
    outputs/synth_preview.jpg          (mosaïque de 8 exemples avec OBB dessinées)
"""
import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synth_generator import (  # noqa: E402
    SCENE_H,
    SCENE_W,
    _load_assets,
    generate_scene,
    save_annotations,
)

DATA_DIR = ROOT / "data"
CARD_DIR = DATA_DIR / "card_templates"
BG_DIR = DATA_DIR / "bg_patches"
OUT_IMG = DATA_DIR / "synth_images"
OUT_ANN = DATA_DIR / "synth_annotations"
PREVIEW_PATH = ROOT / "outputs" / "synth_preview.jpg"


def draw_obb(image: np.ndarray, annotations: list[tuple]) -> np.ndarray:
    """Dessine les rotated bboxes pour la viz."""
    vis = image.copy()
    for (cx, cy, w, h, angle, name) in annotations:
        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect).astype(np.intp)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
        cv2.putText(vis, name, (int(cx) - 30, int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="number of synthetic images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview", type=int, default=8,
                   help="number of preview thumbnails to save with OBB drawn")
    args = p.parse_args()

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_ANN.mkdir(parents=True, exist_ok=True)

    print("Loading assets...")
    templates, bg_paths = _load_assets(CARD_DIR, BG_DIR)
    print(f"  templates: {len(templates)}  bg patches: {len(bg_paths)}")
    if not templates or not bg_paths:
        print("ERROR: missing assets")
        return 1

    rng = random.Random(args.seed)

    t0 = time.time()
    preview_imgs = []
    for i in tqdm(range(args.n), desc="Synth"):
        seed = args.seed + i
        rng = random.Random(seed)
        np.random.seed(seed)
        try:
            img, annotations = generate_scene(templates, bg_paths, rng)
        except Exception as e:
            print(f"Skipped {i}: {e}")
            continue
        img_id = f"synth_{i:06d}"
        cv2.imwrite(str(OUT_IMG / f"{img_id}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        save_annotations(annotations, OUT_ANN / f"{img_id}.txt",
                         img_w=SCENE_W, img_h=SCENE_H)
        if len(preview_imgs) < args.preview:
            preview_imgs.append(draw_obb(img, annotations))

    dt = time.time() - t0
    print(f"\nGenerated {args.n} images in {dt:.1f}s ({args.n / dt:.1f} img/s)")

    # Save preview mosaic
    if preview_imgs:
        rows = 2
        cols = (len(preview_imgs) + rows - 1) // rows
        # Resize for mosaic
        th, tw = 400, 600
        thumbs = [cv2.resize(p, (tw, th)) for p in preview_imgs]
        while len(thumbs) < rows * cols:
            thumbs.append(np.zeros_like(thumbs[0]))
        mosaic_rows = []
        for r in range(rows):
            row = np.hstack(thumbs[r * cols:(r + 1) * cols])
            mosaic_rows.append(row)
        mosaic = np.vstack(mosaic_rows)
        PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(PREVIEW_PATH), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"Preview saved : {PREVIEW_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
