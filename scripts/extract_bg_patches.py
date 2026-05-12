"""Extrait des patches de fond depuis train_images/ (zones sans cartes).

Pour chaque image train, on masque les cartes via card_mask_from_image (puis
dilatation pour inclure les bordures), puis on sample des patches rectangulaires
de taille « carte » (~370×590 px) entièrement contenus dans la zone hors-carte.

Les patches sont sauvés dans data/bg_patches/<image_id>_<i>.png et serviront
de matériel d'entraînement pour la classe « non_card » (pour apprendre au CNN
à dire « ce crop n'est pas une carte »).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DIR  # noqa: E402
from src.detection import detect_cards_in_scene  # noqa: E402

OUT = ROOT / "data" / "bg_patches"
OUT.mkdir(parents=True, exist_ok=True)

PATCH_W, PATCH_H = 370, 590
N_PATCHES_PER_IMAGE = 15
MIN_BG_FRACTION = 0.92


def extract_patches(image: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Échantillonne des patches dans les zones hors-carte.

    On masque les zones cartes via `detect_cards_in_scene` (qui marche pour les
    deux types de fond), puis on dilate généreusement, et on sample des patches
    PATCH_W×PATCH_H qui sont au moins MIN_BG_FRACTION en zone non-carte.
    """
    h, w = image.shape[:2]
    cards = detect_cards_in_scene(image)
    card_mask = np.zeros((h, w), dtype=np.uint8)
    for c in cards:
        box = cv2.boxPoints(c["rect"]).astype(np.intp)
        cv2.drawContours(card_mask, [box], 0, 255, thickness=cv2.FILLED)
    card_mask = cv2.dilate(card_mask, np.ones((50, 50), np.uint8))
    bg_mask = cv2.bitwise_not(card_mask)

    patches: list[np.ndarray] = []
    attempts = 0
    while len(patches) < N_PATCHES_PER_IMAGE and attempts < 300:
        attempts += 1
        x = int(rng.integers(0, w - PATCH_W))
        y = int(rng.integers(0, h - PATCH_H))
        roi_mask = bg_mask[y:y + PATCH_H, x:x + PATCH_W]
        if roi_mask.mean() / 255.0 < MIN_BG_FRACTION:
            continue
        patches.append(image[y:y + PATCH_H, x:x + PATCH_W].copy())
    return patches


def main() -> None:
    rng = np.random.default_rng(42)
    paths = sorted(TRAIN_DIR.glob("*.jpg"))
    print(f"Extraction depuis {len(paths)} images train...")
    n_total = 0
    for path in tqdm(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        patches = extract_patches(image, rng)
        for i, p in enumerate(patches):
            out_path = OUT / f"{path.stem}_{i:02d}.png"
            cv2.imwrite(str(out_path), p)
            n_total += 1
    print(f"\n=> {n_total} patches sauvés dans {OUT}")


if __name__ == "__main__":
    main()
