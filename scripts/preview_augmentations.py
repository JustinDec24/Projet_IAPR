"""Génère une mosaïque d'exemples d'augmentation pour quelques templates.

Pour chaque carte choisie, on affiche le template original puis N variantes
augmentées, pour vérifier visuellement que l'augmentation introduit assez de
diversité sans détruire l'identité de la carte (couleur + chiffre/symbole).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.augmentation import augment  # noqa: E402

TEMPLATES_DIR = ROOT / "data" / "card_templates"
OUT = ROOT / "outputs" / "augmentations_preview.png"

# Sélection de cartes représentatives : couleurs / chiffres / actions / wilds
SAMPLE_LABELS = ["y_4", "r_skip", "g_reverse", "b_3", "y_draw_2", "wild", "draw_4"]
N_VARIANTS = 9
CELL_W, CELL_H = 140, 210
LABEL_H = 24
PAD = 6


def main() -> None:
    rng = np.random.default_rng(42)
    n_cols = 1 + N_VARIANTS  # original + N augmentations
    n_rows = len(SAMPLE_LABELS)
    cell_full_h = CELL_H + LABEL_H

    mosaic = np.full(
        (n_rows * cell_full_h + (n_rows + 1) * PAD,
         n_cols * CELL_W + (n_cols + 1) * PAD, 3),
        255, dtype=np.uint8,
    )

    def place(cell_img: np.ndarray, text: str, row: int, col: int) -> None:
        y0 = PAD + row * (cell_full_h + PAD)
        x0 = PAD + col * (CELL_W + PAD)
        resized = cv2.resize(cell_img, (CELL_W, CELL_H))
        mosaic[y0:y0 + CELL_H, x0:x0 + CELL_W] = resized
        cv2.putText(mosaic, text, (x0 + 4, y0 + CELL_H + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    for ri, label in enumerate(SAMPLE_LABELS):
        template = cv2.imread(str(TEMPLATES_DIR / f"{label}.png"))
        place(template, f"{label} (orig)", ri, 0)
        for ci in range(N_VARIANTS):
            aug = augment(template, rng)
            place(aug, f"#{ci + 1}", ri, 1 + ci)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), mosaic)
    print(f"Mosaïque enregistrée : {OUT}")
    print(f"Dimensions : {mosaic.shape[1]} x {mosaic.shape[0]} px")


if __name__ == "__main__":
    main()
