"""Génère une mosaïque PNG de tous les templates pour inspection visuelle.

Layout :
    rangée 1 : y_0 ... y_9, y_skip, y_reverse, y_draw_2
    rangée 2 : r_*  (idem)
    rangée 3 : g_*
    rangée 4 : b_*
    rangée 5 : wild, draw_4 (centrés)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CARD_CLASSES  # noqa: E402

TEMPLATES_DIR = ROOT / "data" / "card_templates"
OUT = ROOT / "outputs" / "templates_preview.png"

CELL_W, CELL_H = 140, 210  # crop displayed
LABEL_H = 28
PAD = 6

VALUES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "skip", "reverse", "draw_2"]
COLORS = ["y", "r", "g", "b"]


def load_cell(label: str) -> np.ndarray:
    path = TEMPLATES_DIR / f"{label}.png"
    cell = np.full((CELL_H + LABEL_H, CELL_W, 3), 240, dtype=np.uint8)
    if not path.exists():
        cv2.putText(cell, "MISSING", (10, CELL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    else:
        img = cv2.imread(str(path))
        img = cv2.resize(img, (CELL_W, CELL_H))
        cell[:CELL_H] = img
    cv2.putText(cell, label, (4, CELL_H + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    return cell


def main() -> None:
    n_cols = len(VALUES)
    n_rows = len(COLORS) + 1  # 4 colors + wilds row
    cell_full_h = CELL_H + LABEL_H

    mosaic = np.full(
        (n_rows * cell_full_h + (n_rows + 1) * PAD,
         n_cols * CELL_W + (n_cols + 1) * PAD, 3),
        255, dtype=np.uint8,
    )

    def place(cell: np.ndarray, row: int, col: int) -> None:
        y0 = PAD + row * (cell_full_h + PAD)
        x0 = PAD + col * (CELL_W + PAD)
        mosaic[y0:y0 + cell_full_h, x0:x0 + CELL_W] = cell

    for ri, color in enumerate(COLORS):
        for ci, val in enumerate(VALUES):
            place(load_cell(f"{color}_{val}"), ri, ci)

    # Wild row, centred
    wild_row = len(COLORS)
    middle_col = (n_cols - 2) // 2
    place(load_cell("wild"), wild_row, middle_col)
    place(load_cell("draw_4"), wild_row, middle_col + 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), mosaic)
    print(f"Mosaïque enregistrée : {OUT}")
    print(f"Dimensions : {mosaic.shape[1]} x {mosaic.shape[0]} px")
    print(f"Templates manquants : {sorted(set(CARD_CLASSES) - {p.stem for p in TEMPLATES_DIR.glob('*.png')})}")


if __name__ == "__main__":
    main()
