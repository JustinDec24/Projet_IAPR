"""Phase 1 V3 — Extraction des coins haut-gauche depuis les crops labellisés.

Les annotations YOLO n'ont pas de label par bbox (classe unique "card"). On
utilise donc les crops réels déjà auto-labellisés (`data/real_crops/<label>/`),
qui sont warpés en canonique 200×300 → le coin discriminant (couleur+valeur)
est toujours dans le quadrant haut-gauche.

Coin = 35 % × 35 % du crop, en haut à gauche.

Sortie : data/corners_test/<label>/<id>.png
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_corners")

SRC_DIR = ROOT / "data" / "real_crops"
OUT_DIR = ROOT / "data" / "corners_test"
CORNER_FRAC = 0.35  # 35% × 35% du coin haut-gauche


def main() -> int:
    if not SRC_DIR.exists():
        logger.error("Source manquante : %s", SRC_DIR)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_total = 0
    per_class: dict[str, int] = {}
    for label_dir in sorted(SRC_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        out_label = OUT_DIR / label
        out_label.mkdir(parents=True, exist_ok=True)
        for crop_path in sorted(label_dir.glob("*.png")):
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue
            h, w = crop.shape[:2]
            ch, cw = int(h * CORNER_FRAC), int(w * CORNER_FRAC)
            corner = crop[:ch, :cw]
            if corner.size == 0:
                continue
            cv2.imwrite(str(out_label / crop_path.name), corner)
            n_total += 1
            per_class[label] = per_class.get(label, 0) + 1

    logger.info("Extrait %d coins dans %s", n_total, OUT_DIR)
    logger.info("Classes couvertes : %d / 54", len(per_class))
    missing = sorted(set(
        f"{c}_{n}" for c in ("r", "g", "b", "y") for n in list(range(10)) + ["skip", "reverse", "draw_2"]
    ) | {"wild", "draw_4"})
    covered = set(per_class)
    not_covered = [c for c in missing if c not in covered]
    if not_covered:
        logger.warning("Classes SANS coin de test : %s", not_covered)
    print(f"\nTotal : {n_total} coins, {len(per_class)} classes")
    print("Distribution (min/max) :",
          min(per_class.values()), "/", max(per_class.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
