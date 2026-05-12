"""Extrait les 54 templates de cartes vers data/card_templates/<class>.png.

Source : data/reference_images/ (4 planches officielles de la compétition).
Sortie : data/card_templates/<class>.png pour les 54 classes UNO.
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CARD_CLASSES, REFERENCE_DIR  # noqa: E402
from src.templates import LAYOUTS, extract_from_image  # noqa: E402

OUT_DIR = ROOT / "data" / "card_templates"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    extracted: dict[str, Path] = {}

    for image_path in sorted(REFERENCE_DIR.iterdir()):
        if image_path.stem not in LAYOUTS:
            print(f"Skip {image_path.name} (pas de layout défini)")
            continue
        print(f"\n[{image_path.name}]")
        templates = extract_from_image(image_path)
        for label, crop in templates.items():
            out = OUT_DIR / f"{label}.png"
            cv2.imwrite(str(out), crop)
            extracted[label] = out
            print(f"  {label} -> {out.name}")

    missing = sorted(set(CARD_CLASSES) - set(extracted))
    print(f"\n=> Extracted {len(extracted)} / {len(CARD_CLASSES)} classes")
    if missing:
        print(f"!! MANQUANTES : {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
