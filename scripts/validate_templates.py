"""Vérifie que chaque template a la couleur dominante attendue par son label."""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CARD_CLASSES  # noqa: E402
from src.templates import dominant_color  # noqa: E402

TEMPLATES_DIR = ROOT / "data" / "card_templates"


def expected_color(label: str) -> str:
    if label in ("wild", "draw_4"):
        return "k"
    return label[0]


def main() -> int:
    print(f"{'label':12} {'expected':>8} {'detected':>8} {'status':>10}")
    n_ok, n_bad, n_missing = 0, 0, 0
    for label in CARD_CLASSES:
        path = TEMPLATES_DIR / f"{label}.png"
        if not path.exists():
            print(f"{label:12} {expected_color(label):>8} {'-':>8} {'MISSING':>10}")
            n_missing += 1
            continue
        crop = cv2.imread(str(path))
        det = dominant_color(crop)
        exp = expected_color(label)
        ok = det == exp
        status = "ok" if ok else "MISMATCH"
        print(f"{label:12} {exp:>8} {det:>8} {status:>10}")
        if ok:
            n_ok += 1
        else:
            n_bad += 1
    print(f"\n=> ok={n_ok} mismatch={n_bad} missing={n_missing}")
    return 0 if n_bad == 0 and n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
