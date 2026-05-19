"""Filtre les crops réels pour ne garder que ceux où au moins 1 coin de carte
est visible (= corners blancs + digit interne lisible).

Critères :
- Cas 1 (carte entière) : au moins 3 des 4 coins du crop ont >60% de pixels blancs
  (= la bordure blanche de la carte). Indique une carte centrée et complète.
- Cas 2 (coin visible) : au moins 1 des 4 coins a une zone blanche AVEC stroke
  sombre à l'intérieur (= mini-digit du coin lisible).

Sinon : rejette (déplace dans real_crops_rejected/ pour inspection).
"""
import sys
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REAL_CROPS = ROOT / "data" / "real_crops"
REJECTED = ROOT / "data" / "real_crops_rejected"


def has_visible_card_features(crop: np.ndarray) -> tuple[bool, str]:
    """Retourne (is_valid, reason)."""
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Définir les 4 régions corners (~10% de chaque dim)
    ch, cw = max(20, h // 6), max(15, w // 6)
    corners = {
        "TL": hsv[:ch, :cw],
        "TR": hsv[:ch, w - cw:],
        "BL": hsv[h - ch:, :cw],
        "BR": hsv[h - ch:, w - cw:],
    }

    white_fracs = {}
    dark_fracs = {}
    for name, reg in corners.items():
        white = ((reg[..., 2] > 175) & (reg[..., 1] < 55)).mean()
        dark = (reg[..., 2] < 80).mean()
        white_fracs[name] = float(white)
        dark_fracs[name] = float(dark)

    # Cas 1 : carte entière (bordure blanche dans la majorité des corners)
    n_white_corners = sum(1 for f in white_fracs.values() if f > 0.55)
    if n_white_corners >= 3:
        return True, "full_card"

    # Cas 2 : au moins 1 coin avec mini-digit (blanc + stroke sombre)
    for name in corners:
        if white_fracs[name] > 0.30 and 0.03 < dark_fracs[name] < 0.45:
            return True, f"corner_{name}"

    return False, "no_visible_features"


def main() -> int:
    if not REAL_CROPS.exists():
        print(f"Not found: {REAL_CROPS}")
        return 1
    REJECTED.mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_rejected = 0
    reasons = defaultdict(int)

    for label_dir in sorted(REAL_CROPS.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for crop_path in sorted(label_dir.glob("*.png")):
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue
            ok, reason = has_visible_card_features(crop)
            reasons[reason] += 1
            if ok:
                n_kept += 1
            else:
                # Move to rejected
                rej_dir = REJECTED / label
                rej_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(crop_path), str(rej_dir / crop_path.name))
                n_rejected += 1

    print(f"\n=== FILTER RESULTS ===")
    print(f"Kept     : {n_kept}")
    print(f"Rejected : {n_rejected} (moved to {REJECTED.name}/)")
    print(f"\nBreakdown :")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason:>22} : {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
