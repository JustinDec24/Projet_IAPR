"""Auto-labélise les train_images : produit des crops réels labellisés pour
fine-tuner le classifieur.

Algo (pour chaque train_image) :
1. Détecter les cartes via le pipeline (`detect_cards_in_scene`).
2. Pour chaque carte, déterminer le slot joueur (`assign_player`).
3. Dans chaque slot, lire les cartes GT du CSV.
4. Matcher crops ↔ labels GT par COULEUR :
   - On groupe GT et crops par couleur (y/r/g/b/k).
   - Si len(crops_couleur) == len(gt_couleur) et toutes les GT ont la MÊME valeur,
     on labelle tous les crops avec cette valeur.
   - Si len(crops_couleur) == len(gt_couleur) == 1, on labelle le crop avec le seul GT.
   - Sinon (ambiguïté valeur) : on skip ce groupe couleur — on préfère ne pas
     introduire de mauvais labels.
5. Sauve les crops matchés dans `data/real_crops/<label>/<image_id>_<idx>.png`.

Sortie attendue : 300-500 crops réels étiquetés (sur les 600+ cartes des 81 images train).
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.detection import (  # noqa: E402
    assign_player,
    detect_active_token,
    detect_cards_in_scene,
)

OUT_DIR = ROOT / "data" / "real_crops"


def color_of(label: str) -> str:
    """y_4 → y ; wild/draw_4 → k."""
    return "k" if label in ("wild", "draw_4") else label[0]


def parse_hand(s: str) -> list[str]:
    return [] if (not s or s == "EMPTY") else s.split(";")


def match_slot(crops: list[dict], gt_labels: list[str]) -> dict[int, str]:
    """Renvoie un mapping {crop_index → label} pour les matches confiants."""
    crops_by_color: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(crops):
        crops_by_color[c["color"]].append(i)
    gt_by_color: dict[str, list[str]] = defaultdict(list)
    for label in gt_labels:
        gt_by_color[color_of(label)].append(label)

    result: dict[int, str] = {}
    for color in set(crops_by_color) | set(gt_by_color):
        crop_idx = crops_by_color.get(color, [])
        gt_lbls = gt_by_color.get(color, [])
        if not crop_idx or not gt_lbls:
            continue
        # Tous les GT de cette couleur ont la même valeur → labelle tous les crops
        if len(set(gt_lbls)) == 1:
            for i in crop_idx[:len(gt_lbls)]:
                result[i] = gt_lbls[0]
        # Sinon : 1 crop + 1 GT (cas évident)
        elif len(crop_idx) == 1 and len(gt_lbls) == 1:
            result[crop_idx[0]] = gt_lbls[0]
        # else: ambiguïté (ex. 2 jaunes différents) → on skip pour éviter du bruit
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)

    n_saved = 0
    n_skipped_slot = 0
    by_class: dict[str, int] = defaultdict(int)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Auto-label"):
        path = TRAIN_DIR / f"{row.image_id}.jpg"
        if not path.exists():
            continue
        image = cv2.imread(str(path))

        token = detect_active_token(image)
        cards = detect_cards_in_scene(image, exclude_xy=token)
        if not cards:
            continue

        # Group cards by player slot
        by_slot: dict[str, list[dict]] = defaultdict(list)
        for c in cards:
            slot = assign_player(c["cx"], c["cy"], image.shape)
            by_slot[slot].append(c)

        # GT par slot
        gt_per_slot = {
            "p1": parse_hand(row.player_1_cards),
            "p2": parse_hand(row.player_2_cards),
            "p3": parse_hand(row.player_3_cards),
            "p4": parse_hand(row.player_4_cards),
            "center": [row.center_card] if row.center_card and row.center_card != "EMPTY" else [],
        }

        # Match dans chaque slot
        for slot, slot_crops in by_slot.items():
            slot_gt = gt_per_slot.get(slot, [])
            if not slot_gt:
                n_skipped_slot += 1
                continue
            matched = match_slot(slot_crops, slot_gt)
            for crop_idx, label in matched.items():
                label_dir = OUT_DIR / label
                label_dir.mkdir(parents=True, exist_ok=True)
                out_path = label_dir / f"{row.image_id}_{slot}_{crop_idx:02d}.png"
                cv2.imwrite(str(out_path), slot_crops[crop_idx]["warped"])
                n_saved += 1
                by_class[label] += 1

    print(f"\n=> {n_saved} crops réels étiquetés sauvés dans {OUT_DIR}")
    print(f"   {n_skipped_slot} slots skipped (no GT or all ambiguous)")
    print(f"\nDistribution par classe :")
    for label in sorted(by_class):
        print(f"  {label:12} : {by_class[label]:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
