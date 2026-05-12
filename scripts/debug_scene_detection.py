"""Debug : visualise la détection sur quelques images train annotées.

Pour chaque image, affiche :
- les rotated rects des cartes détectées (vert)
- la position du jeton actif (cercle bleu)
- l'assignation joueur de chaque carte (texte)
- la couleur dominante détectée vs ce que dit train.csv
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TEST_DIR, TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.detection import (  # noqa: E402
    assign_player,
    assign_token_to_player,
    detect_active_token,
    detect_cards_in_scene,
)

DEBUG_DIR = ROOT / "outputs" / "debug_scenes"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def gt_for(image_id: str, df: pd.DataFrame) -> dict:
    row = df[df["image_id"] == image_id]
    if row.empty:
        return {}
    r = row.iloc[0].to_dict()
    return r


def draw_card(vis: np.ndarray, card: dict, player: str) -> None:
    box = cv2.boxPoints(card["rect"]).astype(np.intp)
    cv2.drawContours(vis, [box], 0, (0, 200, 0), 5)
    cx, cy = int(card["cx"]), int(card["cy"])
    cv2.putText(vis, f"{player}/{card['color']}", (cx - 100, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 4)


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    # Sélection d'images variées : white bg simple, white bg + overlap, noisy bg
    selected = ["L1000770", "L1000843", "L1000909", "L1000972", "L1000820"]
    for image_id in selected:
        path = TRAIN_DIR / f"{image_id}.jpg"
        if not path.exists():
            print(f"Skip {image_id} (introuvable)")
            continue
        image = cv2.imread(str(path))
        cards = detect_cards_in_scene(image)
        token = detect_active_token(image)

        gt = gt_for(image_id, df)
        print(f"\n=== {image_id} ===")
        print(f"  GT : center={gt.get('center_card')} active={gt.get('active_player')}")
        print(f"  Cartes détectées : {len(cards)}")
        if token is not None:
            via_angle = assign_player(*token, image.shape)
            via_cards = assign_token_to_player(token, cards, image.shape)
            print(f"  Jeton actif : ({token[0]:.0f}, {token[1]:.0f}) "
                  f"-> par angle: {via_angle} | par carte la plus proche: {via_cards}")
        else:
            print("  Jeton actif : non trouvé")

        vis = image.copy()
        # Cercle au centre de l'image
        h, w = image.shape[:2]
        cv2.circle(vis, (w // 2, h // 2), 30, (0, 0, 255), -1)
        for card in cards:
            player = assign_player(card["cx"], card["cy"], image.shape)
            draw_card(vis, card, player)
        if token is not None:
            cv2.circle(vis, (int(token[0]), int(token[1])), 50, (255, 100, 0), 8)

        out = DEBUG_DIR / f"{image_id}.jpg"
        cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
