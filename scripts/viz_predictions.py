"""Génère des images annotées comparant prédiction vs GT pour analyse qualitative.

Pour chaque image sélectionnée, dessine :
- les rectangles des cartes détectées avec leur label prédit (vert si correct, rouge si faux)
- la position du jeton détecté
- en titre : GT vs prédiction
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.detection import (  # noqa: E402
    assign_player,
    assign_token_to_player,
    detect_active_token,
    detect_cards_in_scene,
)
from src.inference import Classifier  # noqa: E402

CHECKPOINT = ROOT / "outputs" / "models" / "classifier.pt"
OUT_DIR = ROOT / "outputs" / "viz_preds"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12, help="number of images to visualize")
    p.add_argument("--mode", choices=["worst", "best", "all"], default="worst")
    p.add_argument("--confidence", type=float, default=0.40)
    return p.parse_args()


def parse_hand(s: str) -> list[str]:
    return [] if (not s or s == "EMPTY") else s.split(";")


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    classifier = Classifier(CHECKPOINT)
    df = pd.read_csv(TRAIN_CSV)

    if args.mode == "all":
        rows = df.itertuples(index=False)
    else:
        # Read previously computed eval_per_image.csv if available
        eval_csv = ROOT / "outputs" / "eval_per_image.csv"
        if not eval_csv.exists():
            print("eval_per_image.csv missing — run eval_detailed.py first")
            return 1
        eval_df = pd.read_csv(eval_csv)
        ascending = args.mode == "worst"
        ids = eval_df.sort_values("f1", ascending=ascending).head(args.n)["image_id"].tolist()
        rows = [df[df.image_id == i].iloc[0] for i in ids]

    for row in tqdm(list(rows)):
        path = TRAIN_DIR / f"{row.image_id}.jpg"
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        cards = detect_cards_in_scene(image)
        crops = [c["warped"] for c in cards]
        preds = classifier.classify(crops)
        for c, (lbl, conf) in zip(cards, preds):
            c["label"] = lbl
            c["conf"] = conf

        cards = [c for c in cards if c["conf"] >= args.confidence]
        token = detect_active_token(image)
        active = assign_token_to_player(token, cards, image.shape) if token else None

        gt_player_cards = {f"p{i}": parse_hand(getattr(row, f"player_{i}_cards"))
                           for i in (1, 2, 3, 4)}
        gt_all = sum(gt_player_cards.values(), [])

        # Determine which predicted cards are TP vs FP w.r.t. GT
        gt_remaining = list(gt_all)
        for c in cards:
            if c["label"] in gt_remaining:
                gt_remaining.remove(c["label"])
                c["match"] = True
            else:
                c["match"] = False

        # Annotate image
        vis = image.copy()
        for c in cards:
            color = (0, 200, 0) if c["match"] else (40, 40, 220)
            box = cv2.boxPoints(c["rect"]).astype(np.intp)
            cv2.drawContours(vis, [box], 0, color, 8)
            cv2.putText(vis, f"{c['label']}({c['conf']:.2f})",
                        (int(c["cx"]) - 110, int(c["cy"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)

        if token:
            cv2.circle(vis, (int(token[0]), int(token[1])), 60, (255, 100, 0), 10)

        # Title bar with GT/pred summary
        h, w = vis.shape[:2]
        gt_str = f"GT: center={row.center_card} active={row.active_player} "
        gt_str += f"cards=[{','.join(gt_all)}]"
        pred_str = f"Pred: active={active} ({len(cards)} cards detected)"
        cv2.rectangle(vis, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(vis, gt_str[:140], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(vis, pred_str[:140], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out = OUT_DIR / f"{args.mode}_{row.image_id}.jpg"
        cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"\nViz saved in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
