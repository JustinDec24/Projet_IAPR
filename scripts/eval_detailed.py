"""Évaluation détaillée : par image, montre la prédiction vs la GT pour debug."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.inference import Classifier, predict_scene  # noqa: E402

CHECKPOINT = ROOT / "outputs" / "models" / "classifier.pt"


def parse_hand(s: str) -> list[str]:
    return [] if (not s or s == "EMPTY") else s.split(";")


def f1_multiset(pred: list[str], gt: list[str]) -> float:
    p, g = {}, {}
    for c in pred: p[c] = p.get(c, 0) + 1
    for c in gt: g[c] = g.get(c, 0) + 1
    tp = sum(min(p.get(k, 0), g.get(k, 0)) for k in set(p) | set(g))
    fp = sum(p.values()) - tp
    fn = sum(g.values()) - tp
    return 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> int:
    classifier = Classifier(CHECKPOINT)
    df = pd.read_csv(TRAIN_CSV)

    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        path = TRAIN_DIR / f"{row.image_id}.jpg"
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        pred = predict_scene(image, row.image_id, classifier)

        gt_hands = {p: parse_hand(getattr(row, f"player_{i + 1}_cards"))
                    for i, p in enumerate(["p1", "p2", "p3", "p4"])}
        gt_all = sum(gt_hands.values(), [])
        pred_all = sum(pred.player_cards.values(), [])
        f1 = f1_multiset(pred_all, gt_all)

        rows.append({
            "image_id": row.image_id,
            "center_ok": int(pred.center_card == row.center_card),
            "active_ok": int(pred.active_player == row.active_player),
            "f1": round(f1, 3),
            "n_gt": len(gt_all),
            "n_pred": len(pred_all),
            "gt_center": row.center_card, "pred_center": pred.center_card,
            "gt_active": row.active_player, "pred_active": pred.active_player,
        })

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "eval_per_image.csv", index=False)

    print("\nWORST 15 (par F1) :")
    print(out.nsmallest(15, "f1").to_string(index=False))

    print("\nBEST 10 (par F1) :")
    print(out.nlargest(10, "f1").to_string(index=False))

    print("\n=== Erreurs ActiveAcc ===")
    bad = out[out["active_ok"] == 0]
    print(bad[["image_id", "gt_active", "pred_active"]].head(20).to_string(index=False))

    print(f"\nGlobal: CenterAcc={out['center_ok'].mean():.3f} "
          f"ActiveAcc={out['active_ok'].mean():.3f} F1={out['f1'].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
