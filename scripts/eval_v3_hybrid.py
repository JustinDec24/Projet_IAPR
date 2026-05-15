"""Éval hybride V3-detector + V2 full-card classifier vs V2 0.787."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR
from src.inference import Classifier
from src.v3.corner_detector_runtime import CornerDetectorRuntime
from src.v3.predict_scene_v3 import CornerClassifierRuntime
from src.v3.predict_scene_hybrid import predict_scene_hybrid


def parse_hand(s):
    return [] if (not s or s == "EMPTY") else s.split(";")


def f1_multiset(p, g):
    pc, gc = Counter(p), Counter(g)
    tp = sum((pc & gc).values())
    fp = sum(pc.values()) - tp
    fn = sum(gc.values()) - tp
    return 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--diag-min", type=float, default=200.0)
    ap.add_argument("--diag-max", type=float, default=520.0)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    det = CornerDetectorRuntime(ROOT / "checkpoints" / "v3_corner_detector.pth")
    cclf = CornerClassifierRuntime(ROOT / "checkpoints" / "v3_corner_classifier.pth")
    fclf = Classifier(ROOT / "outputs" / "models" / "classifier_student.pt")
    df = pd.read_csv(TRAIN_CSV)

    c_ok = a_ok = n = 0
    f1s = []
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="hybrid"):
        p = TRAIN_DIR / f"{row.image_id}.jpg"
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        pr = predict_scene_hybrid(img, row.image_id, det, cclf, fclf,
                                  conf_threshold=args.conf,
                                  diag_min=args.diag_min,
                                  diag_max=args.diag_max,
                                  use_tta=not args.no_tta)
        n += 1
        c_ok += int(pr.center_card == row.center_card)
        a_ok += int(pr.active_player == row.active_player)
        gt, pd_ = [], []
        for i, s in enumerate(["p1", "p2", "p3", "p4"]):
            gt += parse_hand(getattr(row, f"player_{i+1}_cards"))
            pd_ += pr.player_cards[s]
        f1s.append(f1_multiset(pd_, gt))
        rows.append({"image_id": row.image_id, "f1": round(f1s[-1], 3),
                     "center_ok": int(pr.center_card == row.center_card),
                     "active_ok": int(pr.active_player == row.active_player),
                     "n_gt": len(gt), "n_pred": len(pd_)})

    ca, aa, f1 = c_ok / n, a_ok / n, sum(f1s) / len(f1s)
    score = 0.1 * ca + 0.1 * aa + 0.8 * f1
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "v3_hybrid_eval.csv", index=False)
    print("\n" + "=" * 50)
    print(f"V3-HYBRID ({n} img)  conf={args.conf} tta={not args.no_tta}")
    print("=" * 50)
    print(f"  CenterAcc {ca:.3f} | ActiveAcc {aa:.3f} | F1 {f1:.3f}")
    print(f"  SCORE {score:.3f}   (V2+ = 0.787)")
    print(f"  {'+' if score>=0.787 else ''}{score-0.787:+.3f} vs V2+")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
