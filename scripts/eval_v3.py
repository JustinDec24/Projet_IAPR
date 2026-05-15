"""Phase 7 V3 — Évaluation du pipeline corner-based sur les 81 images train.

Score compétition = 0.1·CenterAcc + 0.1·ActiveAcc + 0.8·F1(multiset cartes).
Compare avec V2 (0.764 baseline / 0.787 v2).
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.v3.corner_detector_runtime import CornerDetectorRuntime  # noqa: E402
from src.v3.predict_scene_v3 import (CornerClassifierRuntime,  # noqa: E402
                                     predict_scene_v3)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_v3")


def parse_hand(s: str) -> list[str]:
    return [] if (not s or s == "EMPTY") else s.split(";")


def f1_multiset(pred: list[str], gt: list[str]) -> float:
    p, g = Counter(pred), Counter(gt)
    tp = sum((p & g).values())
    fp = sum(p.values()) - tp
    fn = sum(g.values()) - tp
    return 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path,
                    default=ROOT / "checkpoints" / "v3_corner_detector.pth")
    ap.add_argument("--classifier", type=Path,
                    default=ROOT / "checkpoints" / "v3_corner_classifier.pth")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--diag-min", type=float, default=130.0)
    ap.add_argument("--diag-max", type=float, default=320.0)
    args = ap.parse_args()

    if not args.detector.exists() or not args.classifier.exists():
        logger.error("Checkpoints manquants: %s / %s", args.detector, args.classifier)
        return 1

    det = CornerDetectorRuntime(args.detector)
    clf = CornerClassifierRuntime(args.classifier)
    df = pd.read_csv(TRAIN_CSV)

    c_ok = a_ok = 0
    f1s: list[float] = []
    n = 0
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="eval_v3"):
        p = TRAIN_DIR / f"{row.image_id}.jpg"
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        pr = predict_scene_v3(img, row.image_id, det, clf,
                              conf_threshold=args.conf,
                              diag_min=args.diag_min, diag_max=args.diag_max)
        n += 1
        c_ok += int(pr.center_card == row.center_card)
        a_ok += int(pr.active_player == row.active_player)
        gt = []
        pd_ = []
        for i, slot in enumerate(["p1", "p2", "p3", "p4"]):
            gt += parse_hand(getattr(row, f"player_{i+1}_cards"))
            pd_ += pr.player_cards[slot]
        f1s.append(f1_multiset(pd_, gt))
        rows.append({"image_id": row.image_id,
                     "center_ok": int(pr.center_card == row.center_card),
                     "active_ok": int(pr.active_player == row.active_player),
                     "f1": round(f1s[-1], 3),
                     "n_gt": len(gt), "n_pred": len(pd_)})

    center_acc = c_ok / n
    active_acc = a_ok / n
    f1 = sum(f1s) / len(f1s)
    score = 0.1 * center_acc + 0.1 * active_acc + 0.8 * f1

    out = pd.DataFrame(rows)
    out_csv = ROOT / "reports" / "v3_eval.csv"
    out.to_csv(out_csv, index=False)

    print("\n" + "=" * 52)
    print(f"V3 EVAL ({n} images train)")
    print("=" * 52)
    print(f"  CenterAcc : {center_acc:.3f}")
    print(f"  ActiveAcc : {active_acc:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  SCORE     : {score:.3f}")
    print("  ---")
    print(f"  V2 baseline : 0.764  | V2+ : 0.787")
    print(f"  V3          : {score:.3f}  "
          f"({'+' if score >= 0.787 else ''}{score - 0.787:+.3f} vs V2+)")
    print(f"  CSV : {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
