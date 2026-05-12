"""Évalue le pipeline end-to-end sur les images d'entraînement annotées.

Calcule la métrique de la compétition :
    Score = 0.1 · CenterAcc + 0.1 · ActiveAcc + 0.8 · F1(cartes par joueur)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.detector_inference import CardDetectorRuntime  # noqa: E402
from src.inference import Classifier, predict_scene  # noqa: E402

CHECKPOINT = ROOT / "outputs" / "models" / "classifier.pt"
DETECTOR_CHECKPOINT = ROOT / "outputs" / "models" / "detector.pt"


def parse_hand(s: str) -> list[str]:
    return [] if (not s or s == "EMPTY") else s.split(";")


def f1_multiset(pred: list[str], gt: list[str]) -> float:
    """F1 sur multi-ensembles : 2·TP / (2·TP + FP + FN)."""
    pred_counts: dict[str, int] = {}
    gt_counts: dict[str, int] = {}
    for c in pred:
        pred_counts[c] = pred_counts.get(c, 0) + 1
    for c in gt:
        gt_counts[c] = gt_counts.get(c, 0) + 1
    tp = sum(min(pred_counts.get(k, 0), gt_counts.get(k, 0)) for k in set(pred_counts) | set(gt_counts))
    fp = sum(pred_counts.values()) - tp
    fn = sum(gt_counts.values()) - tp
    if 2 * tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--confidence", type=float, default=0.40)
    p.add_argument("--tta", action="store_true", help="Activate test-time augmentation")
    p.add_argument("--no-detector", action="store_true", help="Use heuristic detection")
    p.add_argument("--hybrid", action="store_true",
                   help="Heuristic for center card + learned detector for players")
    args = p.parse_args()

    if not CHECKPOINT.exists():
        print(f"Checkpoint introuvable : {CHECKPOINT}")
        return 1

    classifier = Classifier(CHECKPOINT)
    print(f"Modèle : {CHECKPOINT} (device={classifier.device})")
    print(f"Confidence threshold : {args.confidence}")

    detector = None
    if not args.no_detector and DETECTOR_CHECKPOINT.exists():
        detector = CardDetectorRuntime(DETECTOR_CHECKPOINT, device=str(classifier.device))
        print(f"Détecteur : {DETECTOR_CHECKPOINT.name} (appris)")
    else:
        print("Détecteur : heuristique HSV")

    df = pd.read_csv(TRAIN_CSV)

    center_oks, active_oks, f1s = [], [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Eval"):
        path = TRAIN_DIR / f"{row.image_id}.jpg"
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        pred = predict_scene(image, row.image_id, classifier,
                             confidence_threshold=args.confidence,
                             use_tta=args.tta,
                             detector=detector,
                             hybrid=args.hybrid)

        # CenterAcc : 1 si la prédiction matche, 0 sinon
        center_oks.append(int(pred.center_card == row.center_card))
        # ActiveAcc : pareil
        active_oks.append(int(pred.active_player == row.active_player))
        # F1 sur l'ensemble des cartes des 4 joueurs (multiset)
        gt_all: list[str] = []
        pred_all: list[str] = []
        for col, slot in [
            ("player_1_cards", "p1"), ("player_2_cards", "p2"),
            ("player_3_cards", "p3"), ("player_4_cards", "p4"),
        ]:
            gt_all += parse_hand(getattr(row, col))
            pred_all += pred.player_cards[slot]
        f1s.append(f1_multiset(pred_all, gt_all))

    center_acc = float(np.mean(center_oks))
    active_acc = float(np.mean(active_oks))
    f1 = float(np.mean(f1s))
    score = 0.1 * center_acc + 0.1 * active_acc + 0.8 * f1
    print(f"\n=== Résultats sur {len(center_oks)} images d'entraînement ===")
    print(f"  CenterAcc : {center_acc:.3f}")
    print(f"  ActiveAcc : {active_acc:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  Score     : {score:.3f}  (baseline DL ~0.647)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
