"""Génère le fichier de soumission Kaggle pour la compétition IAPR-26 UNO Vision.

Usage :
    python main.py [--checkpoint outputs/models/classifier.pt] \
                   [--output outputs/submissions/submission.csv]

Hypothèses :
- `data/test_images/` contient les images de test (.jpg)
- `data/card_templates/` est déjà généré (script `extract_templates.py`)
- `outputs/models/classifier.pt` contient le checkpoint du CNN entraîné
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import SUBMISSIONS_DIR, TEST_DIR  # noqa: E402
from src.detector_inference import CardDetectorRuntime  # noqa: E402
from src.inference import Classifier, predict_scene  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "outputs" / "models" / "classifier.pt"
DEFAULT_DETECTOR = ROOT / "outputs" / "models" / "detector.pt"
DEFAULT_OUTPUT = SUBMISSIONS_DIR / "submission.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR,
                   help="Checkpoint du détecteur (utilisé si présent, sinon heuristique)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--test-dir", type=Path, default=TEST_DIR)
    p.add_argument("--confidence", type=float, default=0.50)
    p.add_argument("--tta", action="store_true", help="Activate test-time augmentation")
    p.add_argument("--no-detector", action="store_true",
                   help="Désactiver le détecteur appris (fallback sur la détection heuristique)")
    p.add_argument("--hybrid", action="store_true",
                   help="Heuristique pour la carte centrale + détecteur appris pour les joueurs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint introuvable : {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.test_dir.exists():
        print(f"ERROR: Dossier test introuvable : {args.test_dir}", file=sys.stderr)
        return 1

    classifier = Classifier(args.checkpoint)
    print(f"Classifier : {args.checkpoint.name} (device={classifier.device})")

    detector = None
    if not args.no_detector and args.detector.exists():
        detector = CardDetectorRuntime(args.detector, device=str(classifier.device))
        print(f"Detector  : {args.detector.name} (appris)")
    else:
        print("Detector  : heuristique (HSV + ovales blancs)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.test_dir.glob("*.jpg"))
    print(f"Test images : {len(image_paths)}")

    fieldnames = [
        "image_id", "center_card", "active_player",
        "player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path in tqdm(image_paths, desc="Inference"):
            image = cv2.imread(str(path))
            if image is None:
                continue
            pred = predict_scene(image, path.stem, classifier,
                                 confidence_threshold=args.confidence,
                                 use_tta=args.tta,
                                 detector=detector,
                                 hybrid=args.hybrid)
            writer.writerow(pred.to_csv_row())

    print(f"\nSoumission écrite : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
