"""Produces the EXACT Kaggle submission for the IAPR-26 UNO Vision challenge.

Final pipeline (Score 0.850, 11.08M params):
- learned CenterNet detector  outputs/models/detector.pt    (4.77M)
- distilled student classifier outputs/models/classifier.pt  (6.31M)
- hybrid detection + 8x TTA + 0-param center-card ensemble

Just run:
    python main.py
→ writes outputs/submissions/submission.csv (the file uploaded to Kaggle).

TTA and hybrid mode are ON by default (required to reproduce 0.850).
Disable with --no-tta / --no-hybrid for ablation only.

Assumptions: data/test_images/*.jpg present, data/card_templates/ generated,
checkpoints present in outputs/models/.
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
    # TTA + hybrid ON by default (required for the 0.850 Kaggle submission).
    p.add_argument("--no-tta", dest="tta", action="store_false",
                   help="Disable 8x test-time augmentation (ablation only)")
    p.add_argument("--no-hybrid", dest="hybrid", action="store_false",
                   help="Disable hybrid mode + center ensemble (ablation only)")
    p.add_argument("--no-detector", action="store_true",
                   help="Disable learned detector (pure heuristic fallback)")
    p.set_defaults(tta=True, hybrid=True)
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
