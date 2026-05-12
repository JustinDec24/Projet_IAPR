"""Sauve tous les crops detectés d'une image + leur classification, pour inspection."""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DIR  # noqa: E402
from src.detection import detect_cards_in_scene  # noqa: E402
from src.inference import Classifier  # noqa: E402

OUT = ROOT / "outputs" / "crops_debug"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = ROOT / "outputs" / "models" / "classifier.pt"
classifier = Classifier(CHECKPOINT)

for image_id in ["L1000909", "L1000972", "L1000917"]:
    img = cv2.imread(str(TRAIN_DIR / f"{image_id}.jpg"))
    cards = detect_cards_in_scene(img)
    crops = [c["warped"] for c in cards]
    preds = classifier.classify(crops)
    print(f"\n=== {image_id} : {len(cards)} detections ===")
    for i, (c, (lbl, conf)) in enumerate(zip(cards, preds)):
        out = OUT / f"{image_id}_{i:02d}_{lbl}_{conf:.2f}.png"
        cv2.imwrite(str(out), c["warped"])
        print(f"  [{i:>2}] cx={c['cx']:>5.0f} cy={c['cy']:>5.0f} "
              f"src={c['source']:>4} color={c['color']} -> {lbl} ({conf:.2f})")
