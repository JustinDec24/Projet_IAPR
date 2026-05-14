"""Visualise le pipeline complet sur une image (zones joueurs, détections, crops, prédictions).

Usage :
    python scripts/viz_pipeline.py L1000770
    python scripts/viz_pipeline.py L1000770 L1000909 L1000843
    python scripts/viz_pipeline.py --all-train          # toutes les images train
    python scripts/viz_pipeline.py --first-test 10      # 10 premières images test

Sortie : outputs/viz_pipeline/<image_id>.jpg
Chaque viz contient :
- L'image originale avec :
    - les ZONES JOUEURS coloriées (semi-transparent : p1=bleu, p2=vert, p3=orange, p4=violet, center=jaune)
    - les CARTES DÉTECTÉES (bbox de la couleur du joueur, label prédit + confiance)
    - le JETON ACTIF (cercle orange épais)
- Un panneau à droite avec les CROPS WARPÉS 200×300 que le CNN reçoit, avec leur label prédit
"""

import argparse
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
from src.detector_inference import CardDetectorRuntime, detect_cards_with_model  # noqa: E402
from src.inference import Classifier, predict_scene  # noqa: E402

CHECKPOINT = ROOT / "outputs" / "models" / "classifier.pt"
DETECTOR_CHECKPOINT = ROOT / "outputs" / "models" / "detector.pt"
OUT_DIR = ROOT / "outputs" / "viz_pipeline"

# Couleur BGR par "zone" joueur
PLAYER_COLORS = {
    "p1": (220, 100, 80),    # bleu
    "p2": (80, 180, 80),     # vert
    "p3": (60, 140, 230),    # orange
    "p4": (200, 80, 200),    # violet
    "center": (60, 220, 220),  # jaune
}
TOKEN_COLOR = (40, 130, 255)


def draw_player_zones(image: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    """Dessine les régions de chaque joueur en overlay semi-transparent.

    On colorise chaque pixel selon `assign_player(cx, cy)`. Pour des raisons de
    vitesse, on subsample en grille puis on tile.
    """
    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)
    step = 40  # grille de pixels (40 px)
    for y in range(0, h, step):
        for x in range(0, w, step):
            cx = x + step / 2
            cy = y + step / 2
            player = assign_player(cx, cy, image.shape)
            color = PLAYER_COLORS[player]
            cv2.rectangle(overlay, (x, y), (x + step, y + step), color, thickness=-1)
    return cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)


def annotate_image(image: np.ndarray, cards: list[dict],
                   token: tuple[float, float] | None,
                   active_player: str | None,
                   gt: dict | None = None) -> np.ndarray:
    """Dessine zones joueurs + cartes + jeton sur l'image."""
    vis = draw_player_zones(image)

    for c in cards:
        player = assign_player(c["cx"], c["cy"], image.shape)
        color = PLAYER_COLORS[player]
        box = cv2.boxPoints(c["rect"]).astype(np.intp)
        # Solid stroke for heuristic, dashed-like (thinner) for learned detector
        thickness = 10 if c.get("source") != "yolo" else 6
        cv2.drawContours(vis, [box], 0, color, thickness=thickness)
        label = c.get("label", "?")
        conf = c.get("confidence", 0.0)
        src_marker = "*" if c.get("source") == "yolo" else ""
        text = f"{label}{src_marker} ({conf:.2f})"
        cv2.putText(vis, text, (int(c["cx"]) - 130, int(c["cy"]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
        cv2.putText(vis, text, (int(c["cx"]) - 130, int(c["cy"]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)

    if token is not None:
        cv2.circle(vis, (int(token[0]), int(token[1])), 70, TOKEN_COLOR, 12)
        cv2.circle(vis, (int(token[0]), int(token[1])), 12, TOKEN_COLOR, -1)

    # Titre en haut
    h, w = vis.shape[:2]
    header_h = 140
    header = np.full((header_h, w, 3), 30, dtype=np.uint8)
    if gt is not None:
        gt_text = f"GT     : center={gt['center']} active={gt['active']}"
        cv2.putText(header, gt_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 3)
    pred_center = next((c["label"] for c in cards
                        if assign_player(c["cx"], c["cy"], image.shape) == "center"), "EMPTY")
    pred_text = f"Pred   : center={pred_center} active={active_player or 'EMPTY'} ({len(cards)} cards)"
    cv2.putText(header, pred_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 230, 230), 3)
    return np.vstack([header, vis])


def build_crops_panel(cards: list[dict], target_height: int) -> np.ndarray:
    """Construit un panneau vertical des crops warpés avec leur label prédit."""
    if not cards:
        return np.full((target_height, 250, 3), 30, dtype=np.uint8)

    crop_w, crop_h = 200, 300
    label_h = 50
    cell_h = crop_h + label_h
    n_cols = max(1, target_height // cell_h)
    n_rows_per_col = -(-len(cards) // n_cols)  # ceil
    actual_height = max(target_height, n_rows_per_col * cell_h + 20)
    panel_w = n_cols * (crop_w + 20) + 20

    panel = np.full((actual_height, panel_w, 3), 30, dtype=np.uint8)
    for i, c in enumerate(cards):
        col = i // n_rows_per_col
        row = i % n_rows_per_col
        x = 20 + col * (crop_w + 20)
        y = 20 + row * cell_h
        panel[y:y + crop_h, x:x + crop_w] = c["warped"]
        text = f"{c.get('label', '?')} ({c.get('confidence', 0):.2f})"
        cv2.putText(panel, text, (x + 5, y + crop_h + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2)
    return panel[:target_height] if actual_height >= target_height else panel


def visualize(image_id: str, image: np.ndarray, classifier: Classifier,
              gt: dict | None,
              detector: CardDetectorRuntime | None = None,
              hybrid: bool = False,
              use_tta: bool = False,
              confidence_threshold: float = 0.40) -> np.ndarray:
    """Pipeline complet + rendu : image annotée + panneau crops.

    Reproduit la logique de `predict_scene` (hybride/détecteur/heuristique) en
    gardant les `cards` détaillées pour la viz.
    """
    token = detect_active_token(image)

    if hybrid and detector is not None:
        heur_cards = detect_cards_in_scene(image, exclude_xy=token)
        model_cards = detect_cards_with_model(image, detector, exclude_xy=token)
        cards = [c for c in heur_cards
                 if assign_player(c["cx"], c["cy"], image.shape) == "center"]
        cards += [c for c in model_cards
                  if assign_player(c["cx"], c["cy"], image.shape) != "center"]
    elif detector is not None:
        cards = detect_cards_with_model(image, detector, exclude_xy=token)
    else:
        cards = detect_cards_in_scene(image, exclude_xy=token)

    crops = [c["warped"] for c in cards]
    if use_tta:
        import torch
        probs = classifier.classify_full_tta(crops) if crops else torch.zeros(0, 0)
        from src.config import IDX_TO_CLASS
        for i, card in enumerate(cards):
            top1_idx = int(probs[i].argmax().item())
            card["label"] = IDX_TO_CLASS[top1_idx]
            card["confidence"] = float(probs[i, top1_idx].item())
    else:
        preds = classifier.classify(crops)
        for card, (label, conf) in zip(cards, preds):
            card["label"] = label
            card["confidence"] = conf

    cards = [c for c in cards if c["confidence"] >= confidence_threshold]

    active_player = assign_token_to_player(token, cards, image.shape) if token else None

    annotated = annotate_image(image, cards, token, active_player, gt)
    crops_panel = build_crops_panel(cards, target_height=annotated.shape[0])

    # Resize annotated to match panel height
    h_match = annotated.shape[0]
    aspect = annotated.shape[1] / annotated.shape[0]
    # Limit max width to keep file size reasonable
    target_w = min(annotated.shape[1], 2400)
    target_h = int(target_w / aspect)
    annotated_resized = cv2.resize(annotated, (target_w, target_h))
    if crops_panel.shape[0] != annotated_resized.shape[0]:
        ratio = annotated_resized.shape[0] / crops_panel.shape[0]
        crops_panel = cv2.resize(
            crops_panel, (int(crops_panel.shape[1] * ratio), annotated_resized.shape[0])
        )
    return np.hstack([annotated_resized, crops_panel])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("image_ids", nargs="*", help="image IDs to visualize (sans .jpg)")
    p.add_argument("--all-train", action="store_true")
    p.add_argument("--first-test", type=int, default=0, metavar="N",
                   help="visualize the first N test images")
    p.add_argument("--hybrid", action="store_true",
                   help="Heuristique (centre) + détecteur appris (joueurs)")
    p.add_argument("--no-detector", action="store_true",
                   help="Désactive le détecteur appris (heuristique seule)")
    p.add_argument("--tta", action="store_true", help="Test-Time Augmentation 8×")
    p.add_argument("--confidence", type=float, default=0.40)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    gt_lookup = df.set_index("image_id").to_dict(orient="index")

    paths: list[Path] = []
    if args.all_train:
        paths = sorted(TRAIN_DIR.glob("*.jpg"))
    elif args.first_test:
        paths = sorted(TEST_DIR.glob("*.jpg"))[:args.first_test]
    else:
        for image_id in args.image_ids:
            for d in (TRAIN_DIR, TEST_DIR):
                if (d / f"{image_id}.jpg").exists():
                    paths.append(d / f"{image_id}.jpg")
                    break
            else:
                print(f"Not found : {image_id}")

    if not paths:
        print("Usage : python scripts/viz_pipeline.py <image_id> [...] OR --all-train OR --first-test N")
        return 1

    classifier = Classifier(CHECKPOINT)
    print(f"Classifier loaded from {CHECKPOINT}")

    detector = None
    if not args.no_detector and DETECTOR_CHECKPOINT.exists():
        detector = CardDetectorRuntime(DETECTOR_CHECKPOINT, device=str(classifier.device))
        mode = "hybrid" if args.hybrid else "learned"
        print(f"Detector loaded ({mode} mode){' + TTA' if args.tta else ''}")
    else:
        print(f"Detector : heuristique seule{' + TTA' if args.tta else ''}")

    print(f"Generating {len(paths)} visualizations...")

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gt_row = gt_lookup.get(path.stem)
        gt = {"center": gt_row["center_card"], "active": gt_row["active_player"]} if gt_row else None
        viz = visualize(path.stem, image, classifier, gt,
                        detector=detector, hybrid=args.hybrid,
                        use_tta=args.tta, confidence_threshold=args.confidence)
        out = OUT_DIR / f"{path.stem}.jpg"
        cv2.imwrite(str(out), viz, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"  {path.stem} -> {out.name}")
    print(f"\nSaved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
