"""Phase 2a V3 — Extraction des assets pour la génération synthétique.

1. Cartes : les 54 templates 200×300 → RGBA avec alpha en rectangle arrondi
   (le coin arrondi des cartes UNO, pour un compositing propre sans halo).
2. Fonds : patches extraits des 81 images train dans les zones SANS carte
   (inverse des bboxes YOLO + dilatation), séparés white/ vs noisy/.

Usage :
    python -m src.synthetic.extract_assets --config configs/v3_synthetic.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_assets")

ROOT = Path(__file__).resolve().parents[2]


def _rounded_alpha(h: int, w: int, radius_frac: float) -> np.ndarray:
    """Masque alpha (uint8) en rectangle à coins arrondis."""
    r = max(1, int(min(h, w) * radius_frac))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def extract_cards(cfg: dict) -> int:
    src = ROOT / cfg["assets"]["card_templates_dir"]
    out = ROOT / cfg["assets"]["out_cards_dir"]
    out.mkdir(parents=True, exist_ok=True)
    rf = cfg["assets"]["card_corner_radius_frac"]
    n = 0
    for p in sorted(src.glob("*.png")):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        alpha = _rounded_alpha(h, w, rf)
        rgba = np.dstack([bgr, alpha])
        cv2.imwrite(str(out / p.name), rgba)
        n += 1
    logger.info("Cartes RGBA extraites : %d -> %s", n, out)
    return n


def _load_yolo_boxes(annot_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Renvoie les bboxes en pixels (x0,y0,x1,y1)."""
    if not annot_path.exists():
        return []
    boxes = []
    for line in annot_path.read_text().strip().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        x0 = int((cx - bw / 2) * w)
        y0 = int((cy - bh / 2) * h)
        x1 = int((cx + bw / 2) * w)
        y1 = int((cy + bh / 2) * h)
        boxes.append((x0, y0, x1, y1))
    return boxes


def extract_backgrounds(cfg: dict) -> tuple[int, int]:
    img_dir = ROOT / cfg["assets"]["train_images_dir"]
    ann_dir = ROOT / cfg["assets"]["yolo_annotations_dir"]
    out_dir = ROOT / cfg["assets"]["out_bg_dir"]
    (out_dir / "white").mkdir(parents=True, exist_ok=True)
    (out_dir / "noisy").mkdir(parents=True, exist_ok=True)
    patch = cfg["assets"]["bg_patch_size"]
    per_img = cfg["assets"]["bg_patches_per_image"]
    v_thresh = cfg["assets"]["bg_white_v_threshold"]
    rng = np.random.default_rng(cfg["seed"])

    n_white, n_noisy = 0, 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = _load_yolo_boxes(ann_dir / f"{img_path.stem}.txt", w, h)
        # Masque des zones cartes (dilaté pour marge), on échantillonne hors masque
        card_mask = np.zeros((h, w), dtype=np.uint8)
        for (x0, y0, x1, y1) in boxes:
            cv2.rectangle(card_mask, (x0, y0), (x1, y1), 255, -1)
        card_mask = cv2.dilate(card_mask, np.ones((81, 81), np.uint8))

        tries = 0
        saved = 0
        while saved < per_img and tries < per_img * 12:
            tries += 1
            px = int(rng.integers(0, w - patch))
            py = int(rng.integers(0, h - patch))
            if card_mask[py:py + patch, px:px + patch].mean() > 5:
                continue  # chevauche une carte
            crop = img[py:py + patch, px:px + patch]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            median_v = float(np.median(hsv[..., 2]))
            median_s = float(np.median(hsv[..., 1]))
            if median_v > v_thresh and median_s < 40:
                cv2.imwrite(str(out_dir / "white" / f"{img_path.stem}_{saved}.png"), crop)
                n_white += 1
            else:
                cv2.imwrite(str(out_dir / "noisy" / f"{img_path.stem}_{saved}.png"), crop)
                n_noisy += 1
            saved += 1
    logger.info("Fonds : %d white, %d noisy -> %s", n_white, n_noisy, out_dir)
    return n_white, n_noisy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/v3_synthetic.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())

    n_cards = extract_cards(cfg)
    n_white, n_noisy = extract_backgrounds(cfg)
    print(f"\nAssets : {n_cards} cartes RGBA | {n_white} fonds blancs | {n_noisy} fonds noisy")
    if n_cards < 54:
        logger.warning("Seulement %d/54 cartes — vérifie card_templates", n_cards)
    if n_white + n_noisy < 80:
        logger.warning("Peu de fonds (%d) — augmente bg_patches_per_image", n_white + n_noisy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
