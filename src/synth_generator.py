"""Génère un dataset synthétique de scènes UNO depuis assets internes.

Composition légale au sens du règlement (assets internes uniquement) :
- Cartes : depuis `data/card_templates/` (54 templates extraits des reference_images)
- Fonds : depuis `data/bg_patches/` (1215 patches extraits des train_images)
- Aucune donnée externe.

Pour chaque scène générée :
- 5-13 cartes placées avec rotation libre, scale variable, occlusions partielles
- Background composé d'un bg_patch redimensionné + crops aléatoires
- Annotations OBB exactes par construction (cx, cy, w, h, θ) dans `<id>.txt`
- Augmentations photométriques light (brightness, color jitter) appliquées après composition

Format annotations YOLO étendu :
    class_idx cx_norm cy_norm w_norm h_norm angle_deg

(class_idx = 0 puisque c'est 1 seule classe "card", comme le dataset réel)
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from .config import CARD_CLASSES

# Cibles de résolution (3:2 aspect ratio, comme les vraies images 4000×2662)
SCENE_W = 2000
SCENE_H = 1333

# Cartes : taille en pixels (échantillonnée uniformément dans cette range)
CARD_W_MIN = 110
CARD_W_MAX = 220
CARD_ASPECT = 1.5  # h/w (cartes UNO sont portrait 2:3)

# Nombre de cartes par scène
N_CARDS_MIN = 4
N_CARDS_MAX = 14

# Stack probability : proba qu'une carte placée se trouve PARTIELLEMENT
# occultée par celle d'avant (i.e. on superpose)
STACK_PROBA = 0.35
STACK_MAX_OVERLAP = 0.55  # max 55% d'occlusion pour rester "détectable"


def _load_assets(card_dir: Path, bg_dir: Path) -> tuple[dict, list]:
    templates: dict[str, np.ndarray] = {}
    for c in CARD_CLASSES:
        p = card_dir / f"{c}.png"
        if not p.exists():
            continue
        templates[c] = cv2.imread(str(p))
    bg_paths = sorted(bg_dir.glob("*.png"))
    return templates, bg_paths


def _generate_background(bg_paths: list[Path], rng: random.Random) -> np.ndarray:
    """Tile + scale random bg patches to form a SCENE_W×SCENE_H background."""
    # Option 1 (60%): un seul patch agrandi
    # Option 2 (40%): tile de 4 patches en 2×2
    if rng.random() < 0.6:
        path = rng.choice(bg_paths)
        patch = cv2.imread(str(path))
        # Scale up to cover SCENE_W×SCENE_H
        ph, pw = patch.shape[:2]
        scale = max(SCENE_W / pw, SCENE_H / ph) * rng.uniform(1.0, 1.6)
        big = cv2.resize(patch, (int(pw * scale), int(ph * scale)),
                         interpolation=cv2.INTER_LINEAR)
        bh, bw = big.shape[:2]
        x0 = rng.randint(0, max(0, bw - SCENE_W))
        y0 = rng.randint(0, max(0, bh - SCENE_H))
        return big[y0:y0 + SCENE_H, x0:x0 + SCENE_W].copy()
    else:
        # Tile 2×2
        tw, th = SCENE_W // 2, SCENE_H // 2
        bg = np.zeros((SCENE_H, SCENE_W, 3), dtype=np.uint8)
        for i in range(2):
            for j in range(2):
                path = rng.choice(bg_paths)
                patch = cv2.imread(str(path))
                resized = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_LINEAR)
                bg[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = resized
        return bg


def _rotate_card(card: np.ndarray, angle_deg: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate card around its center. Renvoie (carte_rotated, mask_rotated)."""
    h, w = card.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    rotated = cv2.warpAffine(card, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.ones((h, w), dtype=np.uint8) * 255, M,
                          (new_w, new_h),
                          flags=cv2.INTER_NEAREST,
                          borderValue=0)
    return rotated, mask


def _place_card(scene: np.ndarray, card: np.ndarray, mask: np.ndarray,
                cx: int, cy: int) -> bool:
    """Composite card onto scene at (cx, cy). Returns True if placed."""
    H, W = scene.shape[:2]
    h, w = card.shape[:2]
    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = x0 + w
    y1 = y0 + h
    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return False
    region = scene[y0:y1, x0:x1]
    m = mask[..., None] / 255.0
    scene[y0:y1, x0:x1] = (region * (1 - m) + card * m).astype(np.uint8)
    return True


def _photometric_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Brightness + small color jitter."""
    out = image.astype(np.float32)
    # Brightness
    brightness = rng.uniform(0.7, 1.3)
    out = np.clip(out * brightness, 0, 255)
    # Color shift
    shift = np.array([rng.uniform(-15, 15) for _ in range(3)], dtype=np.float32)
    out = np.clip(out + shift, 0, 255)
    # Light noise
    if rng.random() < 0.4:
        noise = rng.gauss(0, 1)
        noise_arr = np.random.normal(0, 5, image.shape).astype(np.float32)
        out = np.clip(out + noise_arr, 0, 255)
    return out.astype(np.uint8)


def generate_scene(templates: dict[str, np.ndarray],
                   bg_paths: list[Path],
                   rng: random.Random,
                   ) -> tuple[np.ndarray, list[tuple]]:
    """Génère une scène : (image, list de (cx, cy, w, h, angle) en coords absolues)."""
    bg = _generate_background(bg_paths, rng)

    n_cards = rng.randint(N_CARDS_MIN, N_CARDS_MAX)
    placed: list[tuple[int, int, int, int]] = []  # (cx, cy, w, h)
    annotations: list[tuple] = []

    card_names = list(templates.keys())
    attempts = 0
    while len(annotations) < n_cards and attempts < n_cards * 8:
        attempts += 1
        name = rng.choice(card_names)
        template = templates[name]
        # Target card width (portrait orientation, w < h)
        card_w = rng.randint(CARD_W_MIN, CARD_W_MAX)
        card_h = int(card_w * CARD_ASPECT)
        resized = cv2.resize(template, (card_w, card_h),
                             interpolation=cv2.INTER_LINEAR)
        angle = rng.uniform(0, 360)
        rotated, mask = _rotate_card(resized, angle)

        # Pick position with overlap constraint
        margin = max(rotated.shape[:2]) // 2
        if margin + 10 >= SCENE_W // 2 or margin + 10 >= SCENE_H // 2:
            continue
        cx = rng.randint(margin + 10, SCENE_W - margin - 10)
        cy = rng.randint(margin + 10, SCENE_H - margin - 10)

        # Overlap check with existing cards
        if placed and rng.random() > STACK_PROBA:
            too_close = False
            for (pcx, pcy, pw, ph) in placed:
                # Approximate: rectangles overlap if centers closer than half-sums
                dx = abs(cx - pcx)
                dy = abs(cy - pcy)
                if dx < (card_w + pw) / 2 * 0.85 and dy < (card_h + ph) / 2 * 0.85:
                    too_close = True
                    break
            if too_close:
                continue

        if not _place_card(bg, rotated, mask, cx, cy):
            continue

        placed.append((cx, cy, card_w, card_h))
        annotations.append((float(cx), float(cy), float(card_w),
                            float(card_h), float(angle), name))

    bg = _photometric_jitter(bg, rng)
    return bg, annotations


def save_annotations(annotations: list[tuple], path: Path,
                     img_w: int = SCENE_W, img_h: int = SCENE_H) -> None:
    """Sauve au format YOLO étendu : class_idx cx_norm cy_norm w_norm h_norm angle_deg."""
    lines = []
    for (cx, cy, w, h, angle, _name) in annotations:
        lines.append(
            f"0 {cx / img_w:.6f} {cy / img_h:.6f} "
            f"{w / img_w:.6f} {h / img_h:.6f} {angle:.2f}"
        )
    path.write_text("\n".join(lines))
