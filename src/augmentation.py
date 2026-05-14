"""Pipeline d'augmentation pour générer des variantes synthétiques de templates.

Objectif : simuler ce que le classifieur verra à l'inférence, c'est-à-dire des
crops warpés issus de la détection de cartes — pas des scènes complètes. Les
augmentations introduisent du bruit, du flou, des tilts résiduels et des
distorsions géométriques que la chaîne de détection peut produire.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugConfig:
    # Rotation discrète (les cartes UNO ont 2 orientations équivalentes : 0° et 180°)
    base_rotations: tuple[int, ...] = (0, 90, 180, 270)
    # Petit tilt après rotation discrète
    small_rotation_deg: float = 15.0
    # Distorsion de perspective (px de bruit sur chaque coin)
    perspective_jitter_px: float = 12.0
    perspective_prob: float = 0.7
    # Échelle (avant translation)
    scale_min: float = 0.88
    scale_max: float = 1.10
    # Translation en px (post-warp)
    translate_px: float = 10.0
    # Photométrie — élargi pour mieux couvrir les conditions réelles
    brightness_delta: float = 50.0
    contrast_factor: tuple[float, float] = (0.65, 1.35)
    saturation_factor: tuple[float, float] = (0.55, 1.45)
    hue_delta_deg: float = 8.0
    # Flou gaussien
    blur_prob: float = 0.5
    blur_sigma_max: float = 2.0
    # Bruit gaussien
    noise_prob: float = 0.6
    noise_sigma_max: float = 14.0
    # Fringe de fond (bg leakage) : simule les crops de détection imparfaite où
    # un peu de fond (blanc ou bruité) déborde autour de la carte.
    bg_fringe_prob: float = 0.6
    bg_fringe_min_visible: float = 0.82
    bg_fringe_max_visible: float = 0.98
    # Partial crop : simule un détecteur qui clip la carte. Au moins un des 2
    # coins-digits (TL ou BR) est préservé pour garder l'image identifiable.
    partial_crop_prob: float = 0.35
    partial_crop_min_keep: float = 0.40  # garde au moins 40% de la dim coupée


def augment(template: np.ndarray, rng: np.random.Generator, cfg: AugConfig | None = None) -> np.ndarray:
    """Applique une chaîne d'augmentations à un template (BGR, H×W×3) et renvoie un crop de même taille."""
    cfg = cfg or AugConfig()
    img = template.copy()
    h, w = img.shape[:2]

    # 1. Rotation discrète (multiple de 90°)
    base_rot = int(rng.choice(cfg.base_rotations))
    if base_rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif base_rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif base_rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]

    # 2. Affine : tilt + scale + translation
    angle = float(rng.uniform(-cfg.small_rotation_deg, cfg.small_rotation_deg))
    scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
    tx = float(rng.uniform(-cfg.translate_px, cfg.translate_px))
    ty = float(rng.uniform(-cfg.translate_px, cfg.translate_px))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. Distorsion de perspective (probabiliste)
    if rng.random() < cfg.perspective_prob:
        j = cfg.perspective_jitter_px
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst = src + rng.uniform(-j, j, src.shape).astype(np.float32)
        Mp = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, Mp, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 4. Photométrie en HSV (hue + saturation), puis brightness/contrast en BGR
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = float(rng.uniform(-cfg.hue_delta_deg, cfg.hue_delta_deg))
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] *= float(rng.uniform(*cfg.saturation_factor))
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    contrast = float(rng.uniform(*cfg.contrast_factor))
    brightness = float(rng.uniform(-cfg.brightness_delta, cfg.brightness_delta))
    img = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    # 5. Flou (probabiliste)
    if rng.random() < cfg.blur_prob:
        sigma = float(rng.uniform(0.3, cfg.blur_sigma_max))
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

    # 6. Bruit gaussien (probabiliste)
    if rng.random() < cfg.noise_prob:
        sigma = float(rng.uniform(2.0, cfg.noise_sigma_max))
        noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 7. Fringe de fond : la carte est rétrécie et placée sur un canvas de fond
    # (blanc nuancé ou bruité multicolore) pour simuler des crops imparfaits.
    if rng.random() < cfg.bg_fringe_prob:
        img = _apply_bg_fringe(img, rng, cfg)

    # 8. Partial crop : simule un détecteur qui clip 1-2 bords de la carte tout
    # en préservant au moins le coin TL OU le coin BR (où se trouve un digit).
    if rng.random() < cfg.partial_crop_prob:
        img = _apply_partial_crop(img, rng, cfg)

    return img


def _apply_partial_crop(img: np.ndarray, rng: np.random.Generator,
                        cfg: AugConfig) -> np.ndarray:
    """Crop 1-2 bords de la carte en gardant 1 coin avec digit (TL ou BR).

    Le résultat fait toujours la même taille que l'image originale : la zone
    croppée est remplacée par du fond synthétique (pour simuler ce que le
    détecteur produirait quand le bbox déborde sur le fond).
    """
    h, w = img.shape[:2]
    keep_corner = "TL" if rng.random() < 0.5 else "BR"
    min_keep = cfg.partial_crop_min_keep
    # Choisir des fractions de crop (0 = pas de crop, 1-min_keep = max crop)
    max_crop = 1.0 - min_keep
    crop_a = float(rng.uniform(0.0, max_crop))
    crop_b = float(rng.uniform(0.0, max_crop))

    bg = _synthesize_bg(h, w, rng)
    if keep_corner == "TL":
        # On crop le bas et la droite → garde le haut-gauche
        keep_h = max(int(h * min_keep), int(h * (1 - crop_a)))
        keep_w = max(int(w * min_keep), int(w * (1 - crop_b)))
        bg[:keep_h, :keep_w] = img[:keep_h, :keep_w]
    else:
        # On crop le haut et la gauche → garde le bas-droit
        start_h = min(int(h * (1 - min_keep)), int(h * crop_a))
        start_w = min(int(w * (1 - min_keep)), int(w * crop_b))
        bg[start_h:, start_w:] = img[start_h:, start_w:]
    return bg


def _synthesize_bg(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Génère un patch de fond aléatoire (blanc nuancé OU bruité multicolore)."""
    if rng.random() < 0.5:
        # Fond blanc-ish nuancé (variations de luminosité + bruit fin)
        base = float(rng.uniform(200, 245))
        bg = np.full((h, w, 3), base, dtype=np.float32)
        bg += rng.normal(0.0, 6.0, bg.shape)
        return np.clip(bg, 0, 255).astype(np.uint8)
    # Fond "feuillage" : disques de couleurs aléatoires saturées, puis flou
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(int(rng.integers(3, 8))):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(15, max(16, max(h, w) // 3)))
        color = tuple(int(c) for c in rng.integers(20, 220, 3))
        cv2.circle(bg, (cx, cy), r, color, -1)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=8.0)
    return bg


def _apply_bg_fringe(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    """Rétrécit la carte au sein du frame et remplit autour avec un fond synthétique."""
    h, w = img.shape[:2]
    visible = float(rng.uniform(cfg.bg_fringe_min_visible, cfg.bg_fringe_max_visible))
    new_h = max(1, int(h * visible))
    new_w = max(1, int(w * visible))
    card = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = _synthesize_bg(h, w, rng)
    off_y = int(rng.integers(0, max(1, h - new_h + 1)))
    off_x = int(rng.integers(0, max(1, w - new_w + 1)))
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = card
    return canvas
