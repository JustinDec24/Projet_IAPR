"""Phase 2b V3 — Génération de scènes synthétiques avec annotations COINS.

Chaque carte UNO a son digit imprimé en haut-gauche ET bas-droite. On annote
ces 2 coins (et leur visibilité, calculée via un owner-map des cartes empilées).

Sortie :
  data/synthetic/images/scene_NNNNN.jpg
  data/synthetic/labels/scene_NNNNN.txt   (1 ligne / coin visible)
    format : <class> <corner_x> <corner_y> <is_occluded> <crop_size>
             coords en pixels, is_occluded ∈ {0,1}, crop_size = côté carré
             (px) de la fenêtre de crop digit+couleur autour du point.

Usage :
    python -m src.synthetic.generate_scenes --config configs/v3_synthetic.yaml \
        --n 5000
"""
from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_scenes")

ROOT = Path(__file__).resolve().parents[2]

# Centre du digit-coin UNO dans le repère carte normalisé (w,h template).
# Le petit chiffre/symbole de coin est centré ~14% du bord.
CORNER_TL = (0.14, 0.14)   # (fx, fy) haut-gauche
CORNER_BR = (0.86, 0.86)   # bas-droite
# Le crop digit+couleur fait ~42% de la largeur carte (contient le chiffre +
# une marge de couleur, indispensable au classifieur couleur+valeur).
CROP_FRAC = 0.42
OCC_WIN = 26               # demi-fenêtre (px) pour mesurer l'occlusion d'un coin


def _load_cards(cards_dir: Path) -> dict[str, np.ndarray]:
    cards = {}
    for p in sorted(cards_dir.glob("*.png")):
        rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.shape[2] == 4:
            cards[p.stem] = rgba
    return cards


def _load_bgs(bg_dir: Path) -> tuple[list[Path], list[Path]]:
    white = sorted((bg_dir / "white").glob("*.png"))
    noisy = sorted((bg_dir / "noisy").glob("*.png"))
    return white, noisy


def _make_background(white: list[Path], noisy: list[Path],
                     W: int, H: int, rng: random.Random) -> np.ndarray:
    pool = white if rng.random() < 0.5 and white else (noisy or white)
    patch = cv2.imread(str(rng.choice(pool)))
    ph, pw = patch.shape[:2]
    scale = max(W / pw, H / ph) * rng.uniform(1.0, 1.5)
    big = cv2.resize(patch, (int(pw * scale), int(ph * scale)))
    bh, bw = big.shape[:2]
    x0 = rng.randint(0, max(0, bw - W))
    y0 = rng.randint(0, max(0, bh - H))
    return big[y0:y0 + H, x0:x0 + W].copy()


def _transform_card(card_rgba: np.ndarray, target_w: int, angle_deg: float
                    ) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """Resize + rotation. Renvoie (rgb, alpha, corner_TL_xy, corner_BR_xy)
    dans le repère du patch transformé."""
    h0, w0 = card_rgba.shape[:2]
    target_h = int(target_w * h0 / w0)
    card = cv2.resize(card_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
    # Points de coin AVANT rotation (repère carte resizée)
    pts = np.array([
        [CORNER_TL[0] * target_w, CORNER_TL[1] * target_h],
        [CORNER_BR[0] * target_w, CORNER_BR[1] * target_h],
    ], dtype=np.float32)

    M = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(target_h * sin_a + target_w * cos_a)
    nh = int(target_h * cos_a + target_w * sin_a)
    M[0, 2] += nw / 2 - target_w / 2
    M[1, 2] += nh / 2 - target_h / 2
    rot = cv2.warpAffine(card, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))
    rgb = rot[..., :3]
    alpha = rot[..., 3]
    pts_h = np.hstack([pts, np.ones((2, 1), dtype=np.float32)])
    new_pts = (M @ pts_h.T).T  # (2,2)
    return rgb, alpha, tuple(new_pts[0]), tuple(new_pts[1])


def _zone_boxes(cfg: dict, W: int, H: int) -> dict[str, tuple[int, int, int, int]]:
    z = {}
    for name, (x0, y0, x1, y1) in cfg["scene"]["zones"].items():
        z[name] = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
    return z


def generate_one(idx: int, cards: dict, white: list[Path], noisy: list[Path],
                  cfg: dict, rng: random.Random
                  ) -> tuple[np.ndarray, list[tuple]]:
    sc = cfg["scene"]
    W, H = sc["width"], sc["height"]
    scene = _make_background(white, noisy, W, H, rng)
    owner = np.full((H, W), -1, dtype=np.int32)  # index carte du dessus
    zones = _zone_boxes(cfg, W, H)
    card_names = list(cards.keys())

    placements: list[dict] = []  # {label, corners:[(x,y),(x,y)]}
    n_cards = rng.randint(sc["cards_min"], sc["cards_max"])

    # Répartit les cartes dans les zones (centre = 0 ou 1 carte)
    zone_names = ["p1", "p2", "p3", "p4"]
    assignments: list[str] = []
    if rng.random() < 0.85:
        assignments.append("center")
    while len(assignments) < n_cards:
        assignments.append(rng.choice(zone_names))

    card_idx = 0
    for zone in assignments:
        zx0, zy0, zx1, zy1 = zones[zone]
        cw = rng.randint(*sc["card_width_px"])
        if zone == "center":
            cw = int(cw * 1.15)
        # Empilement éventuel
        is_stack = zone != "center" and rng.random() < sc["stack_prob"]
        stack_n = rng.randint(*sc["stack_size"]) if is_stack else 1
        base_x = rng.randint(zx0 + cw // 2, max(zx0 + cw // 2 + 1, zx1 - cw // 2))
        base_y = rng.randint(zy0 + cw // 2, max(zy0 + cw // 2 + 1, zy1 - cw // 2))
        for s in range(stack_n):
            if card_idx >= n_cards:
                break
            name = rng.choice(card_names)
            angle = rng.uniform(-sc["rotation_deg"], sc["rotation_deg"])
            rgb, alpha, c_tl, c_br = _transform_card(cards[name], cw, angle)
            ph, pw = alpha.shape
            off = 0 if s == 0 else rng.randint(*sc["stack_offset_px"])
            cx = int(np.clip(base_x + off * (s) - pw // 2, 0, W - pw - 1))
            cy = int(np.clip(base_y + off * (s) - ph // 2, 0, H - ph - 1))
            if cx < 0 or cy < 0 or cx + pw >= W or cy + ph >= H:
                continue
            region = scene[cy:cy + ph, cx:cx + pw]
            a = (alpha.astype(np.float32) / 255.0)[..., None]
            # Ombre douce
            sh_off = sc["shadow_offset_px"]
            sy0, sx0 = cy + sh_off, cx + sh_off
            if sy0 + ph < H and sx0 + pw < W:
                shadow = (alpha > 30).astype(np.uint8) * 255
                shadow = cv2.GaussianBlur(shadow, (0, 0), sc["shadow_blur"])
                sa = (shadow.astype(np.float32) / 255.0 * sc["shadow_opacity"])[..., None]
                sreg = scene[sy0:sy0 + ph, sx0:sx0 + pw]
                scene[sy0:sy0 + ph, sx0:sx0 + pw] = (sreg * (1 - sa)).astype(np.uint8)
            scene[cy:cy + ph, cx:cx + pw] = (region * (1 - a) + rgb * a).astype(np.uint8)
            mask_bool = alpha > 30
            owner[cy:cy + ph, cx:cx + pw][mask_bool] = card_idx
            placements.append({
                "idx": card_idx, "label": name,
                "crop_size": int(round(cw * CROP_FRAC)),
                "corners": [(cx + c_tl[0], cy + c_tl[1]),
                            (cx + c_br[0], cy + c_br[1])],
            })
            card_idx += 1

    # Calcul de visibilité des coins via owner-map
    annotations: list[tuple] = []
    for pl in placements:
        for (cxp, cyp) in pl["corners"]:
            xi, yi = int(round(cxp)), int(round(cyp))
            if not (0 <= xi < W and 0 <= yi < H):
                continue
            x0 = max(0, xi - OCC_WIN)
            x1 = min(W, xi + OCC_WIN)
            y0 = max(0, yi - OCC_WIN)
            y1 = min(H, yi + OCC_WIN)
            win = owner[y0:y1, x0:x1]
            total = win.size
            mine = int((win == pl["idx"]).sum())
            frac_visible = mine / max(total, 1)
            if frac_visible < 0.10:
                continue  # coin totalement masqué -> on ne l'annote pas
            is_occ = int(frac_visible < cfg["scene"]["occlusion_visibility_thresh"])
            annotations.append((pl["label"], xi, yi, is_occ, pl["crop_size"]))

    return scene, annotations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/v3_synthetic.yaml")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    n_scenes = args.n if args.n is not None else cfg["scene"]["n_scenes"]

    cards = _load_cards(ROOT / cfg["assets"]["out_cards_dir"])
    white, noisy = _load_bgs(ROOT / cfg["assets"]["out_bg_dir"])
    logger.info("Assets : %d cartes, %d white, %d noisy", len(cards), len(white), len(noisy))
    if not cards or not (white or noisy):
        logger.error("Assets manquants — lance extract_assets d'abord")
        return 1

    img_dir = ROOT / cfg["scene"]["out_images_dir"]
    lbl_dir = ROOT / cfg["scene"]["out_labels_dir"]
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    base_seed = cfg["seed"]
    n_corners_total = 0
    for i in range(n_scenes):
        rng = random.Random(base_seed + i)
        np.random.seed(base_seed + i)
        scene, ann = generate_one(i, cards, white, noisy, cfg, rng)
        sid = f"scene_{i:05d}"
        cv2.imwrite(str(img_dir / f"{sid}.jpg"), scene, [cv2.IMWRITE_JPEG_QUALITY, 88])
        lines = [f"{lbl} {x} {y} {o} {cs}" for (lbl, x, y, o, cs) in ann]
        (lbl_dir / f"{sid}.txt").write_text("\n".join(lines))
        n_corners_total += len(ann)
        if (i + 1) % 500 == 0:
            logger.info("Généré %d/%d scènes", i + 1, n_scenes)

    logger.info("Terminé : %d scènes, %d coins annotés (moy %.1f/scène)",
                n_scenes, n_corners_total, n_corners_total / max(n_scenes, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
