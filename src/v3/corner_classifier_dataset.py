"""Phase 4 V3 — Dataset du corner-classifier (54 classes).

3 sources de coins labellisés :
  A. Templates : coins TL+BR des 54 cartes de référence (signal canonique propre)
  B. Synthétique : crops `crop_size` autour des coins annotés des 5000 scènes
     (riche : fonds, rotations, occlusions)
  C. Réel : coins extraits des crops réels labellisés data/real_crops/<label>/

Échantillonnage équilibré par classe (chaque classe tirée ~uniformément).
Augmentations agressives SANS flip (6 vs 9 distincts !).
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CLASS_TO_IDX, NUM_CLASSES
from src.v3.corner_classifier import INPUT_SIZE

# Position du digit-coin dans le template canonique 200×300 (fx, fy)
TPL_TL = (0.14, 0.14)
TPL_BR = (0.86, 0.86)
TPL_CROP_FRAC = 0.42


def _crop_around(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = size // 2
    x0, y0 = int(cx - half), int(cy - half)
    x1, y1 = x0 + size, y0 + size
    # pad si déborde
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - w), max(0, y1 - h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    crop = img[y0:y1, x0:x1]
    if px0 or py0 or px1 or py1:
        crop = cv2.copyMakeBorder(crop, py0, py1, px0, px1,
                                  cv2.BORDER_REPLICATE)
    return crop


class CornerClassifierDataset(Dataset):
    def __init__(self, root: Path, augment: bool = True,
                 n_samples: int = 30000, seed: int = 42,
                 include_occluded: bool = True) -> None:
        self.root = Path(root)
        self.augment = augment
        self.n_samples = n_samples
        self.seed = seed

        # index : label -> liste d'items (type, payload)
        self.by_class: dict[str, list[tuple]] = defaultdict(list)

        # --- Source A : templates ---
        tpl_dir = self.root / "data" / "card_templates"
        for p in sorted(tpl_dir.glob("*.png")):
            lbl = p.stem
            if lbl in CLASS_TO_IDX:
                self.by_class[lbl].append(("tpl", str(p)))

        # --- Source B : synthétique ---
        syn_img = self.root / "data" / "synthetic" / "images"
        syn_lbl = self.root / "data" / "synthetic" / "labels"
        for lf in sorted(syn_lbl.glob("*.txt")):
            img_path = syn_img / f"{lf.stem}.jpg"
            if not img_path.exists():
                continue
            for line in lf.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                lbl = parts[0]
                if lbl not in CLASS_TO_IDX:
                    continue
                x, y, occ = int(parts[1]), int(parts[2]), int(parts[3])
                cs = int(parts[4]) if len(parts) > 4 else 50
                if occ and not include_occluded:
                    continue
                self.by_class[lbl].append(("syn", (str(img_path), x, y, cs)))

        # --- Source C : real_crops ---
        rc_dir = self.root / "data" / "real_crops"
        if rc_dir.exists():
            for ld in sorted(rc_dir.iterdir()):
                if ld.is_dir() and ld.name in CLASS_TO_IDX:
                    for p in sorted(ld.glob("*.png")):
                        self.by_class[ld.name].append(("rc", str(p)))

        self.classes = sorted(self.by_class.keys())
        if not self.classes:
            raise RuntimeError("Aucun coin labellisé trouvé")

    def __len__(self) -> int:
        return self.n_samples

    def _augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape[:2]
        # rotation ±15° (PAS de flip)
        ang = float(rng.uniform(-15, 15))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # perspective légère
        if rng.random() < 0.4:
            j = 0.05 * w
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = src + rng.uniform(-j, j, src.shape).astype(np.float32)
            img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                                      (w, h), borderMode=cv2.BORDER_REPLICATE)
        # color jitter HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-7, 7)) % 180
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        hsv[..., 2] *= rng.uniform(0.7, 1.3)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        # blur léger occasionnel
        if rng.random() < 0.2:
            img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.4, 1.3)))
        # cutout (≤25% surface)
        if rng.random() < 0.5:
            cs = int(rng.uniform(0.10, 0.25) * w)
            cx = int(rng.uniform(0, w - cs)); cy = int(rng.uniform(0, h - cs))
            img[cy:cy + cs, cx:cx + cs] = int(rng.uniform(0, 255))
        return img

    def _get_corner(self, item: tuple, rng: np.random.Generator) -> np.ndarray:
        kind, payload = item
        if kind == "tpl":
            tpl = cv2.imread(payload)
            h, w = tpl.shape[:2]
            fx, fy = (TPL_TL if rng.random() < 0.5 else TPL_BR)
            sz = int(TPL_CROP_FRAC * w)
            return _crop_around(tpl, fx * w, fy * h, sz)
        if kind == "syn":
            img_path, x, y, cs = payload
            img = cv2.imread(img_path)
            return _crop_around(img, x, y, cs)
        # rc : crop réel canonique 200×300
        rc = cv2.imread(payload)
        h, w = rc.shape[:2]
        fx, fy = (TPL_TL if rng.random() < 0.5 else TPL_BR)
        sz = int(TPL_CROP_FRAC * w)
        return _crop_around(rc, fx * w, fy * h, sz)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        wk = torch.utils.data.get_worker_info()
        seed = self.seed + idx + (wk.id * 99991 if wk else 0)
        rng = np.random.default_rng(seed)
        # tirage équilibré : classe uniforme puis instance aléatoire
        lbl = self.classes[int(rng.integers(0, len(self.classes)))]
        items = self.by_class[lbl]
        item = items[int(rng.integers(0, len(items)))]
        crop = self._get_corner(item, rng)
        if crop is None or crop.size == 0:
            crop = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        if self.augment:
            crop = self._augment(crop, rng)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t, CLASS_TO_IDX[lbl]
