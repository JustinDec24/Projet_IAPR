"""Phase 3 V3 — Dataset du détecteur de coins.

Deux sources :
  - synthétique : data/synthetic/{images,labels} (coins exacts par construction)
  - réel       : data/train_images + data/yolo_annotations (bboxes axis-aligned
                 → on génère 2 coins TL/BR de chaque bbox)

Le ratio synth/réel par batch est piloté par `real_ratio`.
Cibles générées :
  - heatmap gaussienne (1, H/4, W/4), pixel central forcé à 1.0
  - offsets (2, H/4, W/4) = position sous-pixel, masqués aux pixels positifs
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .corner_detector import DOWNSAMPLE
from .real_corner_extract import load_real_corners

INPUT_W = 512
INPUT_H = 384
GRID_W = INPUT_W // DOWNSAMPLE
GRID_H = INPUT_H // DOWNSAMPLE


def _gaussian(shape: tuple[int, int], cx: float, cy: float, sigma: float
              ) -> np.ndarray:
    h, w = shape
    ys, xs = np.ogrid[:h, :w]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


class CornerDataset(Dataset):
    """Args:
        synth_img/synth_lbl : dossiers synthétiques
        real_img/real_ann   : dossiers réels (images + yolo)
        real_ratio          : proba de tirer un sample réel
        augment             : flip/jitter/perspective
        n_samples           : taille virtuelle d'epoch
    """

    def __init__(self, synth_img: Path, synth_lbl: Path,
                 real_img: Path, real_ann: Path,
                 real_ratio: float = 0.30, augment: bool = True,
                 n_samples: int = 4000, seed: int = 42) -> None:
        self.synth = sorted(Path(synth_img).glob("*.jpg"))
        self.synth_lbl = Path(synth_lbl)
        self.real = sorted(Path(real_img).glob("*.jpg"))
        self.real_ann = Path(real_ann)
        self.real_ratio = real_ratio
        self.augment = augment
        self.n_samples = n_samples
        self.seed = seed
        if not self.synth:
            raise FileNotFoundError(f"Pas de synth dans {synth_img}")

    def __len__(self) -> int:
        return self.n_samples

    # ---- chargement des coins en pixels (repère image originale) ----
    def _load_synth(self, idx: int, rng: random.Random
                    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        p = self.synth[rng.randrange(len(self.synth))]
        img = cv2.imread(str(p))
        lbl = (self.synth_lbl / f"{p.stem}.txt").read_text().strip()
        pts = []
        for line in lbl.split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            pts.append((float(parts[1]), float(parts[2])))
        return img, pts

    def _load_real(self, rng: random.Random
                   ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        p = self.real[rng.randrange(len(self.real))]
        img = cv2.imread(str(p))
        ann = self.real_ann / f"{p.stem}.txt"
        pts: list[tuple[float, float]] = []
        if ann.exists():
            lines = [ln for ln in ann.read_text().strip().split("\n") if ln.strip()]
            # Extraction rotation-aware (rotated rect via couleur dominante),
            # bien plus précise que l'approx axis-aligned pour les cartes tournées.
            pts = load_real_corners(img, lines)
        return img, pts

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        wk = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + idx + (wk.id * 99991 if wk else 0))
        np_rng = np.random.default_rng(self.seed + idx + (wk.id * 99991 if wk else 0))

        use_real = bool(self.real) and rng.random() < self.real_ratio
        img, pts = self._load_real(rng) if use_real else self._load_synth(idx, rng)
        h0, w0 = img.shape[:2]

        # Resize → INPUT_W × INPUT_H (les coins suivent l'échelle)
        sx, sy = INPUT_W / w0, INPUT_H / h0
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        pts = [(x * sx, y * sy) for (x, y) in pts]

        if self.augment:
            if rng.random() < 0.5:  # flip horizontal
                img = cv2.flip(img, 1)
                pts = [(INPUT_W - 1 - x, y) for (x, y) in pts]
            # color jitter HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
            hsv[..., 1] *= rng.uniform(0.75, 1.25)
            hsv[..., 2] *= rng.uniform(0.75, 1.25)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                               cv2.COLOR_HSV2BGR)
            if rng.random() < 0.25:
                img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tens = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        hm = np.zeros((GRID_H, GRID_W), np.float32)
        off = np.zeros((2, GRID_H, GRID_W), np.float32)
        mask = np.zeros((GRID_H, GRID_W), np.float32)
        for (x, y) in pts:
            gx, gy = x / DOWNSAMPLE, y / DOWNSAMPLE
            gxi, gyi = int(gx), int(gy)
            if not (0 <= gxi < GRID_W and 0 <= gyi < GRID_H):
                continue
            hm = np.maximum(hm, _gaussian((GRID_H, GRID_W), gx, gy, sigma=2.0))
            hm[gyi, gxi] = 1.0
            off[0, gyi, gxi] = gx - gxi
            off[1, gyi, gxi] = gy - gyi
            mask[gyi, gxi] = 1.0

        return {
            "image": tens,
            "heatmap": torch.from_numpy(hm).unsqueeze(0),
            "offset": torch.from_numpy(off),
            "mask": torch.from_numpy(mask).unsqueeze(0),
        }
