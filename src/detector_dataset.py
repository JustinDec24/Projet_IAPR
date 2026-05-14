"""Dataset PyTorch pour entraîner le détecteur de cartes.

Charge les 81 images annotées + leurs bboxes YOLO. Applique :
- Resize à (INPUT_H, INPUT_W) avec préservation aspect
- Augmentations : flip horizontal, scale+crop random, color jitter, blur, noise
- Génération des targets :
    * obj_target : heatmap gaussienne (B, 1, H/16, W/16) centrée sur chaque bbox
    * bbox_target : (B, 4, H/16, W/16) = (cx, cy, w, h) normalisés [0, 1]
    * bbox_mask : (B, 1, H/16, W/16) = 1 au pixel central de chaque bbox
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_W = 512
INPUT_H = 352
DOWNSAMPLE = 16
GRID_W = INPUT_W // DOWNSAMPLE  # 32
GRID_H = INPUT_H // DOWNSAMPLE  # 22


def _gaussian_2d(shape: tuple[int, int], cx: float, cy: float, sigma: float) -> np.ndarray:
    """Renvoie une heatmap gaussienne 2D de taille shape, centrée en (cx, cy)."""
    h, w = shape
    y_grid, x_grid = np.ogrid[:h, :w]
    return np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))


class CardDetectionDataset(Dataset):
    """Détection axis-aligned. Accepte multi-sources (synth + real)."""
    def __init__(
        self,
        image_dirs: list[Path] | Path,
        annot_dirs: list[Path] | Path,
        weights: list[float] | None = None,
        augment: bool = True,
        n_samples: int = 1000,
        seed: int = 0,
    ) -> None:
        # Support legacy single-dir argument
        if isinstance(image_dirs, (str, Path)):
            image_dirs = [image_dirs]
            annot_dirs = [annot_dirs]
        self.image_dirs = [Path(d) for d in image_dirs]
        self.annot_dirs = [Path(d) for d in annot_dirs]
        self.augment = augment
        self.n_samples = n_samples
        self.seed = seed

        # Charger les sources
        self.sources: list[tuple[list[Path], list[np.ndarray]]] = []
        for img_dir, ann_dir in zip(self.image_dirs, self.annot_dirs):
            image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            if not image_paths:
                continue
            annotations = []
            for img_path in image_paths:
                annot_path = ann_dir / f"{img_path.stem}.txt"
                if not annot_path.exists():
                    annotations.append(np.zeros((0, 4), dtype=np.float32))
                    continue
                boxes = []
                for line in annot_path.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split()
                    # Support 5 (axis-aligned) ou 6 (avec angle, drop angle) valeurs
                    if len(parts) < 5:
                        continue
                    boxes.append([float(p) for p in parts[1:5]])
                annotations.append(np.array(boxes, dtype=np.float32).reshape(-1, 4))
            self.sources.append((image_paths, annotations))

        if not self.sources:
            raise FileNotFoundError(f"No images found in {image_dirs}")

        # Weights : par défaut proportionnel au nombre d'images par source
        if weights is None:
            counts = [len(s[0]) for s in self.sources]
            total = sum(counts)
            self.weights = [c / total for c in counts]
        else:
            s = sum(weights)
            self.weights = [w / s for w in weights]

        # Backward compat
        self.image_paths = self.sources[0][0]
        self.annotations = self.sources[0][1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        worker = torch.utils.data.get_worker_info()
        seed = self.seed + idx + (worker.id * 1_000_003 if worker else 0)
        rng = np.random.default_rng(seed)

        # Choix de source pondéré
        src_idx = int(rng.choice(len(self.sources), p=self.weights))
        image_paths, annotations = self.sources[src_idx]
        img_idx = int(rng.integers(0, len(image_paths)))
        image = cv2.imread(str(image_paths[img_idx]))
        bboxes = annotations[img_idx].copy()  # (N, 4) normalisé image originale

        h0, w0 = image.shape[:2]

        if self.augment:
            image, bboxes = self._augment(image, bboxes, rng, w0, h0)
        else:
            image = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)

        # Convert image to tensor
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        # Générer les targets sur la grille (downsample 16×)
        obj_target = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        bbox_target = np.zeros((4, GRID_H, GRID_W), dtype=np.float32)
        bbox_mask = np.zeros((GRID_H, GRID_W), dtype=np.float32)

        for cx, cy, w, h in bboxes:
            if w <= 0 or h <= 0 or cx <= 0 or cy <= 0 or cx >= 1 or cy >= 1:
                continue
            # Position sur la grille
            gx = cx * GRID_W
            gy = cy * GRID_H
            gxi = int(np.clip(gx, 0, GRID_W - 1))
            gyi = int(np.clip(gy, 0, GRID_H - 1))

            # Heatmap gaussienne — sigma proportionnel à la taille de la bbox
            sigma = max(1.0, (w * GRID_W + h * GRID_H) / 12.0)
            heatmap = _gaussian_2d((GRID_H, GRID_W), gx, gy, sigma)
            obj_target = np.maximum(obj_target, heatmap)
            # Forcer target=1.0 au pixel central pour que pos_mask sélectionne
            # bien ce pixel dans la focal loss (le max de la gaussienne discrète
            # est sinon < 1.0 quand le centre tombe entre deux cellules).
            obj_target[gyi, gxi] = 1.0

            # Bbox au pixel central
            bbox_target[0, gyi, gxi] = cx
            bbox_target[1, gyi, gxi] = cy
            bbox_target[2, gyi, gxi] = w
            bbox_target[3, gyi, gxi] = h
            bbox_mask[gyi, gxi] = 1.0

        return {
            "image": img_tensor,
            "obj_target": torch.from_numpy(obj_target).unsqueeze(0),       # (1, GH, GW)
            "bbox_target": torch.from_numpy(bbox_target),                  # (4, GH, GW)
            "bbox_mask": torch.from_numpy(bbox_mask).unsqueeze(0),         # (1, GH, GW)
        }

    def _augment(self, image: np.ndarray, bboxes: np.ndarray,
                 rng: np.random.Generator, w0: int, h0: int
                 ) -> tuple[np.ndarray, np.ndarray]:
        # 1. Random scale + crop (data multiplication)
        scale = float(rng.uniform(0.65, 1.0))
        crop_w = int(w0 * scale)
        crop_h = int(h0 * scale)
        x0 = int(rng.uniform(0, w0 - crop_w))
        y0 = int(rng.uniform(0, h0 - crop_h))
        image = image[y0:y0 + crop_h, x0:x0 + crop_w]
        # Ajuster les bboxes au crop
        if bboxes.shape[0] > 0:
            # Convertir bboxes normalisées en absolu (image originale)
            abs_cx = bboxes[:, 0] * w0
            abs_cy = bboxes[:, 1] * h0
            abs_w = bboxes[:, 2] * w0
            abs_h = bboxes[:, 3] * h0
            # Translater par (x0, y0)
            abs_cx -= x0
            abs_cy -= y0
            # Reconvertir en normalisé par rapport au crop
            new_bboxes = np.stack([abs_cx / crop_w, abs_cy / crop_h,
                                   abs_w / crop_w, abs_h / crop_h], axis=1)
            # Garder uniquement bboxes dont le centre est dans le crop
            valid = ((new_bboxes[:, 0] > 0.02) & (new_bboxes[:, 0] < 0.98) &
                     (new_bboxes[:, 1] > 0.02) & (new_bboxes[:, 1] < 0.98))
            bboxes = new_bboxes[valid]

        # 2. Resize to target
        image = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)

        # 3. Random horizontal flip
        if rng.random() < 0.5:
            image = cv2.flip(image, 1)
            if bboxes.shape[0] > 0:
                bboxes[:, 0] = 1.0 - bboxes[:, 0]

        # 4. Color jitter
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = float(rng.uniform(-10, 10))
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        hsv[..., 1] *= float(rng.uniform(0.7, 1.3))
        hsv[..., 2] *= float(rng.uniform(0.7, 1.3))
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 5. Blur / noise (faible probabilité)
        if rng.random() < 0.3:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=float(rng.uniform(0.3, 1.5)))
        if rng.random() < 0.4:
            sigma = float(rng.uniform(2, 10))
            noise = rng.normal(0, sigma, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image, bboxes
