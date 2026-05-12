"""PyTorch Dataset pour la classification de cartes UNO.

Charge les 54 templates + N bg patches en mémoire au démarrage. Chaque
échantillon est une variante augmentée d'un template (54 classes cartes) ou
d'un bg patch (classe « non_card »). Échantillonnage uniforme par classe.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import AugConfig, augment
from .config import CARD_CLASSES, CLASS_TO_IDX, NON_CARD_IDX, NON_CARD_LABEL


class UnoTemplateDataset(Dataset):
    """Dataset synthétique : un échantillon = template aléatoire + augmentation.

    Args:
        templates_dir: dossier contenant les 54 fichiers <class>.png.
        n_samples: nombre d'échantillons par "epoch" (samples par classe × 54).
        input_size: (height, width) cible après resize. Cards sont en portrait
            donc h > w.
        aug_cfg: configuration d'augmentation.
        seed: seed de base ; chaque worker dérive son propre rng.
    """

    def __init__(
        self,
        templates_dir: Path,
        bg_patches_dir: Path | None = None,
        real_crops_dir: Path | None = None,
        real_crop_prob: float = 0.5,
        n_samples: int = 27_000,
        input_size: tuple[int, int] = (144, 96),
        aug_cfg: AugConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.bg_patches_dir = Path(bg_patches_dir) if bg_patches_dir else None
        self.real_crops_dir = Path(real_crops_dir) if real_crops_dir else None
        self.real_crop_prob = float(real_crop_prob)
        self.n_samples = int(n_samples)
        self.input_size = input_size
        self.aug_cfg = aug_cfg or AugConfig()
        self.seed = seed

        # 54 templates de cartes (toujours chargés)
        self.templates: list[np.ndarray] = []
        for label in CARD_CLASSES:
            img = cv2.imread(str(self.templates_dir / f"{label}.png"))
            if img is None:
                raise FileNotFoundError(f"Template manquant : {label}.png")
            self.templates.append(img)

        # Crops réels par classe (auto-labellisés depuis train_images)
        self.real_crops: list[list[np.ndarray]] = [[] for _ in CARD_CLASSES]
        if self.real_crops_dir and self.real_crops_dir.exists():
            for idx, label in enumerate(CARD_CLASSES):
                class_dir = self.real_crops_dir / label
                if not class_dir.exists():
                    continue
                for path in sorted(class_dir.glob("*.png")):
                    img = cv2.imread(str(path))
                    if img is not None:
                        self.real_crops[idx].append(img)
        self.n_real_crops_total = sum(len(c) for c in self.real_crops)

        # Bg patches (classe non_card) — désactivé par défaut, optionnel
        self.bg_patches: list[np.ndarray] = []
        if self.bg_patches_dir and self.bg_patches_dir.exists():
            for path in sorted(self.bg_patches_dir.glob("*.png")):
                img = cv2.imread(str(path))
                if img is not None:
                    self.bg_patches.append(img)

        self.has_non_card = len(self.bg_patches) > 0
        self.n_classes = len(self.templates) + (1 if self.has_non_card else 0)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        worker = torch.utils.data.get_worker_info()
        seed = self.seed + idx + (worker.id * 1_000_003 if worker else 0)
        rng = np.random.default_rng(seed)

        # Échantillonnage uniforme par classe
        class_idx = int(rng.integers(0, self.n_classes))
        if class_idx == NON_CARD_IDX and self.has_non_card:
            patch = self.bg_patches[int(rng.integers(0, len(self.bg_patches)))]
            img = self._augment_bg(patch, rng)
        else:
            # Avec probabilité real_crop_prob, on prend un crop réel si dispo,
            # sinon on retombe sur le template synthétique. Les crops réels sont
            # augmentés plus légèrement (déjà imparfaits, pas besoin de bg fringe).
            use_real = (
                len(self.real_crops[class_idx]) > 0
                and rng.random() < self.real_crop_prob
            )
            if use_real:
                crop = self.real_crops[class_idx][
                    int(rng.integers(0, len(self.real_crops[class_idx])))
                ]
                img = self._augment_real(crop, rng)
            else:
                template = self.templates[class_idx]
                img = augment(template, rng, self.aug_cfg)

        h_target, w_target = self.input_size
        img = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor, class_idx

    def _augment_real(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Augmentation légère pour les crops réels (déjà imparfaits → pas de bg fringe)."""
        img = crop.copy()
        h, w = img.shape[:2]
        # Rotation discrète (les cartes peuvent être dans 4 orientations)
        rot = int(rng.choice([0, 90, 180, 270]))
        if rot == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rot == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img.shape[:2]
        # Petit tilt + scale + translation
        angle = float(rng.uniform(-10, 10))
        scale = float(rng.uniform(0.92, 1.08))
        tx = float(rng.uniform(-8, 8))
        ty = float(rng.uniform(-8, 8))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # Photométrie modérée
        contrast = float(rng.uniform(0.85, 1.15))
        brightness = float(rng.uniform(-25, 25))
        img = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        if rng.random() < 0.4:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(rng.uniform(0.3, 1.2)))
        if rng.random() < 0.5:
            sigma = float(rng.uniform(2, 8))
            noise = rng.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def _augment_bg(self, patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Augmentation légère pour les bg patches : pas de bg fringe (déjà fond)."""
        img = patch.copy()
        # Rotation discrète + flip
        rot = int(rng.choice([0, 90, 180, 270]))
        if rot == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rot == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rng.random() < 0.5:
            img = cv2.flip(img, 1)
        # Photométrie modérée
        contrast = float(rng.uniform(0.8, 1.2))
        brightness = float(rng.uniform(-25, 25))
        img = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        if rng.random() < 0.4:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(rng.uniform(0.3, 1.5)))
        if rng.random() < 0.5:
            sigma = float(rng.uniform(2, 10))
            noise = rng.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img
