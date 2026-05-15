"""Phase 6 V3 — Runtime d'inférence du détecteur de coins.

image → liste de coins (x, y, score) en coords image originale.
Peak detection style CenterNet (max-pool 3×3) + offsets sous-pixel.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.v3.corner_dataset import DOWNSAMPLE, INPUT_H, INPUT_W
from src.v3.corner_detector import CornerDetector


class CornerDetectorRuntime:
    def __init__(self, ckpt_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                               else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        ch = tuple(ckpt.get("channels", (40, 80, 160, 320)))
        self.model = CornerDetector(channels=ch)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

    @torch.no_grad()
    def detect(self, image: np.ndarray, thr: float = 0.30,
               max_corners: int = 60) -> list[tuple[float, float, float]]:
        """Renvoie [(x, y, score), ...] en pixels de l'image d'origine."""
        h0, w0 = image.shape[:2]
        rs = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        out = self.model(t.to(self.device))
        hm = torch.sigmoid(out["heatmap"])          # (1,1,Gh,Gw)
        off = out["offset"]                          # (1,2,Gh,Gw)
        pooled = F.max_pool2d(hm, 3, 1, 1)
        peaks = (hm == pooled) & (hm > thr)
        ys, xs = torch.where(peaks[0, 0])
        sx, sy = w0 / INPUT_W, h0 / INPUT_H
        res: list[tuple[float, float, float]] = []
        for gy, gx in zip(ys.tolist(), xs.tolist()):
            score = float(hm[0, 0, gy, gx])
            dx = float(off[0, 0, gy, gx])
            dy = float(off[0, 1, gy, gx])
            x = (gx + dx) * DOWNSAMPLE * sx
            y = (gy + dy) * DOWNSAMPLE * sy
            res.append((x, y, score))
        res.sort(key=lambda r: -r[2])
        return res[:max_corners]
