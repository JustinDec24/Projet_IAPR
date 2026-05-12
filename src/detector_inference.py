"""Inference du détecteur appris : image → liste de bboxes.

Post-processing :
- Seuillage de la heatmap d'objectness
- Pour chaque pixel positif, lire la bbox prédite
- NMS pour dédupliquer (cas où plusieurs pixels près du centre prédisent la même bbox)

Sortie : liste de cartes avec rect + warped crop, comme `detect_cards_in_scene`,
prête à être consommée par le classifieur CNN.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .detector import CardDetector
from .detector_dataset import INPUT_H, INPUT_W
from .templates import CARD_H, CARD_W, _order_corners, dominant_color


def _nms(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
         ) -> list[int]:
    """Non-Max Suppression. bboxes : (N, 4) en (x_min, y_min, x_max, y_max)."""
    if len(bboxes) == 0:
        return []
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


class CardDetectorRuntime:
    """Wrapper pour charger un détecteur entraîné et faire l'inference image → bboxes."""

    def __init__(self, checkpoint_path: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = CardDetector()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

    @torch.no_grad()
    def predict_bboxes(
        self, image: np.ndarray,
        obj_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 30,
    ) -> list[tuple[int, int, int, int, float]]:
        """Renvoie une liste [(x_min, y_min, x_max, y_max, score), ...] dans les
        coords de l'image originale.
        """
        h0, w0 = image.shape[:2]
        img_resized = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        out = self.model(tensor)
        obj = torch.sigmoid(out["obj"])[0, 0].cpu().numpy()   # (GH, GW)
        bbox = out["bbox"][0].cpu().numpy()                    # (4, GH, GW)

        # Récupérer toutes les positions au-dessus du seuil
        ys, xs = np.where(obj > obj_threshold)
        if len(ys) == 0:
            return []
        scores = obj[ys, xs]
        cx_n = bbox[0, ys, xs]
        cy_n = bbox[1, ys, xs]
        w_n = bbox[2, ys, xs]
        h_n = bbox[3, ys, xs]

        # Convertir en coords absolues de l'image originale
        cx = cx_n * w0
        cy = cy_n * h0
        bw = w_n * w0
        bh = h_n * h0
        x_min = np.clip(cx - bw / 2, 0, w0 - 1)
        y_min = np.clip(cy - bh / 2, 0, h0 - 1)
        x_max = np.clip(cx + bw / 2, 0, w0 - 1)
        y_max = np.clip(cy + bh / 2, 0, h0 - 1)
        bboxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

        keep_idx = _nms(bboxes, scores, iou_threshold=iou_threshold)
        keep_idx = keep_idx[:max_detections]

        return [
            (int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 2]), int(bboxes[i, 3]),
             float(scores[i]))
            for i in keep_idx
        ]


def warp_bbox_to_canonical(image: np.ndarray, x_min: int, y_min: int,
                            x_max: int, y_max: int) -> np.ndarray:
    """Warp une bbox axis-aligned vers un crop 200×300 portrait.

    Comme le détecteur produit des bboxes axis-aligned, on ne peut pas faire de
    perspective warp comme avec un rotated rect. On crop la bbox + on resize au
    ratio canonique. Si bbox plus large que haute, on rotate de 90° pour avoir
    portrait.
    """
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
    h, w = crop.shape[:2]
    if w > h:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(crop, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)


def detect_cards_with_model(image: np.ndarray, runtime: CardDetectorRuntime,
                            exclude_xy: tuple[float, float] | None = None,
                            exclude_radius: float = 220) -> list[dict]:
    """Pipeline équivalent à `detect_cards_in_scene` mais via le détecteur appris.

    Renvoie une liste de cartes au même format ({cx, cy, rect, warped, color, ...})
    pour être plug-and-play avec le pipeline d'inférence existant.
    """
    bboxes = runtime.predict_bboxes(image)
    cards: list[dict] = []
    for x_min, y_min, x_max, y_max, score in bboxes:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        if exclude_xy is not None:
            ex, ey = exclude_xy
            if (cx - ex) ** 2 + (cy - ey) ** 2 < exclude_radius ** 2:
                continue
        w = x_max - x_min
        h = y_max - y_min
        warped = warp_bbox_to_canonical(image, x_min, y_min, x_max, y_max)
        # Pour rester compatible avec le format existant, on fabrique un "rotated
        # rect" axis-aligned (angle=0) qui correspond à la bbox.
        rect = ((float(cx), float(cy)), (float(w), float(h)), 0.0)
        cards.append({
            "cx": float(cx), "cy": float(cy), "rect": rect, "warped": warped,
            "color": dominant_color(warped), "source": "yolo", "score": float(score),
        })
    return cards
