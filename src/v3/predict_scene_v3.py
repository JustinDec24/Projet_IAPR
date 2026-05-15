"""Phase 6 V3 — Pipeline d'inférence intégré (corner detection + classification).

1. token (HSV classique)        → joueur actif
2. détecteur de coins           → liste (x,y,score)
3. classifieur de coins         → (x,y,label,score) par coin
4. déduplication géométrique    → cartes (1-2 coins)
5. filtre confiance
6. assignation par zone         → center + p1..p4
7. CSV Kaggle
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import IDX_TO_CLASS
from src.detection import assign_player  # méthode géométrique V2 (éprouvée)
from src.v3.corner_classifier import CornerClassifier, INPUT_SIZE
from src.v3.corner_classifier_dataset import _crop_around
from src.v3.corner_deduplication import Corner, deduplicate_corners
from src.v3.corner_detector_runtime import CornerDetectorRuntime
from src.v3.token_detector import detect_token


@dataclass
class ScenePredV3:
    image_id: str
    center_card: str = "EMPTY"
    active_player: str = "EMPTY"
    player_cards: dict[str, list[str]] = field(
        default_factory=lambda: {p: [] for p in ("p1", "p2", "p3", "p4")})

    def to_csv_row(self) -> dict[str, str]:
        def hand(c: list[str]) -> str:
            return ";".join(c) if c else "EMPTY"
        return {
            "image_id": self.image_id,
            "center_card": self.center_card or "EMPTY",
            "active_player": self.active_player or "EMPTY",
            "player_1_cards": hand(self.player_cards["p1"]),
            "player_2_cards": hand(self.player_cards["p2"]),
            "player_3_cards": hand(self.player_cards["p3"]),
            "player_4_cards": hand(self.player_cards["p4"]),
        }


class CornerClassifierRuntime:
    def __init__(self, ckpt_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                               else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        ch = tuple(ckpt.get("channels", (48, 96, 160, 224)))
        self.model = CornerClassifier(channels=ch)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

    @torch.no_grad()
    def classify(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        batch = []
        for c in crops:
            c = cv2.resize(c, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            batch.append(torch.from_numpy(c).permute(2, 0, 1).float() / 255.0)
        x = torch.stack(batch).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)
        conf, idx = probs.max(dim=1)
        return [(IDX_TO_CLASS[int(i)], float(c))
                for i, c in zip(idx.tolist(), conf.tolist())]


def predict_scene_v3(image: np.ndarray, image_id: str,
                     detector: CornerDetectorRuntime,
                     classifier: CornerClassifierRuntime,
                     conf_threshold: float = 0.5,
                     corner_crop_frac: float = 0.42,
                     diag_min: float = 200.0,
                     diag_max: float = 520.0,
                     zones: dict | None = None) -> ScenePredV3:
    h, w = image.shape[:2]
    pred = ScenePredV3(image_id=image_id)

    # 1. token
    token = detect_token(image)

    # 2. coins
    raw = detector.detect(image)  # [(x,y,score)]
    if not raw:
        if token is not None:
            z = assign_player(token[0], token[1], image.shape)
            pred.active_player = "p1" if z == "center" else z
        return pred

    # 3. classification de chaque coin (crop dimensionné à la carte)
    # taille de crop ≈ corner_crop_frac × largeur carte. On estime la largeur
    # carte via l'espacement médian des coins (proxy robuste).
    pts = np.array([[x, y] for x, y, _ in raw])
    if len(pts) >= 2:
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        med_nn = float(np.median(d.min(axis=1)))
        crop_sz = int(np.clip(med_nn * 1.1, 36, 140))
    else:
        crop_sz = 70
    crops = [_crop_around(image, x, y, crop_sz) for x, y, _ in raw]
    labels = classifier.classify(crops)

    corners = [Corner(x=raw[i][0], y=raw[i][1], label=labels[i][0],
                      score=raw[i][2] * labels[i][1])
               for i in range(len(raw))]

    # 4. déduplication géométrique
    cards = deduplicate_corners(corners, diag_min=diag_min, diag_max=diag_max)

    # 5. filtre confiance
    cards = [c for c in cards if c.confidence >= conf_threshold]

    # 6. assignation géométrique V2 (distance centre image + angle) — robuste
    center_candidates = [c for c in cards
                         if assign_player(c.cx, c.cy, image.shape) == "center"]
    if center_candidates:
        # la plus proche du centre image (comme V2)
        best = min(center_candidates,
                   key=lambda c: (c.cx - w / 2) ** 2 + (c.cy - h / 2) ** 2)
        pred.center_card = best.label
    for c in cards:
        z = assign_player(c.cx, c.cy, image.shape)
        if z != "center" and z in pred.player_cards:
            pred.player_cards[z].append(c.label)

    # 7. joueur actif
    if token is not None:
        z = assign_player(token[0], token[1], image.shape)
        pred.active_player = "p1" if z == "center" else z

    return pred
