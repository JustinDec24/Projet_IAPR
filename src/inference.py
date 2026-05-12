"""Pipeline d'inférence : image de jeu → prédictions structurées.

Combine :
- la détection des cartes (`detection.detect_cards_in_scene`)
- la classification de chaque crop par le CNN entraîné
- la détection du jeton actif et son assignation à un joueur
- l'assignation géométrique des cartes aux joueurs
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import IDX_TO_CLASS, NON_CARD_LABEL, NUM_CLASSES
from .detection import (
    assign_player,
    assign_token_to_player,
    detect_active_token,
    detect_cards_in_scene,
)
from .detector_inference import CardDetectorRuntime, detect_cards_with_model
from .model import UnoCNN

# Confidence en-dessous de laquelle on considère qu'un crop n'est pas une carte
# (typiquement faux positif détecté dans le feuillage).
DEFAULT_CONFIDENCE_THRESHOLD = 0.40


@dataclass
class ScenePrediction:
    image_id: str
    center_card: str = "EMPTY"
    active_player: str = "EMPTY"
    player_cards: dict[str, list[str]] = field(default_factory=lambda: {p: [] for p in ("p1", "p2", "p3", "p4")})

    def to_csv_row(self) -> dict[str, str]:
        def hand(cards: list[str]) -> str:
            return ";".join(cards) if cards else "EMPTY"
        return {
            "image_id": self.image_id,
            "center_card": self.center_card,
            "active_player": self.active_player,
            "player_1_cards": hand(self.player_cards["p1"]),
            "player_2_cards": hand(self.player_cards["p2"]),
            "player_3_cards": hand(self.player_cards["p3"]),
            "player_4_cards": hand(self.player_cards["p4"]),
        }


class Classifier:
    """Wrapper autour du CNN entraîné pour classifier des crops 200×300 BGR."""

    def __init__(self, checkpoint_path: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.input_size: tuple[int, int] = tuple(ckpt["input_size"])
        self.model = UnoCNN(num_classes=ckpt.get("num_classes", NUM_CLASSES))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

    def _prepare_batch(self, crops: list[np.ndarray]) -> torch.Tensor:
        h, w = self.input_size
        tensors = []
        for crop in crops:
            img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def classify(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Renvoie [(label, confidence)] pour chaque crop (top-1)."""
        if not crops:
            return []
        batch = self._prepare_batch(crops)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, idx = probs.max(dim=1)
        return [(IDX_TO_CLASS[int(i)], float(c)) for i, c in zip(idx.tolist(), confs.tolist())]

    @torch.no_grad()
    def classify_full(self, crops: list[np.ndarray]) -> torch.Tensor:
        """Renvoie le tenseur (N, num_classes) des probabilités softmax."""
        if not crops:
            return torch.zeros(0, 0)
        batch = self._prepare_batch(crops)
        return torch.softmax(self.model(batch), dim=1)

    @torch.no_grad()
    def classify_full_tta(self, crops: list[np.ndarray]) -> torch.Tensor:
        """Test-Time Augmentation : moyenne des softmax sur 4 rotations (0/90/180/270°)
        + flip horizontal. Coût ≈ 8× forward, mais réduit la variance des prédictions.

        Note : pour les rotations 90°/270° qui changent l'orientation (landscape),
        on rotate puis on remet en portrait par resize — la carte voit son contenu
        à des angles différents, ce qui aide quand la détection donne un crop tilté.
        """
        if not crops:
            return torch.zeros(0, 0)
        all_probs: list[torch.Tensor] = []
        rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        for rot in rotations:
            if rot is None:
                rotated = crops
            else:
                rotated = [cv2.rotate(c, rot) for c in crops]
            batch = self._prepare_batch(rotated)
            all_probs.append(torch.softmax(self.model(batch), dim=1))
            # Avec flip horizontal en plus pour cette rotation
            flipped = [cv2.flip(c, 1) for c in rotated]
            batch_f = self._prepare_batch(flipped)
            all_probs.append(torch.softmax(self.model(batch_f), dim=1))
        return torch.stack(all_probs).mean(dim=0)


def predict_scene(
    image: np.ndarray, image_id: str, classifier: Classifier,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    use_tta: bool = False,
    detector: CardDetectorRuntime | None = None,
    hybrid: bool = False,
) -> ScenePrediction:
    """Pipeline complet : image → ScenePrediction.

    - `detector` seul : on utilise le détecteur appris partout.
    - `hybrid=True` (+ `detector` fourni) : heuristique pour la zone centrale
      (CenterAcc 0.901 sur l'heuristique vs 0.728 sur le détecteur appris),
      détecteur appris pour les zones joueurs (F1 0.753 vs 0.671).
    - sinon : fallback heuristique uniquement.
    """
    # On détecte le jeton AVANT les cartes pour pouvoir exclure sa zone de la
    # détection (évite que le jeton noir/jaune soit classé comme une carte).
    token = detect_active_token(image)
    if hybrid and detector is not None:
        heur_cards = detect_cards_in_scene(image, exclude_xy=token)
        model_cards = detect_cards_with_model(image, detector, exclude_xy=token)
        # Garde l'heuristique pour la zone centrale, le détecteur appris pour les joueurs
        cards = [c for c in heur_cards
                 if assign_player(c["cx"], c["cy"], image.shape) == "center"]
        cards += [c for c in model_cards
                  if assign_player(c["cx"], c["cy"], image.shape) != "center"]
    elif detector is not None:
        cards = detect_cards_with_model(image, detector, exclude_xy=token)
    else:
        cards = detect_cards_in_scene(image, exclude_xy=token)
    if not cards:
        pred = ScenePrediction(image_id=image_id)
        return pred

    # Classification des crops. Si le modèle a une classe « non_card » (55 classes),
    # on fait un fallback intelligent : si très sûr → rejette ; si incertain →
    # prend la meilleure classe carte. Sinon on prend simplement top-1.
    crops = [c["warped"] for c in cards]
    probs = classifier.classify_full_tta(crops) if use_tta else classifier.classify_full(crops)
    has_non_card = probs.numel() > 0 and probs.shape[1] > len(IDX_TO_CLASS) - 1 and IDX_TO_CLASS.get(probs.shape[1] - 1) == NON_CARD_LABEL
    if probs.numel() > 0:
        non_card_idx = probs.shape[1] - 1 if has_non_card else -1
        for i, card in enumerate(cards):
            top1_idx = int(probs[i].argmax().item())
            top1_conf = float(probs[i, top1_idx].item())
            if has_non_card and top1_idx == non_card_idx and top1_conf >= 0.90:
                card["label"] = NON_CARD_LABEL
                card["confidence"] = top1_conf
            elif has_non_card and top1_idx == non_card_idx:
                card_probs = probs[i].clone()
                card_probs[non_card_idx] = -1.0
                best_card_idx = int(card_probs.argmax().item())
                card["label"] = IDX_TO_CLASS[best_card_idx]
                card["confidence"] = float(probs[i, best_card_idx].item())
            else:
                card["label"] = IDX_TO_CLASS[top1_idx]
                card["confidence"] = top1_conf

    cards = [c for c in cards
             if c["confidence"] >= confidence_threshold and c["label"] != NON_CARD_LABEL]
    if not cards:
        return ScenePrediction(image_id=image_id)

    # Assignation aux joueurs / center
    by_player: dict[str, list[dict]] = defaultdict(list)
    for c in cards:
        slot = assign_player(c["cx"], c["cy"], image.shape)
        by_player[slot].append(c)

    pred = ScenePrediction(image_id=image_id)
    # Carte centrale : la plus proche du centre parmi celles classées "center"
    if by_player.get("center"):
        h, w = image.shape[:2]
        best = min(by_player["center"],
                   key=lambda c: (c["cx"] - w / 2) ** 2 + (c["cy"] - h / 2) ** 2)
        pred.center_card = best["label"]

    for slot in ("p1", "p2", "p3", "p4"):
        pred.player_cards[slot] = [c["label"] for c in by_player.get(slot, [])]

    # Joueur actif : le jeton a déjà été détecté en amont (avant la détection
    # des cartes, pour exclure sa zone).
    if token is not None:
        active = assign_token_to_player(token, cards, image.shape)
        if active is not None:
            pred.active_player = active

    return pred
