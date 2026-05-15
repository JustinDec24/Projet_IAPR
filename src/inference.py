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

from .config import DATA_DIR, IDX_TO_CLASS, NON_CARD_LABEL, NUM_CLASSES

_TEMPLATES_DIR = DATA_DIR / "card_templates"
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
DEFAULT_CONFIDENCE_THRESHOLD = 0.50


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
        # Détecte les channels depuis le state_dict si non spécifié
        channels = ckpt.get("channels")
        if channels is None:
            stem_weight = ckpt["state_dict"]["stem.0.weight"]
            c1 = stem_weight.shape[0]
            channels = (c1, c1 * 2, c1 * 4, c1 * 8)
        self.model = UnoCNN(num_classes=ckpt.get("num_classes", NUM_CLASSES),
                            channels=tuple(channels))
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
    center_tpl_match: bool = False,  # NCC template-match : echec (CenterAcc 0.21)
    center_tpl_min_conf: float = 0.55,
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
        # Joueurs : détecteur appris. Centre : on garde les DEUX vues
        # (heuristique + détecteur appris) → ensemble 0-param sur la carte
        # centrale (on choisira la plus confiante en aval).
        cards = [c for c in heur_cards
                 if assign_player(c["cx"], c["cy"], image.shape) == "center"]
        for c in model_cards:
            z = assign_player(c["cx"], c["cy"], image.shape)
            if z != "center":
                cards.append(c)
            else:
                cards.append(c)  # vue détecteur-appris du centre (ensemble)
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

    # On sépare la carte centrale des cartes joueurs AVANT le filtre confiance.
    # La carte centrale est unique (1 par image) : une prédiction même peu sûre
    # vaut mieux qu'EMPTY (toujours faux). On NE la filtre donc PAS par confiance.
    h, w = image.shape[:2]
    center_candidates = [c for c in cards
                         if assign_player(c["cx"], c["cy"], image.shape) == "center"
                         and c["label"] != NON_CARD_LABEL]

    pred = ScenePrediction(image_id=image_id)
    if center_candidates:
        # Ensemble 0-param : parmi les vues du centre (heuristique +
        # détecteur appris), on prend la prédiction la plus confiante,
        # en se restreignant aux candidats vraiment proches du centre image.
        cx_w, cy_h = w / 2, h / 2
        d_sorted = sorted(center_candidates,
                          key=lambda c: (c["cx"] - cx_w) ** 2 + (c["cy"] - cy_h) ** 2)
        diag2 = (w * w + h * h)
        near = [c for c in d_sorted
                if ((c["cx"] - cx_w) ** 2 + (c["cy"] - cy_h) ** 2) < 0.05 * diag2]
        pool = near if near else d_sorted[:1]
        best = max(pool, key=lambda c: c.get("confidence", 0.0))
        center_label = best["label"]
        # Override par template-matching (0 param) : la carte centrale est
        # isolée/nette → un NN sur les 54 templates y est très fiable. On
        # remplace le label du student si le match template est confiant.
        if center_tpl_match:
            try:
                from .v3.center_card_matcher import match_center_card
                tpl_lbl, tpl_conf = match_center_card(
                    best["warped"], _TEMPLATES_DIR)
                if tpl_lbl and tpl_conf >= center_tpl_min_conf:
                    center_label = tpl_lbl
            except Exception:
                pass
        pred.center_card = center_label

    # Cartes joueurs : filtrées par confiance (précision sur le F1)
    cards = [c for c in cards
             if c["confidence"] >= confidence_threshold and c["label"] != NON_CARD_LABEL]
    if not cards:
        return pred

    # Assignation aux joueurs (center exclu — déjà traité)
    by_player: dict[str, list[dict]] = defaultdict(list)
    for c in cards:
        slot = assign_player(c["cx"], c["cy"], image.shape)
        if slot != "center":
            by_player[slot].append(c)

    for slot in ("p1", "p2", "p3", "p4"):
        pred.player_cards[slot] = [c["label"] for c in by_player.get(slot, [])]

    # Joueur actif : le jeton a déjà été détecté en amont (avant la détection
    # des cartes, pour exclure sa zone).
    if token is not None:
        active = assign_token_to_player(token, cards, image.shape)
        if active is not None:
            pred.active_player = active

    return pred
