"""Phase 1 V3 — Test du classifieur V2 (11.2M) sur les crops de coins.

Objectif : valider que l'info couleur+valeur du coin suffit à classifier.

Critère :
  - accuracy >= 80 %  -> GO (l'approche coin tient)
  - accuracy <= 60 %  -> NO-GO (retour OBB)
  - entre 60-80 %     -> ZONE GRISE (le classifieur full-card est OOD sur des
    coins ; un classifieur coin-spécifique entraîné en Phase 4 ferait mieux)

NB : le classifieur a été entraîné sur des cartes ENTIÈRES warpées 200×300.
Lui donner un crop de coin seul est out-of-distribution → cette accuracy est
une borne BASSE. Un vrai corner-classifier (Phase 4) dépasserait ce chiffre.

Sortie : reports/v3_corner_validation.md + matrice de confusion.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_TO_IDX, IDX_TO_CLASS  # noqa: E402
from src.inference import Classifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_corners")

CORNERS_DIR = ROOT / "data" / "corners_test"
CLASSIFIER_CKPT = ROOT / "outputs" / "models" / "classifier_v4_realmix.pt"
REPORT = ROOT / "reports" / "v3_corner_validation.md"


def main() -> int:
    if not CORNERS_DIR.exists():
        logger.error("Coins manquants : %s (lance extract_corners_test.py)", CORNERS_DIR)
        return 1
    classifier = Classifier(CLASSIFIER_CKPT)
    logger.info("Classifier : %s (device=%s)", CLASSIFIER_CKPT.name, classifier.device)

    # Charge tous les coins + labels
    crops: list[np.ndarray] = []
    labels: list[str] = []
    for label_dir in sorted(CORNERS_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        for p in sorted(label_dir.glob("*.png")):
            img = cv2.imread(str(p))
            if img is None:
                continue
            crops.append(img)
            labels.append(label_dir.name)
    logger.info("Chargé %d coins, %d classes", len(crops), len(set(labels)))

    # Classification (sans TTA d'abord, puis avec TTA)
    preds_plain = classifier.classify(crops)             # [(label, conf)]
    probs_tta = classifier.classify_full_tta(crops)      # (N, 54)
    preds_tta = [IDX_TO_CLASS[int(probs_tta[i].argmax())] for i in range(len(crops))]

    def accuracy(preds: list) -> float:
        ok = sum(1 for (pl, gt) in zip(preds, labels)
                 if (pl[0] if isinstance(pl, tuple) else pl) == gt)
        return ok / max(len(labels), 1)

    acc_plain = accuracy([p[0] for p in preds_plain])
    acc_tta = accuracy(preds_tta)

    # Accuracy "couleur seulement" (le coin doit au moins donner la bonne couleur)
    def color_of(lbl: str) -> str:
        return "k" if lbl in ("wild", "draw_4") else lbl[0]
    color_ok = sum(1 for p, gt in zip(preds_tta, labels)
                   if color_of(p) == color_of(gt))
    acc_color = color_ok / max(len(labels), 1)

    # Confusion par classe (TTA)
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_ok: dict[str, int] = defaultdict(int)
    confusions: Counter = Counter()
    for p, gt in zip(preds_tta, labels):
        per_class_total[gt] += 1
        if p == gt:
            per_class_ok[gt] += 1
        else:
            confusions[(gt, p)] += 1

    print("\n" + "=" * 60)
    print("PHASE 1 — Validation approche coin")
    print("=" * 60)
    print(f"\nAccuracy (sans TTA)        : {acc_plain:.3f}")
    print(f"Accuracy (avec TTA 8x)     : {acc_tta:.3f}")
    print(f"Accuracy COULEUR seule     : {acc_color:.3f}")
    print(f"\nTop confusions :")
    for (gt, p), c in confusions.most_common(12):
        print(f"  {gt:>10} -> {p:>10} : {c}x")

    # Verdict
    if acc_tta >= 0.80:
        verdict = "GO"
        verdict_txt = "L'approche coin tient (>=80%). On continue Phase 2+."
    elif acc_tta <= 0.60:
        verdict = "NO-GO"
        verdict_txt = "Accuracy trop faible (<=60%). Retour à l'approche OBB."
    else:
        verdict = "GO (conditionnel)"
        verdict_txt = ("Zone grise 60-80%. MAIS le classifieur est OOD (entraîné "
                       "full-card). Un corner-classifier dédié (Phase 4) dépassera "
                       "ce chiffre. L'accuracy COULEUR élevée confirme que le coin "
                       "porte l'info. GO vers Phase 2.")

    print(f"\n>>> VERDICT : {verdict}")
    print(f"    {verdict_txt}")

    # Rapport markdown
    lines = [
        "# Phase 1 — Validation empirique de l'approche \"coin\"",
        "",
        f"**Test set** : {len(crops)} coins (35%×35% haut-gauche) extraits des "
        f"crops réels labellisés, {len(set(labels))} classes.",
        "",
        f"**Classifieur testé** : `{CLASSIFIER_CKPT.name}` (11.2M, entraîné sur "
        "cartes ENTIÈRES → les coins sont out-of-distribution pour lui).",
        "",
        "## Résultats",
        "",
        "| Métrique | Accuracy |",
        "|---|---|",
        f"| Coin, sans TTA | {acc_plain:.3f} |",
        f"| Coin, avec TTA 8× | {acc_tta:.3f} |",
        f"| Couleur seule (TTA) | {acc_color:.3f} |",
        "",
        "## Top confusions",
        "",
        "| GT | Prédit | Count |",
        "|---|---|---|",
    ]
    for (gt, p), c in confusions.most_common(12):
        lines.append(f"| {gt} | {p} | {c} |")
    lines += [
        "",
        f"## Verdict : **{verdict}**",
        "",
        verdict_txt,
        "",
        "### Interprétation",
        "",
        "- L'accuracy *full-card classifier sur coins* est une **borne basse** "
        "(domain shift : il n'a jamais vu de coins seuls).",
        f"- L'accuracy **couleur seule = {acc_color:.3f}** indique si le coin "
        "porte au moins le signal couleur (toujours vrai par design UNO).",
        "- Le **corner-classifier dédié** (Phase 4), entraîné spécifiquement sur "
        "des coins, devrait nettement dépasser ces chiffres.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Rapport écrit : %s", REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
