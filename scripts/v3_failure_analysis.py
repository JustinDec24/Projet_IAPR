"""Phase 0 V3 — Failure analysis catégorisée du baseline 0.764.

Catégorise chaque erreur en 6 types :
  1. FN_detection      : carte GT non détectée (manque dans pred, label absent)
  2. FP_detection      : carte prédite inexistante (label en trop, hallucination)
  3. misclassification : carte détectée mais mauvais label (même zone joueur)
  4. wrong_player      : bon label mais mauvaise zone joueur
  5. center_error      : carte centrale fausse
  6. active_error      : joueur actif faux (token)

Modèles : baseline 0.764 = classifier_v4_realmix.pt (11.2M) + detector_baseline.pt (0.53M)

Sortie : reports/v3_failure_analysis.csv + stats agrégées sur stdout.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.detector_inference import CardDetectorRuntime  # noqa: E402
from src.inference import Classifier, predict_scene  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v3_failure")

CLASSIFIER_CKPT = ROOT / "outputs" / "models" / "classifier_v4_realmix.pt"
DETECTOR_CKPT = ROOT / "outputs" / "models" / "detector_baseline.pt"
OUT_CSV = ROOT / "reports" / "v3_failure_analysis.csv"
PLAYERS = ("p1", "p2", "p3", "p4")


def parse_hand(s: str) -> list[str]:
    return [] if (not s or s == "EMPTY") else s.split(";")


def categorize(pred_hands: dict[str, list[str]], gt_hands: dict[str, list[str]],
                ) -> dict[str, int]:
    """Catégorise les erreurs cartes (FN/FP/misclass/wrong_player) sur 1 image.

    Méthode :
    - multiset global labels : compare pred vs gt tous joueurs confondus
      → TP_label = matches de label, FN_label = GT manquants, FP_label = pred en trop
    - puis on raffine : un label correctement détecté mais dans le mauvais joueur
      = wrong_player (pas misclassification).
    """
    gt_all = Counter(c for h in gt_hands.values() for c in h)
    pred_all = Counter(c for h in pred_hands.values() for c in h)

    tp_label = sum((gt_all & pred_all).values())
    fn_label = sum(gt_all.values()) - tp_label   # GT non retrouvés (label absent)
    fp_label = sum(pred_all.values()) - tp_label  # pred sans GT correspondant

    # Parmi les labels correctement présents, combien dans le mauvais joueur ?
    wrong_player = 0
    for slot in PLAYERS:
        gt_slot = Counter(gt_hands.get(slot, []))
        pred_slot = Counter(pred_hands.get(slot, []))
        # labels présents dans le GT de ce slot mais classés ailleurs
        for lbl, gt_n in gt_slot.items():
            pred_n_here = pred_slot.get(lbl, 0)
            # combien de ce label le pred a au total
            total_pred = pred_all.get(lbl, 0)
            total_gt = gt_all.get(lbl, 0)
            matched = min(total_pred, total_gt)
            # mal placés = matched - bien_places
            # approx : si label détecté globalement mais pas dans le bon slot
            mismatch_here = max(0, min(gt_n, matched) - pred_n_here)
            wrong_player += mismatch_here

    # misclassification ≈ FP qui correspondent à des FN (carte vue, mauvais label)
    misclass = min(fp_label, fn_label)
    fn_det = fn_label - misclass     # vraies cartes ratées
    fp_det = fp_label - misclass     # hallucinations
    return {
        "FN_detection": fn_det,
        "FP_detection": fp_det,
        "misclassification": misclass,
        "wrong_player": wrong_player,
    }


def main() -> int:
    if not CLASSIFIER_CKPT.exists() or not DETECTOR_CKPT.exists():
        logger.error("Modèles baseline manquants : %s / %s", CLASSIFIER_CKPT, DETECTOR_CKPT)
        return 1

    classifier = Classifier(CLASSIFIER_CKPT)
    detector = CardDetectorRuntime(DETECTOR_CKPT, device=str(classifier.device))
    logger.info("Baseline : classifier=%s detector=%s", CLASSIFIER_CKPT.name, DETECTOR_CKPT.name)

    df = pd.read_csv(TRAIN_CSV)
    rows: list[dict] = []
    agg = Counter()

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Analysing"):
        path = TRAIN_DIR / f"{row.image_id}.jpg"
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        pred = predict_scene(image, row.image_id, classifier,
                             confidence_threshold=0.40, use_tta=True,
                             detector=detector, hybrid=True)

        gt_hands = {p: parse_hand(getattr(row, f"player_{i + 1}_cards"))
                    for i, p in enumerate(PLAYERS)}
        cats = categorize(pred.player_cards, gt_hands)

        center_error = int(pred.center_card != row.center_card)
        active_error = int(pred.active_player != row.active_player)
        for k, v in cats.items():
            agg[k] += v
        agg["center_error"] += center_error
        agg["active_error"] += active_error

        rows.append({
            "image_id": row.image_id,
            **cats,
            "center_error": center_error,
            "active_error": active_error,
            "gt_center": row.center_card,
            "pred_center": pred.center_card,
            "gt_active": row.active_player,
            "pred_active": pred.active_player,
            "n_gt": sum(len(h) for h in gt_hands.values()),
            "n_pred": sum(len(h) for h in pred.player_cards.values()),
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n = len(out)
    # Score
    total_gt = out["n_gt"].sum()
    center_acc = 1 - out["center_error"].sum() / n
    active_acc = 1 - out["active_error"].sum() / n

    print("\n" + "=" * 60)
    print(f"PHASE 0 — Failure analysis baseline (n={n} images)")
    print("=" * 60)
    print(f"\nCenterAcc : {center_acc:.3f}  ({out['center_error'].sum()} erreurs)")
    print(f"ActiveAcc : {active_acc:.3f}  ({out['active_error'].sum()} erreurs)")

    print("\n--- ERREURS CARTES (catégorisées) ---")
    card_err_total = (agg["FN_detection"] + agg["FP_detection"]
                      + agg["misclassification"] + agg["wrong_player"])
    for cat in ["FN_detection", "FP_detection", "misclassification", "wrong_player"]:
        pct = 100 * agg[cat] / max(card_err_total, 1)
        print(f"  {cat:20s} : {agg[cat]:4d}  ({pct:5.1f}% des err. cartes)")
    print(f"  {'TOTAL err cartes':20s} : {card_err_total:4d}  (sur {total_gt} cartes GT)")

    print("\n--- ERREURS SCÈNE ---")
    print(f"  center_error : {agg['center_error']:3d} / {n}")
    print(f"  active_error : {agg['active_error']:3d} / {n}")

    print("\n--- CONFUSION CENTER (top 8) ---")
    bad_c = out[out["center_error"] == 1]
    for (g, p), c in Counter(zip(bad_c["gt_center"], bad_c["pred_center"])).most_common(8):
        print(f"  {g:>10} -> {p:>10} : {c}x")

    print("\n--- CONFUSION ACTIVE (top 8) ---")
    bad_a = out[out["active_error"] == 1]
    for (g, p), c in Counter(zip(bad_a["gt_active"], bad_a["pred_active"])).most_common(8):
        print(f"  {g:>5} -> {p:>5} : {c}x")

    print(f"\nCSV détaillé : {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
