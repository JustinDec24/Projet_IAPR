"""Failure case analysis : décompose chaque erreur du pipeline pour identifier
le bottleneck réel et chiffrer les gains potentiels par composant.

Pour chaque image train :
  - F1 multiset normal (carte avec joueur)
  - F1 multiset des LABELS seulement (ignore l'assignation joueur)
  - F1 multiset par joueur (où ça casse spécifiquement)
  - n_pred vs n_gt (détection sous/sur-compte)
  - Center OK / Active OK
  - Pour les "active" rates : token détecté ? si oui, mauvaise zone ?

À la fin, calcule des "scores oracle" :
  - Oracle player assignment : si on assignait parfaitement les labels prédits
  - Oracle detection : on connaît n_cards exact (impossible vraiment, mais on
    estime via la précision/rappel observés)
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TRAIN_DIR  # noqa: E402
from src.detector_inference import CardDetectorRuntime  # noqa: E402
from src.inference import Classifier, predict_scene  # noqa: E402

CHECKPOINT = ROOT / "outputs" / "models" / "classifier.pt"
DETECTOR_CHECKPOINT = ROOT / "outputs" / "models" / "detector.pt"
OUT_CSV = ROOT / "outputs" / "failure_analysis.csv"


def parse_hand(s: str) -> list[str]:
    return [] if (not s or s == "EMPTY") else s.split(";")


def f1_multiset(pred: list[str], gt: list[str]) -> float:
    p, g = Counter(pred), Counter(gt)
    tp = sum((p & g).values())
    fp = sum(p.values()) - tp
    fn = sum(g.values()) - tp
    return 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def f1_multiset_components(pred: list[str], gt: list[str]) -> tuple[int, int, int]:
    """Renvoie (TP, FP, FN) du F1 multiset."""
    p, g = Counter(pred), Counter(gt)
    tp = sum((p & g).values())
    fp = sum(p.values()) - tp
    fn = sum(g.values()) - tp
    return tp, fp, fn


def main() -> int:
    classifier = Classifier(CHECKPOINT)
    print(f"Classifier : {CHECKPOINT.name}")

    detector = None
    if DETECTOR_CHECKPOINT.exists():
        detector = CardDetectorRuntime(DETECTOR_CHECKPOINT, device=str(classifier.device))
        print(f"Detector   : {DETECTOR_CHECKPOINT.name} (mode hybrid + TTA)")

    df = pd.read_csv(TRAIN_CSV)

    rows = []
    # Aggregates for oracle scores
    total_tp, total_fp, total_fn = 0, 0, 0
    total_tp_labels, total_fp_labels, total_fn_labels = 0, 0, 0
    total_centers_ok = 0
    total_active_ok = 0
    token_active_failures: list[dict] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Analysing"):
        path = TRAIN_DIR / f"{row.image_id}.jpg"
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        pred = predict_scene(image, row.image_id, classifier,
                             confidence_threshold=0.40, use_tta=True,
                             detector=detector, hybrid=True)

        gt_hands = {p: parse_hand(getattr(row, f"player_{i + 1}_cards"))
                    for i, p in enumerate(["p1", "p2", "p3", "p4"])}
        gt_all = sum(gt_hands.values(), [])
        pred_all = sum(pred.player_cards.values(), [])
        n_gt = len(gt_all)
        n_pred = len(pred_all)

        # F1 standard : (label, player) doivent matcher
        joint_pred = [(c, p) for p in pred.player_cards for c in pred.player_cards[p]]
        joint_gt = [(c, p) for p in gt_hands for c in gt_hands[p]]
        joint_pred_str = [f"{c}@{p}" for c, p in joint_pred]
        joint_gt_str = [f"{c}@{p}" for c, p in joint_gt]
        f1_joint = f1_multiset(joint_pred_str, joint_gt_str)
        tp_j, fp_j, fn_j = f1_multiset_components(joint_pred_str, joint_gt_str)

        # F1 labels-only : ignore l'assignation joueur, juste les cartes détectées+classifiées
        f1_labels = f1_multiset(pred_all, gt_all)
        tp_l, fp_l, fn_l = f1_multiset_components(pred_all, gt_all)

        total_tp += tp_j
        total_fp += fp_j
        total_fn += fn_j
        total_tp_labels += tp_l
        total_fp_labels += fp_l
        total_fn_labels += fn_l

        center_ok = int(pred.center_card == row.center_card)
        active_ok = int(pred.active_player == row.active_player)
        total_centers_ok += center_ok
        total_active_ok += active_ok

        if not active_ok:
            token_active_failures.append({
                "image_id": row.image_id,
                "gt_active": row.active_player,
                "pred_active": pred.active_player,
            })

        rows.append({
            "image_id": row.image_id,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_diff": n_pred - n_gt,
            "f1_joint": round(f1_joint, 3),
            "f1_labels": round(f1_labels, 3),
            "delta_player": round(f1_labels - f1_joint, 3),
            "tp_j": tp_j, "fp_j": fp_j, "fn_j": fn_j,
            "tp_l": tp_l, "fp_l": fp_l, "fn_l": fn_l,
            "center_ok": center_ok,
            "active_ok": active_ok,
            "gt_center": row.center_card,
            "pred_center": pred.center_card,
            "gt_active": row.active_player,
            "pred_active": pred.active_player,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    n = len(out)
    print(f"\n{'=' * 60}")
    print(f"Failure case analysis sur {n} images d'entraînement")
    print(f"{'=' * 60}\n")

    # Global metrics
    f1_joint_global = 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1)
    f1_labels_global = 2 * total_tp_labels / max(2 * total_tp_labels + total_fp_labels + total_fn_labels, 1)
    center_acc = total_centers_ok / n
    active_acc = total_active_ok / n
    score = 0.1 * center_acc + 0.1 * active_acc + 0.8 * f1_joint_global
    print(f"Score        : {score:.3f}")
    print(f"  CenterAcc  : {center_acc:.3f}")
    print(f"  ActiveAcc  : {active_acc:.3f}")
    print(f"  F1 joint   : {f1_joint_global:.3f}  (TP={total_tp}, FP={total_fp}, FN={total_fn})")
    print(f"  F1 labels  : {f1_labels_global:.3f}  (TP={total_tp_labels}, FP={total_fp_labels}, FN={total_fn_labels})")

    print(f"\n--- DÉCOMPOSITION DES ERREURS ---")
    # Si f1_labels >> f1_joint -> l'assignation au joueur est cassée
    player_error_share = (total_fp + total_fn) - (total_fp_labels + total_fn_labels)
    detection_classif_share = total_fp_labels + total_fn_labels
    print(f"Erreurs détection+classif (FP+FN labels) : {detection_classif_share}")
    print(f"Erreurs assignation joueur (extra)       : {player_error_share}")
    print(f"  -> {player_error_share / max(detection_classif_share + player_error_share, 1) * 100:.1f}% des erreurs viennent de l'assignation joueur")

    print(f"\n--- ORACLE SCORES (si on fixait 1 composant à 100%) ---")
    # Oracle player assignment : score si f1_joint = f1_labels (assignation parfaite étant donné les labels)
    oracle_player_score = 0.1 * center_acc + 0.1 * active_acc + 0.8 * f1_labels_global
    print(f"Si player assignment parfait      : {oracle_player_score:.3f}  (+{oracle_player_score - score:.3f})")
    # Oracle center : score si tous les centers OK
    oracle_center_score = 0.1 * 1.0 + 0.1 * active_acc + 0.8 * f1_joint_global
    print(f"Si CenterAcc = 1.0                : {oracle_center_score:.3f}  (+{oracle_center_score - score:.3f})")
    # Oracle active : score si tous les active OK
    oracle_active_score = 0.1 * center_acc + 0.1 * 1.0 + 0.8 * f1_joint_global
    print(f"Si ActiveAcc = 1.0                : {oracle_active_score:.3f}  (+{oracle_active_score - score:.3f})")
    # Oracle detection+classif : score si f1_labels = 1.0 (toutes les cartes correctement détectées+classifiées)
    oracle_dc_score = 0.1 * center_acc + 0.1 * active_acc + 0.8 * 1.0
    print(f"Si F1 détection+classif = 1.0     : {oracle_dc_score:.3f}  (+{oracle_dc_score - score:.3f})")

    print(f"\n--- TOP 15 IMAGES WORST F1_joint ---")
    print(out.nsmallest(15, "f1_joint")[
        ["image_id", "n_gt", "n_pred", "f1_joint", "f1_labels", "delta_player", "center_ok", "active_ok"]
    ].to_string(index=False))

    print(f"\n--- COUNTS D'ERREURS PAR TYPE ---")
    # FP > FN : sur-détection (faux positifs)
    # FN > FP : sous-détection (cartes ratées)
    n_overpred = int((out["n_diff"] > 0).sum())
    n_underpred = int((out["n_diff"] < 0).sum())
    n_match = int((out["n_diff"] == 0).sum())
    print(f"Images avec exactement le bon nombre de cartes : {n_match}/{n} ({n_match / n * 100:.0f}%)")
    print(f"Images en sur-détection (n_pred > n_gt)        : {n_overpred}/{n} ({n_overpred / n * 100:.0f}%)")
    print(f"Images en sous-détection (n_pred < n_gt)       : {n_underpred}/{n} ({n_underpred / n * 100:.0f}%)")
    avg_overpred = float(out[out["n_diff"] > 0]["n_diff"].mean()) if n_overpred > 0 else 0
    avg_underpred = float(out[out["n_diff"] < 0]["n_diff"].mean()) if n_underpred > 0 else 0
    print(f"Sur-détection moyenne : +{avg_overpred:.1f} cartes")
    print(f"Sous-détection moyenne : {avg_underpred:.1f} cartes")

    # Active player failures
    print(f"\n--- ÉCHECS ACTIVE PLAYER ({len(token_active_failures)} sur {n}) ---")
    confusion = Counter((f["gt_active"], f["pred_active"]) for f in token_active_failures)
    for (gt, pr), count in confusion.most_common(10):
        print(f"  GT={gt} -> Pred={pr} : {count}x")

    print(f"\n--- ÉCHECS CENTER ({n - total_centers_ok} sur {n}) ---")
    bad_centers = out[out["center_ok"] == 0]
    confusion_c = Counter((r["gt_center"], r["pred_center"]) for _, r in bad_centers.iterrows())
    for (gt, pr), count in confusion_c.most_common(10):
        print(f"  GT={gt} -> Pred={pr} : {count}x")

    print(f"\n-> CSV détaillé : {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
