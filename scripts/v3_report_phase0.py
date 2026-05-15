"""Imprime la synthèse Phase 0 depuis reports/v3_failure_analysis.csv."""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "reports" / "v3_failure_analysis.csv")
n = len(df)

fn = df["FN_detection"].sum()
fp = df["FP_detection"].sum()
mis = df["misclassification"].sum()
wp = df["wrong_player"].sum()
ce = df["center_error"].sum()
ae = df["active_error"].sum()
total_gt = df["n_gt"].sum()
total_pred = df["n_pred"].sum()
card_err = fn + fp + mis + wp

center_acc = 1 - ce / n
active_acc = 1 - ae / n

print(f"=== PHASE 0 — Baseline 0.764 (n={n}, {total_gt} cartes GT) ===\n")
print(f"CenterAcc : {center_acc:.3f}  ({ce} err)")
print(f"ActiveAcc : {active_acc:.3f}  ({ae} err)")
print(f"n_pred total : {total_pred}  vs  n_gt total : {total_gt}\n")
print("ERREURS CARTES catégorisées :")
for name, v in [("FN_detection (carte ratée)", fn),
                ("FP_detection (hallucination)", fp),
                ("misclassification (mauvais label)", mis),
                ("wrong_player (bon label, mauvaise zone)", wp)]:
    print(f"  {name:42s} : {v:4d}  ({100*v/max(card_err,1):5.1f}%)")
print(f"  {'TOTAL':42s} : {card_err:4d}")

print("\nCONFUSION CENTER (top 10) :")
bad = df[df.center_error == 1]
for (g, p), c in Counter(zip(bad.gt_center, bad.pred_center)).most_common(10):
    print(f"  {g:>10} -> {p:>10} : {c}x")

print("\nCONFUSION ACTIVE (top 10) :")
bada = df[df.active_error == 1]
for (g, p), c in Counter(zip(bada.gt_active, bada.pred_active)).most_common(10):
    print(f"  {g:>5} -> {p:>5} : {c}x")

# Combien d'images parfaites
perfect = df[(df.FN_detection == 0) & (df.FP_detection == 0)
             & (df.misclassification == 0) & (df.wrong_player == 0)
             & (df.center_error == 0) & (df.active_error == 0)]
print(f"\nImages parfaites : {len(perfect)}/{n} ({100*len(perfect)/n:.0f}%)")
