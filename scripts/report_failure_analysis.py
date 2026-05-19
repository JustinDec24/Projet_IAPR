"""Lit outputs/failure_analysis.csv et imprime le rapport complet."""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

df = pd.read_csv(ROOT / "outputs" / "failure_analysis.csv")
n = len(df)

tp_j, fp_j, fn_j = df["tp_j"].sum(), df["fp_j"].sum(), df["fn_j"].sum()
tp_l, fp_l, fn_l = df["tp_l"].sum(), df["fp_l"].sum(), df["fn_l"].sum()
f1_j = 2 * tp_j / max(2 * tp_j + fp_j + fn_j, 1)
f1_l = 2 * tp_l / max(2 * tp_l + fp_l + fn_l, 1)
c_acc = df["center_ok"].mean()
a_acc = df["active_ok"].mean()
score = 0.1 * c_acc + 0.1 * a_acc + 0.8 * f1_j

print(f"=== SCORE GLOBAL : {score:.3f} ===")
print(f"  CenterAcc {c_acc:.3f}  ActiveAcc {a_acc:.3f}  F1_joint {f1_j:.3f}  F1_labels {f1_l:.3f}")
print(f"  TP={tp_j}  FP={fp_j}  FN={fn_j}")

print()
print("=== ORACLE SCORES (gain max si composant fixe a 100%) ===")
print(f"  +F1 detection+classif parfait : {0.1 * c_acc + 0.1 * a_acc + 0.8:.3f}  gain +{0.8 * (1.0 - f1_j):.3f}")
print(f"  +Player assignment parfait    : {0.1 * c_acc + 0.1 * a_acc + 0.8 * f1_l:.3f}  gain +{0.8 * (f1_l - f1_j):.3f}")
print(f"  +CenterAcc = 1.0              : {0.1 + 0.1 * a_acc + 0.8 * f1_j:.3f}  gain +{0.1 * (1 - c_acc):.3f}")
print(f"  +ActiveAcc = 1.0              : {0.1 * c_acc + 0.1 + 0.8 * f1_j:.3f}  gain +{0.1 * (1 - a_acc):.3f}")

print()
print("=== DETECTION vs CLASSIFICATION ===")
n_match = int((df["n_diff"] == 0).sum())
n_over = int((df["n_diff"] > 0).sum())
n_under = int((df["n_diff"] < 0).sum())
print(f"  Images avec bon nombre de cartes : {n_match}/{n} ({100 * n_match / n:.0f}%)")
print(f"  Images sur-detection             : {n_over}/{n} ({100 * n_over / n:.0f}%)  avg +{df[df.n_diff > 0].n_diff.mean():.1f}")
print(f"  Images sous-detection            : {n_under}/{n} ({100 * n_under / n:.0f}%)  avg {df[df.n_diff < 0].n_diff.mean():.1f}")
print(f"  Total sur-comptees   : {df[df.n_diff > 0].n_diff.sum()}")
print(f"  Total sous-comptees  : {-df[df.n_diff < 0].n_diff.sum()}")

print()
print("=== TOP 15 PIRES F1_joint (priorite debug) ===")
print(df.nsmallest(15, "f1_joint")[
    ["image_id", "n_gt", "n_pred", "f1_joint", "f1_labels", "center_ok", "active_ok"]
].to_string(index=False))

print()
print("=== CONFUSION ACTIVE PLAYER ===")
bad = df[df["active_ok"] == 0]
conf = Counter(zip(bad["gt_active"], bad["pred_active"]))
for k, v in conf.most_common():
    print(f"  GT={k[0]:>5} -> Pred={k[1]:>5} : {v}x")

print()
print("=== CONFUSION CENTER ===")
bad_c = df[df["center_ok"] == 0]
conf_c = Counter(zip(bad_c["gt_center"], bad_c["pred_center"]))
for k, v in conf_c.most_common()[:10]:
    print(f"  GT={k[0]:>10} -> Pred={k[1]:>10} : {v}x")
