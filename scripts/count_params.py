"""Phase 7 V3 — Compte des paramètres (contrainte stricte < 12M total)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v3.corner_classifier import CornerClassifier, count_parameters as cc
from src.v3.corner_detector import CornerDetector, count_parameters as cd

det = CornerDetector(channels=(40, 80, 160, 320))
clf = CornerClassifier(channels=(48, 96, 160, 224))
n_det = cd(det)
n_clf = cc(clf)
total = n_det + n_clf

print("=" * 48)
print("V3 — Compte des paramètres")
print("=" * 48)
print(f"  Corner detector  [40,80,160,320] : {n_det:>10,}  ({n_det/1e6:.2f}M)")
print(f"  Corner classifier [48,96,160,224]: {n_clf:>10,}  ({n_clf/1e6:.2f}M)")
print(f"  {'TOTAL':<33}: {total:>10,}  ({total/1e6:.2f}M)")
print("=" * 48)
limit = 12_000_000
status = "OK" if total < limit else "DEPASSEMENT"
print(f"  Limite 12M : {status}  (marge {(limit-total)/1e6:.2f}M)")
sys.exit(0 if total < limit else 1)
