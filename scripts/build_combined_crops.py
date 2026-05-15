"""Fusionne real_crops (heuristique) + real_crops_v3 (corner) → diversité
de styles de crops pour rendre le classifier robuste centre ET joueurs."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src1 = ROOT / "data" / "real_crops"
src2 = ROOT / "data" / "real_crops_v3"
dst = ROOT / "data" / "real_crops_combined"
if dst.exists():
    shutil.rmtree(dst)
n = 0
for src, tag in [(src1, "h"), (src2, "c")]:
    if not src.exists():
        continue
    for ld in src.iterdir():
        if not ld.is_dir():
            continue
        od = dst / ld.name
        od.mkdir(parents=True, exist_ok=True)
        for f in ld.glob("*.png"):
            shutil.copy(f, od / f"{tag}_{f.name}")
            n += 1
print(f"combined crops: {n}  classes: {len(list(dst.iterdir()))}")
