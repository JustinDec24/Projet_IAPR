"""Télécharge le dataset depuis Kaggle dans data/.

Nécessite les credentials Kaggle :
- soit ~/.kaggle/kaggle.json
- soit les variables d'env KAGGLE_USERNAME et KAGGLE_KEY
"""

import shutil
from pathlib import Path

import kagglehub

DEST = Path(__file__).resolve().parent.parent / "data"

print("Téléchargement en cours...")
src = Path(kagglehub.competition_download("iapr-26-uno-vision-challenge"))
print(f"Téléchargé dans : {src}")

DEST.mkdir(exist_ok=True)
for item in src.iterdir():
    target = DEST / item.name
    if target.exists():
        print(f"Skip (déjà présent) : {target.name}")
        continue
    if item.is_dir():
        shutil.copytree(item, target)
    else:
        shutil.copy2(item, target)
    print(f"Copié : {item.name} -> {target}")

print("Terminé.")
