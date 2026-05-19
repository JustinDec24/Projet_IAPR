"""Insère l'en-tête groupe/auteurs en tête du notebook, sans toucher
aux outputs exécutés. Vérifie aussi que le contenu est en anglais."""
import nbformat
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "report.ipynb"

HEADER = (
    "# [IAPR][iapr]: Lab 2 ‒  Object description\n"
    "\n"
    "\n"
    "**Group ID:** 12\n"
    "\n"
    "**Author 1 (sciper):** Ilia Badanin (xxxxx)  \n"
    "**Author 2 (sciper):** Alexandre Boidi (361116)   \n"
    "**Author 3 (sciper):** Justin Décaillet (355943)             \n"
    "**Author 4 (sciper):** Arthur Moscheni (xxxxx)   \n"
    "\n"
    "\n"
    "**Release date:** 18.03.2026  \n"
    "**Due date:** 01.04.2026 (11:59 pm)\n"
)

nb = nbformat.read(open(NB, encoding="utf-8"), 4)

# Sanity : le 1er markdown doit être le titre anglais du rapport
first = "".join(nb.cells[0]["source"])[:80]
print("Cell 0 starts with:", repr(first))
assert "Final Report" in first or "UNO Vision" in first, "structure inattendue"

# Évite un doublon si déjà inséré
if "[IAPR][iapr]: Lab 2" in "".join(nb.cells[0].get("source", "")):
    print("Header déjà présent — rien à faire.")
else:
    hdr = nbformat.v4.new_markdown_cell(HEADER)
    nb.cells.insert(0, hdr)
    nbformat.write(nb, open(NB, "w", encoding="utf-8"))
    print(f"Header inséré. Notebook : {len(nb.cells)} cellules.")

# Vérif rapide : pas de texte FR résiduel dans les markdown
fr_markers = ["éé", " le pipeline", " été ", "Détecteur", "cartes ", "réel"]
md = " ".join("".join(c["source"]) for c in nb.cells if c["cell_type"] == "markdown")
hits = [m for m in ["Détecteur", " réel", "cartes joueurs", " été "] if m in md]
print("FR markers found in markdown:", hits if hits else "none (English OK)")
