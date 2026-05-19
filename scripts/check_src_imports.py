"""Vérifie que tous les modules src/ compilent et que main.py importe sans erreur."""
import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ok = True
for f in sorted(ROOT.joinpath("src").rglob("*.py")):
    try:
        ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e:
        print("SYNTAX FAIL", f, e)
        ok = False
print("src/ syntax:", "ALL OK" if ok else "ERRORS")

# Imports utilisés par main.py
for mod in ["src.config", "src.detector_inference", "src.inference",
            "src.detection", "src.detector", "src.model", "src.templates"]:
    try:
        importlib.import_module(mod)
        print(f"import {mod}: OK")
    except Exception as e:
        print(f"import {mod}: FAIL {type(e).__name__}: {e}")
        ok = False
print("=> ", "DELIVERABLE src/ OK" if ok else "FIX NEEDED")
