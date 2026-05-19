import nbformat
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
nb = nbformat.read(open(ROOT / "notebooks" / "report.ipynb"), 4)
code = [c for c in nb.cells if c.cell_type == "code"]
with_out = [c for c in code if c.get("outputs")]
errs = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
print(f"{len(with_out)}/{len(code)} code cells have outputs")
print(f"errors: {len(errs)}")
for c in code:
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            t = "".join(o.get("text", ""))
            for line in t.splitlines():
                if "SCORE" in line or "CenterAcc" in line or "TOTAL :" in line:
                    print(">>", line.strip())
