"""
Comprehensive one-pass fix for ALL notebook issues:
1. Add 'from pathlib import Path' to Cell 1 imports
2. Fix remaining Spanish text in all cells
3. Ensure export cell has Path import
4. Fix normalisation cell Spanish text
"""
import json

path = "notebooks/01_playground.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ============================================================
# FIX 1: Add Path import to Cell 1
# ============================================================
cell1 = nb["cells"][1]
src1 = "".join(cell1["source"])
if "from pathlib import Path" not in src1:
    # Add Path import after numpy
    new_source = []
    for line in cell1["source"]:
        new_source.append(line)
        if "import numpy as np" in line:
            new_source.append("from pathlib import Path\n")
    cell1["source"] = new_source
    print("FIX 1: Added 'from pathlib import Path' to Cell 1")

# ============================================================
# FIX 2: All remaining Spanish text across ALL cells
# ============================================================
spanish_fixes = [
    # Cell 4 markdown
    ("`CENTER_X` y `CENTER_Y` abajo", "`CENTER_X` and `CENTER_Y` below"),
    # Cell 5 comments
    ("# USER PARAMETERS - Modificar según el plot", "# USER PARAMETERS - Modify per plot"),
    ("Modificar seg·n el plot", "Modify per plot"),
    ("# Coordenadas del centro (obtenidas de CloudCompare Point Picking)", "# Centre coordinates (from CloudCompare Point Picking)"),
    ("# Radio del plot en metros", "# Plot radius in metres"),
    ("Centro:", "Centre:"),
    ("Radio:", "Radius:"),
    # Cell 9
    ("Estatus:", "Status:"),
    ("Normalised?:", "Normalised:"),
    ("Rango Z:", "Z range:"),
    (" a ", " to "),  # "Xm a Ym" -> "Xm to Ym" 
    # Cell 14
    ("Vegetación:", "Vegetation:"),
    # Cell 16
    ("# Export vegetación", "# Export vegetation"),
    ("# Export vegetaci≤n", "# Export vegetation"),
    # General
    ("Nube:", "Cloud:"),
    ("Vegetación", "Vegetation"),
]

n_fixes = 0
for cell in nb["cells"]:
    new_source = []
    for line in cell["source"]:
        orig = line
        for old, new in spanish_fixes:
            line = line.replace(old, new)
        if line != orig:
            n_fixes += 1
        new_source.append(line)
    cell["source"] = new_source

print(f"FIX 2: Applied {n_fixes} Spanish->English fixes")

# ============================================================
# FIX 3: Add Path import to export cell (Cell 16) if missing
# ============================================================
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "OUTPUT_DIR = Path(" in src and "from pathlib import Path" not in src and "import Path" not in src:
            cell["source"].insert(0, "from pathlib import Path\n")
            print(f"FIX 3: Added Path import to export cell")

# ============================================================
# SAVE
# ============================================================
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nAll fixes applied! Done.")
