"""Final verification: show all markdown cells and check for Spanish."""
import json, re
from pathlib import Path

nb = json.load(open(Path("notebooks/01_playground.ipynb"), encoding="utf-8"))

print(f"Total cells: {len(nb['cells'])}")
print("\n=== MARKDOWN CELLS ===")
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown":
        content = "".join(c["source"]).strip()
        print(f"[{i:2d}] {content[:80]}")

print("\n=== SPANISH CHECK ===")
spanish = [
    r'\b[Ee]jecutar', r'\b[Rr]eemplazar', r'\b[Dd]irectorio\b', r'\bcampos\b',
    r'\b[Ee]scalares', r'\b[Dd]isponible', r'\b[Rr]ecorte\b',
    r'\b[Hh]erramienta\b', r'\b[Vv]alores\b', r'\b[Mm]emoria\b',
    r'\bparámetros\b', r'\bÁrboles\b', r'\b[Nn]ormalización',
    r'\b[Vv]isualización', r'\b[Ss]otobosque\b', r'\b[Ll]iberar\b',
    r'\b[Aa]gregar\b', r'\bInstrucciones\b', r'\bNoise Filtering\b',
    r'\b[Ff]iltering de\b', r'PARÁMETROS',
]

found = False
for i, cell in enumerate(nb["cells"]):
    for j, line in enumerate(cell["source"]):
        for pat in spanish:
            if re.search(pat, line, re.IGNORECASE):
                found = True
                print(f"  [{i}:{j}] {line.strip()[:80]}")
                break

if not found:
    print("  All clean! No Spanish text remaining.")
