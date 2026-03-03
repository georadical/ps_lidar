"""Full audit of notebook: show ALL code cells with complete source."""
import json

with open("notebooks/01_playground.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    ct = c["cell_type"]
    src = "".join(c["source"])
    print(f"\n{'='*60}")
    print(f"=== CELL {i} ({ct}) ===")
    print(f"{'='*60}")
    print(src[:500])
    if len(src) > 500:
        print(f"  ... ({len(src)} chars total)")
