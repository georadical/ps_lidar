import json

path = "notebooks/01_playground.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Brick 7 cell and show the trunk_config section
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "trunk_config" in src and "TrunkExtractionConfig" in src:
        print(f"=== Cell {i} ===")
        for j, line in enumerate(cell["source"]):
            print(f"  {j}: {line.rstrip()}")
        break
