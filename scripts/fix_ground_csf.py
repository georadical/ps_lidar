import json

path = "notebooks/01_playground.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the Brick 4 cell (classify_ground call) and add extraction code
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "classify_ground(" in "".join(cell["source"]):
        src = "".join(cell["source"])
        if "vegetation_xyz" not in src:
            cell["source"].extend([
                "\n",
                "# Extract ground and vegetation point clouds\n",
                "ground_xyz = xyz[ground_result.ground_indices]\n",
                "vegetation_xyz = xyz[ground_result.off_ground_indices]\n",
                "\n",
                "print(f\"Ground points: {len(ground_xyz):,}\")\n",
                "print(f\"Vegetation points: {len(vegetation_xyz):,}\")\n",
            ])
            print(f"Added ground/vegetation extraction to cell {i}")
        else:
            print(f"Cell {i} already has vegetation_xyz")
        break

# Fix remaining Spanish text
for cell in nb["cells"]:
    new_source = []
    for line in cell["source"]:
        line = line.replace("Vegetación:", "Vegetation:")
        new_source.append(line)
    cell["source"] = new_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
