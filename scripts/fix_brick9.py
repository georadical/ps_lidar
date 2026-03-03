import json

path = "notebooks/01_playground.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "BRICK 9: EXPORT" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            # Add point_format=6 to all export_point_cloud calls
            line = line.replace(
                "classification=trunk_ids)",
                "classification=trunk_ids, point_format=6)"
            )
            line = line.replace(
                "classification=branch_ids)",
                "classification=branch_ids, point_format=6)"
            )
            line = line.replace(
                "classification=wood_ids)",
                "classification=wood_ids, point_format=6)"
            )
            new_source.append(line)
        cell["source"] = new_source
        print(f"Fixed cell {i}: added point_format=6")
        break

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
