import json

path = "notebooks/01_playground.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Brick 7 cell
brick7_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "trunk_config" in src and "TrunkExtractionConfig" in src:
        brick7_idx = i
        break

if brick7_idx is None:
    print("ERROR")
    exit(1)

# Check if adaptive params already present
source = nb["cells"][brick7_idx]["source"]
src_text = "".join(source)

if "trunk_section_height" in src_text:
    print("Adaptive params already present, skipping")
    exit(0)

# Insert adaptive trunk search params after cluster validation params
new_source = []
for line in source:
    new_source.append(line)
    if "cluster_min_diameter" in line:
        new_source.append("    # ---- ADAPTIVE TRUNK SEARCH (per-tree, per-section) ----\n")
        new_source.append("    trunk_section_height=0.3,          # metres — section height for radius fitting\n")
        new_source.append("    radius_offset_pct=0.10,            # 10% outward offset\n")
        new_source.append("    min_offset_abs=0.02,               # metres — minimum 2cm offset\n")

nb["cells"][brick7_idx]["source"] = new_source
nb["cells"][brick7_idx]["outputs"] = []
nb["cells"][brick7_idx]["execution_count"] = None

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Updated Brick 7 at cell {brick7_idx} with adaptive params")
