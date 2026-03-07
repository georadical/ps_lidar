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
    print("ERROR: Could not find trunk_config cell")
    exit(1)

# Find the line with TrunkExtractionConfig and add cluster validation params
source = nb["cells"][brick7_idx]["source"]
new_source = []
for line in source:
    new_source.append(line)
    # After stem_search_radius line, add cluster validation params
    if "stem_search_radius" in line and "STEM_SEARCH_RADIUS" in line:
        new_source.append("    # ---- CLUSTER VALIDATION (rejects understory/regeneration) ----\n")
        new_source.append("    cluster_circularity_min=0.3,       # min XY circularity\n")
        new_source.append("    cluster_diameter_max_factor=1.5,    # max cluster diam = DBH_MAX * factor\n")
        new_source.append("    cluster_min_height=2.0,             # min stripe height for small trees\n")
        new_source.append("    cluster_min_diameter=0.05,           # min diameter (metres)\n")

nb["cells"][brick7_idx]["source"] = new_source
nb["cells"][brick7_idx]["outputs"] = []
nb["cells"][brick7_idx]["execution_count"] = None

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Updated Brick 7 at cell {brick7_idx} with cluster validation params")
