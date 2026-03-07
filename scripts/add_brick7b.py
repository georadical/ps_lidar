import json

path = "notebooks/01_playground.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the Brick 7B cell
brick7b_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "BRICK 7B" in src:
        brick7b_idx = i
        break

if brick7b_idx is None:
    print("ERROR: Could not find BRICK 7B cell")
    exit(1)

print(f"Found Brick 7B at cell {brick7b_idx}")

# Replace with updated version
nb["cells"][brick7b_idx]["source"] = [
    "# ==============================================================\n",
    "# BRICK 7B: TRUNK SCRUBBING (Stage A)\n",
    "# ==============================================================\n",
    "# Point-level cylinder scrubbing: for each tree, slices trunk into\n",
    "# horizontal sections, fits a circle per section, and keeps only\n",
    "# points within the fitted radius + offset. Attached understory\n",
    "# and branches are removed point-by-point.\n",
    "\n",
    "from src.core.trunk_validation import scrub_trunks, TrunkScrubConfig\n",
    "\n",
    "scrub_config = TrunkScrubConfig(\n",
    "    section_height=1.0,           # metres per horizontal slice\n",
    "    radius_offset=0.03,           # metres — extra tolerance beyond fitted radius\n",
    "    min_points_per_section=30,    # min points per slice for fitting\n",
    "    min_trunk_points_after=200,   # min total trunk points to keep tree\n",
    "    min_trunk_height=5.0,         # metres — min vertical extent of trunk\n",
    "    dbh_max=DBH_MAX,              # from field measurements (Brick 7)\n",
    "    safety_factor=1.5,            # max section radius = DBH_MAX * factor / 2\n",
    "    percentile=75.0,              # robust radius estimation percentile\n",
    ")\n",
    "\n",
    "print(f\"Input: {trunk_result.n_trees} trees, \"\n",
    "      f\"{trunk_result.trunk_mask.sum():,} trunk points\")\n",
    "\n",
    "scrub_result = scrub_trunks(veg_normalized, trunk_result, scrub_config)\n",
    "\n",
    "print(f\"\\n--- Results ---\")\n",
    "print(f\"Trees:  {scrub_result.n_trees_before} → {scrub_result.n_trees_after}\")\n",
    "print(f\"Points: {scrub_result.total_points_before:,} → {scrub_result.total_points_after:,}\")\n",
    "print(f\"Scrubbed: {scrub_result.total_points_scrubbed:,} points removed\")\n",
    "\n",
    "# Export scrubbed trunks for CloudCompare inspection\n",
    "from src.core.io import export_point_cloud\n",
    "from pathlib import Path\n",
    "\n",
    "OUTPUT_DIR = Path(\"D:/OUTPUTS/T460298A\")\n",
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "scrubbed_trunk_pts = veg_normalized[scrub_result.trunk_mask]\n",
    "scrubbed_trunk_ids = scrub_result.tree_ids[scrub_result.trunk_mask]\n",
    "ids_clamped = np.clip(scrubbed_trunk_ids, 0, 255).astype(np.uint8)\n",
    "\n",
    "export_point_cloud(\n",
    "    OUTPUT_DIR / \"trunks_validated.laz\",\n",
    "    scrubbed_trunk_pts,\n",
    "    classification=ids_clamped,\n",
    "    point_format=6,\n",
    ")\n",
    "print(f\"\\n✓ Exported: {OUTPUT_DIR / 'trunks_validated.laz'}\")\n",
    "print(f\"  Points: {len(scrubbed_trunk_pts):,}\")\n",
    "print(f\"  Trees:  {scrub_result.n_trees_after}\")\n",
]
nb["cells"][brick7b_idx]["outputs"] = []
nb["cells"][brick7b_idx]["execution_count"] = None

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Brick 7B updated at cell {brick7b_idx}")
