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

# Replace with stem cleaning + sectioning
nb["cells"][brick7b_idx]["source"] = [
    "# ==============================================================\n",
    "# BRICK 7B: STEM CLEANING + SECTIONING (3DFin approach)\n",
    "# ==============================================================\n",
    "# Step 1: 2nd verticality pass — removes non-vertical material\n",
    "#         (branches, attached understory, foliage)\n",
    "# Step 2: Section-wise circle fitting — computes per-section\n",
    "#         diameters for each tree\n",
    "\n",
    "from src.core.trunk_validation import (\n",
    "    clean_stems, compute_stem_sections, StemCleaningConfig\n",
    ")\n",
    "\n",
    "# ---- CONFIGURATION ----\n",
    "stem_config = StemCleaningConfig(\n",
    "    # 2nd verticality pass\n",
    "    verticality_threshold=0.7,\n",
    "    verticality_scale=0.1,\n",
    "    voxel_resolution_xy=0.02,\n",
    "    voxel_resolution_z=0.02,\n",
    "    # Sectioning (like 3DFin)\n",
    "    section_len=0.2,             # distance between sections (m)\n",
    "    section_wid=0.05,            # half-width of section slice (m)\n",
    "    min_points_section=80,       # min points for circle fitting\n",
    "    r_min=DBH_MIN / 2,           # min valid radius\n",
    "    r_max=DBH_MAX / 2,           # max valid radius\n",
    "    n_sectors=16,\n",
    "    min_sectors=9,\n",
    "    sector_width=0.02,\n",
    "    inner_circle_ratio=0.5,\n",
    "    max_inner_points=5,\n",
    "    minimum_height=0.3,          # lowest section (m)\n",
    "    maximum_height=HEIGHT_MAX,    # highest section (m)\n",
    "    cluster_eps=0.02,\n",
    ")\n",
    "\n",
    "# ---- STEP 1: STEM CLEANING ----\n",
    "print(f\"Input: {trunk_result.n_trees} trees, \"\n",
    "      f\"{trunk_result.trunk_mask.sum():,} trunk points\")\n",
    "\n",
    "cleaning_result = clean_stems(veg_normalized, trunk_result, stem_config)\n",
    "\n",
    "# ---- STEP 2: SECTIONING ----\n",
    "section_result = compute_stem_sections(\n",
    "    veg_normalized,\n",
    "    cleaning_result.stem_mask,\n",
    "    trunk_result.tree_ids,\n",
    "    stem_config,\n",
    ")\n",
    "\n",
    "# ---- EXPORT cleaned stems for CloudCompare ----\n",
    "from src.core.io import export_point_cloud\n",
    "from pathlib import Path\n",
    "\n",
    "OUTPUT_DIR = Path(\"D:/OUTPUTS/T460298A\")\n",
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "clean_pts = veg_normalized[cleaning_result.stem_mask]\n",
    "clean_ids = trunk_result.tree_ids[cleaning_result.stem_mask]\n",
    "ids_clamped = np.clip(clean_ids, 0, 255).astype(np.uint8)\n",
    "\n",
    "export_point_cloud(\n",
    "    OUTPUT_DIR / \"trunks_validated.laz\",\n",
    "    clean_pts,\n",
    "    classification=ids_clamped,\n",
    "    point_format=6,\n",
    ")\n",
    "print(f\"\\n✓ Exported: {OUTPUT_DIR / 'trunks_validated.laz'}\")\n",
    "print(f\"  Points: {len(clean_pts):,}\")\n",
    "print(f\"  Removed: {cleaning_result.n_points_removed:,} non-vertical points\")\n",
]
nb["cells"][brick7b_idx]["outputs"] = []
nb["cells"][brick7b_idx]["execution_count"] = None

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Brick 7B updated at cell {brick7b_idx}")
