import json

path = "notebooks/01_playground.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "BRICK 7: TRUNK EXTRACTION" in "".join(cell["source"]):
        cell["source"] = [
            "# ==============================================================\n",
            "# BRICK 7: TRUNK EXTRACTION\n",
            "# ==============================================================\n",
            "\n",
            "import os, sys, time\n",
            "import numpy as np\n",
            "import laspy\n",
            "from pathlib import Path\n",
            "\n",
            "# Ensure project root is in path\n",
            "project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "if project_root not in sys.path:\n",
            "    sys.path.insert(0, project_root)\n",
            "\n",
            "from src.core.trunk_extraction import extract_trunks, TrunkExtractionConfig\n",
            "\n",
            "# ---- FIELD-MEASURED PARAMETERS (modify per plot) ----\n",
            "\n",
            "# Tree dimensions\n",
            "DBH_MIN = 0.20          # metres — smallest expected stem diameter\n",
            "DBH_MAX = 0.80          # metres — largest expected stem diameter\n",
            "HEIGHT_MAX = 36.0       # metres — tallest tree in the plot\n",
            "\n",
            "# Breast-height detection band\n",
            "STRIPE_LOWER = 2.0      # metres — lower limit of detection stripe\n",
            "STRIPE_UPPER = 6.0      # metres — upper limit of detection stripe\n",
            "\n",
            "# Crown distances (horizontal distance from trunk to tip of longest\n",
            "# branch, measured per cardinal direction on the central tree)\n",
            "CROWN_DISTANCE_A = 8.0  # metres — North\n",
            "CROWN_DISTANCE_B = 7.0  # metres — East\n",
            "CROWN_DISTANCE_C = 6.5  # metres — South\n",
            "CROWN_DISTANCE_D = 7.5  # metres — West\n",
            "\n",
            "# Derived: max crown radius = axis assignment distance\n",
            "MAX_AXIS_DISTANCE = max(CROWN_DISTANCE_A, CROWN_DISTANCE_B,\n",
            "                        CROWN_DISTANCE_C, CROWN_DISTANCE_D)\n",
            "# Derived: stem search radius (half the max DBH, with a small margin)\n",
            "STEM_SEARCH_RADIUS = (DBH_MAX / 2) + 0.10\n",
            "\n",
            "print(f\"Crown distances: A={CROWN_DISTANCE_A}m, B={CROWN_DISTANCE_B}m, \"\n",
            "      f\"C={CROWN_DISTANCE_C}m, D={CROWN_DISTANCE_D}m\")\n",
            "print(f\"Max axis distance (derived): {MAX_AXIS_DISTANCE}m\")\n",
            "print(f\"Stem search radius (derived): {STEM_SEARCH_RADIUS}m\")\n",
            "\n",
            "# ---- ALGORITHM PARAMETERS ----\n",
            "\n",
            "VOXEL_RESOLUTION = 0.05   # metres\n",
            "VERTICALITY_THRESH = 0.7  # 0-1 (higher = stricter)\n",
            "PEELING_ITERATIONS = 2    # passes of verticality peeling\n",
            "MIN_CLUSTER_PTS = 500     # minimum voxels per stem cluster\n",
            "\n",
            "# ---- LOAD CHECKPOINT ----\n",
            "\n",
            "VEG_CHECKPOINT = Path(\"D:/OUTPUTS/T460298A_VEG_NORM.laz\")\n",
            "\n",
            "print(f\"\\nLoading: {VEG_CHECKPOINT.name}\")\n",
            "t0 = time.perf_counter()\n",
            "las = laspy.read(str(VEG_CHECKPOINT))\n",
            "veg_normalized = np.column_stack([\n",
            "    np.array(las.x, dtype=np.float32),\n",
            "    np.array(las.y, dtype=np.float32),\n",
            "    np.array(las.z, dtype=np.float32),\n",
            "])\n",
            "print(f\"  {len(veg_normalized):,} points in {time.perf_counter()-t0:.1f}s\")\n",
            "print(f\"  Z range: {veg_normalized[:,2].min():.2f}m to {veg_normalized[:,2].max():.2f}m\")\n",
            "\n",
            "# ---- RUN EXTRACTION ----\n",
            "\n",
            "trunk_config = TrunkExtractionConfig(\n",
            "    stripe_lower_limit=STRIPE_LOWER,\n",
            "    stripe_upper_limit=STRIPE_UPPER,\n",
            "    dbh_min=DBH_MIN,\n",
            "    dbh_max=DBH_MAX,\n",
            "    height_max=HEIGHT_MAX,\n",
            "    max_axis_distance=MAX_AXIS_DISTANCE,\n",
            "    stem_search_radius=STEM_SEARCH_RADIUS,\n",
            "    voxel_resolution_xy=VOXEL_RESOLUTION,\n",
            "    voxel_resolution_z=VOXEL_RESOLUTION,\n",
            "    verticality_threshold=VERTICALITY_THRESH,\n",
            "    peeling_iterations=PEELING_ITERATIONS,\n",
            "    min_cluster_points=MIN_CLUSTER_PTS,\n",
            ")\n",
            "\n",
            "print(f\"\\nInput points: {len(veg_normalized):,}\")\n",
            "trunk_result = extract_trunks(veg_normalized, trunk_config, verbose=True)\n",
            "print(f\"\\nTrees found: {trunk_result.n_trees}\")\n",
        ]
        print(f"Updated Brick 7 cell {i}")
        break

# Also fix the continuation cell to use trunk_mask
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "ASSIGN ALL POINTS" in "".join(cell["source"]):
        cell["source"] = [
            "# ==============================================================\n",
            "# BRICK 7 (cont): SEPARATE TRUNK vs NON-TRUNK\n",
            "# ==============================================================\n",
            "\n",
            "# trunk_mask = points within stem_search_radius of an axis\n",
            "# tree_ids = which tree each point belongs to (-1 = unassigned)\n",
            "\n",
            "trunk_points = veg_normalized[trunk_result.trunk_mask]\n",
            "non_trunk_mask = ~trunk_result.trunk_mask\n",
            "non_trunk_points = veg_normalized[non_trunk_mask]\n",
            "\n",
            "print(f\"Trunk points:     {len(trunk_points):,} ({trunk_result.trunk_mask.mean():.1%})\")\n",
            "print(f\"Non-trunk points: {len(non_trunk_points):,}\")\n",
            "print(f\"Assigned to a tree: {(trunk_result.tree_ids >= 0).sum():,}\")\n",
            "print(f\"Unassigned:         {(trunk_result.tree_ids == -1).sum():,}\")\n",
        ]
        print(f"Updated continuation cell {i}")
        break

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
