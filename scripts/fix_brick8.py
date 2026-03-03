import json

path = "notebooks/01_playground.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "BRICK 8: BRANCH EXTRACTION" in "".join(cell["source"]):
        cell["source"] = [
            "# ==============================================================\n",
            "# BRICK 8: BRANCH EXTRACTION\n",
            "# ==============================================================\n",
            "\n",
            "from src.core.branch_extraction import extract_branches, BranchExtractionConfig\n",
            "\n",
            "# ---- FIELD-MEASURED PARAMETERS ----\n",
            "MAX_BRANCH_LENGTH = 8.0     # metres — longest expected branch\n",
            "\n",
            "# ---- ALGORITHM PARAMETERS ----\n",
            "LINEARITY_THRESH = 0.5      # 0-1 (higher = stricter)\n",
            "CONNECTIVITY_RADIUS = 0.05  # metres (voxel size for connectivity graph)\n",
            "MIN_BRANCH_POINTS = 50      # minimum points per branch cluster\n",
            "\n",
            "branch_config = BranchExtractionConfig(\n",
            "    max_branch_length=MAX_BRANCH_LENGTH,\n",
            "    linearity_threshold=LINEARITY_THRESH,\n",
            "    connectivity_radius=CONNECTIVITY_RADIUS,\n",
            "    min_branch_points=MIN_BRANCH_POINTS,\n",
            ")\n",
            "\n",
            "print(f\"Branch config: {branch_config}\")\n",
            "print(f\"Input points: {len(veg_normalized):,}\")\n",
            "print(f\"Trunk points: {trunk_result.trunk_mask.sum():,}\")\n",
            "\n",
            "branch_result = extract_branches(\n",
            "    veg_normalized,\n",
            "    trunk_result,\n",
            "    branch_config,\n",
            "    verbose=True\n",
            ")\n",
            "\n",
            "print(f\"\\nBranch points: {branch_result.n_branch_points:,}\")\n",
            "print(f\"Wood (trunk+branch): {branch_result.wood_mask.sum():,} \"\n",
            "      f\"({branch_result.wood_mask.mean():.1%})\")\n",
        ]
        print(f"Fixed Brick 8 cell {i}")
        break

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
