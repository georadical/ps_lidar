import json
from pathlib import Path

notebook_path = Path('notebooks/01_playground.ipynb')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('BRICK 7B: STEM CLEANING + SECTIONING' in line for line in cell['source']):
        source = "".join(cell['source'])
        
        # Add imports
        source = source.replace(
            "from src.core.trunk_validation import (\n    clean_stems, compute_stem_sections, StemCleaningConfig\n)",
            "from src.core.trunk_validation import (\n    clean_stems, compute_stem_sections, StemCleaningConfig,\n    filter_trees, TreeFilterConfig\n)"
        )
        
        # Add filtering code right before export
        export_section = "# ---- EXPORT cleaned stems for CloudCompare ----"
        filtering_code = """
# ---- STEP 3: TREE-LEVEL FILTERS (Height & Edge) ----
filter_config = TreeFilterConfig(
    min_height=10.0,
    plot_center_x=CENTER_X,
    plot_center_y=CENTER_Y,
    max_distance_from_center=PLOT_RADIUS - 0.5,
)

filter_result = filter_trees(
    veg_normalized,
    cleaning_result.stem_mask,
    trunk_result.tree_ids,
    filter_config,
)

"""
        source = source.replace(export_section, filtering_code + export_section)
        
        # Update point selection
        source = source.replace(
            "clean_pts = veg_normalized[cleaning_result.stem_mask]",
            "clean_pts = veg_normalized[filter_result.stem_mask]"
        )
        source = source.replace(
            "clean_ids = trunk_result.tree_ids[cleaning_result.stem_mask]",
            "clean_ids = filter_result.tree_ids[filter_result.stem_mask]"
        )
        
        # Re-split by lines
        lines = source.splitlines(keepends=True)
        cell['source'] = lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
