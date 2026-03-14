import json
from pathlib import Path

notebook_path = Path('notebooks/01_playground.ipynb')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('BRICK 7B: STEM CLEANING + SECTIONING' in line for line in cell['source']):
        source = "".join(cell['source'])
        
        # Remove min_height from config
        source = source.replace("    min_height=10.0,\n", "")
        
        # Remove section_result from filter_trees call
        old_call = """filter_result = filter_trees(
    veg_normalized,
    cleaning_result.stem_mask,
    trunk_result.tree_ids,
    section_result,
    filter_config,
)"""
        new_call = """filter_result = filter_trees(
    veg_normalized,
    cleaning_result.stem_mask,
    trunk_result.tree_ids,
    filter_config,
)"""
        if old_call in source:
            source = source.replace(old_call, new_call)
        
        # Re-split by lines
        lines = source.splitlines(keepends=True)
        cell['source'] = lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully: Height filter removed.")
