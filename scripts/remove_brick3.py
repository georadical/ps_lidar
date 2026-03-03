import json
from pathlib import Path

nb_path = Path("notebooks/01_playground.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Identify Brick 3 cells to remove
cells_to_keep = []
skip_next = False
for i, cell in enumerate(nb["cells"]):
    if skip_next:
        print(f"Removing code cell {i} (Brick 3 implementation)")
        skip_next = False
        continue
        
    content = "".join(cell["source"])
    if cell["cell_type"] == "markdown" and "3. Noise Filtering" in content:
        print(f"Removing markdown cell {i}: {content.strip()[:50]}...")
        skip_next = True
        continue
    
    cells_to_keep.append(cell)

nb["cells"] = cells_to_keep

# 2. Renumber remaining bricks in Markdown and Code comments
replacements = {
    "Brick 4": "Brick 3",
    "Brick 5": "Brick 4",
    "Brick 6": "Brick 5",
    "BRICK 7": "BRICK 6",
    "Brick 7": "Brick 6",
    "BRICK 8": "BRICK 7",
    "Brick 8": "Brick 7",
    "BRICK 9": "BRICK 8",
    "Brick 9": "Brick 8",
    "Brick 10": "Brick 9",
    "Brick 11": "Brick 10",
    "Brick 12": "Brick 11",
    
    # Header numbering
    "## 4. Normalisation": "## 3. Normalisation",
    "## 5. Ground": "## 4. Ground",
    "## 6. Height": "## 5. Height",
    "## 6.5 Export": "## 5.5 Export",
    "## 6.5 3D Visual": "## 5.5 3D Visual",
    
    # Remove unused import
    ", filter_noise_sor": "",
}

# TOC specific replacements
toc_replacements = {
    "- Brick 3: Noise Filtering (SOR)\n": "",
    "- Brick 4:": "- Brick 3:",
    "- Brick 5:": "- Brick 4:",
    "- Brick 6:": "- Brick 5:",
    "- Brick 7:": "- Brick 6:",
    "- Brick 8:": "- Brick 7:",
    "- Brick 9:": "- Brick 8:",
    "- Brick 10:": "- Brick 9:",
}

n_replacements = 0
for cell in nb["cells"]:
    new_source = []
    for line in cell["source"]:
        orig = line
        
        # Apply specific ToC replacements if it's the first cell
        if cell == nb["cells"][0]:
            for old, new in toc_replacements.items():
                line = line.replace(old, new)
        
        # Apply general replacements
        for old, new in replacements.items():
            line = line.replace(old, new)
            
        if line != orig:
            n_replacements += 1
        new_source.append(line)
    cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Removed noise filtering cells. Made {n_replacements} renumbering replacements.")
