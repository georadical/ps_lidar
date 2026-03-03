import json
from pathlib import Path

nb_path = Path("notebooks/01_playground.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scripts/test_pipeline.py", "w", encoding="utf-8") as out:
    out.write("import os, sys, time\n")
    out.write("sys.path.append(os.path.abspath('.'))\n")
    out.write("from src.core import PointCloudLoader, clip_circular_plot, filter_noise_sor\n\n")
    
    # Brick 1
    out.write("print('--- BRICK 1: Loading ---')\n")
    out.write("".join(nb["cells"][3]["source"]) + "\n")
    out.write("".join(nb["cells"][4]["source"]) + "\n")
    
    # Brick 2
    out.write("print('\\n--- BRICK 2: Clipping ---')\n")
    out.write("".join(nb["cells"][7]["source"]) + "\n")
    out.write("".join(nb["cells"][8]["source"]) + "\n")
    
    # Brick 3
    out.write("print('\\n--- BRICK 3: Noise Filtering ---')\n")
    out.write("".join(nb["cells"][10]["source"]) + "\n")

print("Created scripts/test_pipeline.py")
