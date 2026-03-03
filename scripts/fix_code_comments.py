"""Fix the imports cell to match actual __init__.py exports."""
import json
from pathlib import Path

nb_path = Path("notebooks/01_playground.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "ground_filter_csf" in "".join(cell["source"]):
        cell["source"] = [
            "import os\n",
            "import sys\n",
            "import time\n",
            "import numpy as np\n",
            "\n",
            "# Resolve project root\n",
            "project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "if project_root not in sys.path:\n",
            "    sys.path.insert(0, project_root)\n",
            "\n",
            "from src.core import (\n",
            "    PointCloudLoader,\n",
            "    clip_circular_plot,\n",
            "    detect_normalization,\n",
            "    classify_ground,\n",
            "    normalize_heights,\n",
            "    export_point_cloud,\n",
            ")\n",
            "\n",
            "print(f\"Project root: {project_root}\")\n",
            "print(\"All imports loaded successfully.\")\n",
        ]
        print("Fixed imports cell: ground_filter_csf -> classify_ground")
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
