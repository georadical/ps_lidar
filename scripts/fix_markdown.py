"""
Fix remaining Spanish markdown cells in Bricks 1-6.
"""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "01_playground.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Direct cell-by-cell fixes for cells with Spanish
fixes = {
    5: [  # Cell [5]: Circular Clipping instructions
        "---\n",
        "## 2. Circular Clipping (Brick 2)\n",
        "\n",
        "**Instructions:**\n",
        "1. Open the original file in CloudCompare\n",
        "2. Use the \"Point Picking\" tool to locate the mat centre\n",
        "3. Copy the Xg, Yg coordinates shown\n",
        "4. Paste the values into `CENTER_X` and `CENTER_Y` below\n",
    ],
    9: [  # Cell [9]: Noise Filtering
        "---\n",
        "## 3. Noise Filtering (Brick 3)\n",
    ],
    11: [  # Cell [11]: Normalisation Detection
        "---\n",
        "## 4. Normalisation Analysis (Brick 4)\n",
    ],
    13: [  # Cell [13]: Ground Filtering
        "---\n",
        "## 5. Ground Filtering (Brick 5)\n",
    ],
    16: [  # Cell [16]: Height Normalisation
        "---\n",
        "## 6. Height Normalisation (Brick 6)\n",
    ],
    18: [  # Cell [18]: Export Checkpoint
        "---\n",
        "## 6.5 Export Checkpoint\n",
    ],
    20: [  # Cell [20]: 3D Visualisation
        "---\n",
        "## 6.5 3D Visualisation\n",
    ],
}

for idx, new_source in fixes.items():
    if idx < len(cells) and cells[idx]["cell_type"] == "markdown":
        cells[idx]["source"] = new_source
        print(f"  Fixed cell [{idx}]: {new_source[1].strip()}")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
