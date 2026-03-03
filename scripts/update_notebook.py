"""
Complete notebook rebuild.
Keeps cells 0..N (up to first old 7.x or Phase 5 cell), 
removes everything after, and appends clean Bricks 7-9.
Also fixes ALL text (Spanish, doubled brick labels, etc).
"""
import json, re
from pathlib import Path

nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "01_playground.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Original cell count: {len(cells)}")

# ============================================================
# PHASE 1: Find the cut point — remove everything from old 
# "7. Geometric features" / "Brick 7.2" / "Phase 5" onwards
# ============================================================
cut_index = len(cells)  # default: keep all

cut_markers = [
    "7. Geometrics features",
    "Geometric features",
    "Brick 7.2",
    "Brick 7.1",
    "BRICK 7:",
    "Phase 5",
    "Workflow Unificado",
    "Segmentation de",
    "Upload files from",
    "Stem Seeds",
    "Opción B",
]

for i, cell in enumerate(cells):
    content = "".join(cell["source"])
    for marker in cut_markers:
        if marker in content:
            if i < cut_index:
                cut_index = i
                print(f"  Cut point found at cell [{i}]: {marker}")
            break

print(f"Cutting at cell index {cut_index}")
cells = cells[:cut_index]

# ============================================================
# PHASE 1b: Remove noise filtering cells
# ============================================================
clean_cells = []
skip_next = False
for i, cell in enumerate(cells):
    if skip_next and cell["cell_type"] == "code":
        print(f"  Removing SOR code cell [{i}]")
        skip_next = False
        continue
    
    content = "".join(cell["source"])
    if cell["cell_type"] == "markdown" and (
        "Filtering de Ruido" in content or
        "Noise Filtering" in content or
        "2.5 Filtering" in content or
        "BRICK 3.5" in content
    ):
        print(f"  Removing noise filter markdown cell [{i}]")
        skip_next = True
        continue
    
    if cell["cell_type"] == "code" and "filter_noise_sor" in content and "import" not in content:
        print(f"  Removing SOR code cell [{i}]")
        continue
    
    clean_cells.append(cell)

cells = clean_cells
print(f"After cleanup: {len(cells)} cells")

# ============================================================
# PHASE 2: Fix ALL text in remaining cells
# ============================================================
replacements = [
    # Fix doubled brick labels
    ("(Brick 2) (Brick 2) (Brick 2)", "(Brick 2)"),
    ("(Brick 2) (Brick 2)", "(Brick 2)"),
    ("(Brick 3) (Brick 3)", "(Brick 3)"),
    ("(Brick 1) (Brick 1)", "(Brick 1)"),
    
    # Spanish markdown titles
    ("## 1. Carga de Archivo (Brick 1)", "## 1. Load File (Brick 1)"),
    ("## 1. Carga de Archivo", "## 1. Load File (Brick 1)"),
    ("## 2. Recorte Circular (Brick 2)", "## 2. Circular Clipping (Brick 2)"),
    ("## 2. Recorte Circular", "## 2. Circular Clipping (Brick 2)"),
    ("## 3. Análisis de Normalización (Brick 3)", "## 3. Normalisation Analysis (Brick 3)"),
    ("## 3. Análisis de Normalización", "## 3. Normalisation Analysis (Brick 3)"),
    ("## 4. Normalisation Analysis (Brick 3)", "## 3. Normalisation Analysis (Brick 3)"),
    ("## 5. Ground Filtering (Brick 4)", "## 4. Ground Filtering (Brick 4)"),
    ("## 5. Ground Filtering (Brick 5)", "## 4. Ground Filtering (Brick 4)"),
    ("## 5. Normalisation de Height (Brick 5)", "## 5. Height Normalisation (Brick 5)"),
    ("Normalisation de Height", "Height Normalisation"),
    ("## 5. Height Normalisation (Brick 5)", "## 5. Height Normalisation (Brick 5)"),
    ("## 6.5 Export Checkpoints", "## 5.5 Export Checkpoint"),
    ("## 6.5 Export Checkpoint", "## 5.5 Export Checkpoint"),
    ("## 6. Visualisation 3D (Brick 6)", "## 6. 3D Visualisation (Brick 6)"),
    ("Visualisation 3D", "3D Visualisation"),
    
    # Spanish markdown content
    ("Instrucciones:", "Instructions:"),
    ("**Instructions:**", "**Instructions:**"),
    ("1. Abrir el archivo original en CloudCompare", "1. Open the original file in CloudCompare"),
    ("2. Usar herramienta \"Point Picking\" para ubicar el centro del mat", "2. Use the \"Point Picking\" tool to locate the mat centre"),
    ("3. Copiar las coordenadas Xg, Yg mostradas", "3. Copy the Xg, Yg coordinates shown"),
    ("4. Pegar los valores en", "4. Paste the values into"),
    
    # Spanish code comments
    ("# PARÁMETROS DEL USUARIO", "# USER PARAMETERS"),
    ("PARÁMETROS DEL USUARIO", "USER PARAMETERS"),
    ("# Load XYZ y campos escalares disponibles", "# Load XYZ and available scalar fields"),
    ("# Ejecutar recorte circular", "# Run circular clipping"),
    ("# Aplicar recorte a XYZ y campos escalares", "# Apply clipping to XYZ and scalar fields"),
    ("# Directorio de salida", "# Output directory"),
    ("# Reemplazar xyz con points limpios", "# Replace xyz with clean points"),
    ("# Liberar memoria", "# Free memory"),
    ("# Resolver project root", "# Resolve project root"),
    ("# Agregar módulo al path", "# Add module to path"),
    
    # Spanish print statements
    ('print("Ejecutando CSF...")', 'print("Running CSF...")'),
    ("Ejecutando CSF", "Running CSF"),
    ("Recorte aplicado", "Clipping applied"),
    (f"Recorte en", "Clipped in"),
    ("Points originales", "Original points"),
    ("Points en plot", "Points in plot"),
    ("Campos escalares", "Scalar fields"),
    ("campos escalares", "scalar fields"),
    ("Fields escalares", "Scalar fields"),
    ("escalares disponibles", "available scalar fields"),
    ("escalares", "scalar"),
    ("no disponible", "not available"),
    ("valores", "values"),
    ("Memoria XYZ", "XYZ memory"),
    ("Memoria liberada", "Memory freed"),
    (" Memoria ", " Memory "),
    ("memoria", "memory"),
    ("Tamaño", "Size"),
    
    # Remove filter_noise_sor import
    (", filter_noise_sor", ""),
    ("filter_noise_sor, ", ""),
    
    # General Spanish words
    ("Visualización", "Visualisation"),
    ("visualización", "visualisation"),
    ("Normalización", "Normalisation"),
    ("normalización", "normalisation"),
]

n_fixes = 0
for cell in cells:
    new_source = []
    for line in cell["source"]:
        orig = line
        for old, new in replacements:
            line = line.replace(old, new)
        if line != orig:
            n_fixes += 1
        new_source.append(line)
    cell["source"] = new_source

print(f"Applied {n_fixes} text fixes")

# ============================================================
# PHASE 3: Set the header/TOC
# ============================================================
header_source = [
    "# PS LiDAR - Development Playground\n",
    "\n",
    "**Available Bricks:**\n",
    "- Brick 1: Data Loading\n",
    "- Brick 2: Circular Clipping (manual coordinates)\n",
    "- Brick 3: Normalisation Analysis\n",
    "- Brick 4: Ground Filtering\n",
    "- Brick 5: Height Normalisation + Export Checkpoint\n",
    "- Brick 6: 3D Visualisation\n",
    "- Brick 7: **Trunk Extraction** (verticality + DBSCAN + PCA)\n",
    "- Brick 8: **Branch Extraction** (linearity + connectivity)\n",
    "- Brick 9: **Export & Visualisation**\n",
]

for cell in cells:
    if cell["cell_type"] == "markdown":
        cell["source"] = header_source
        break

# ============================================================
# PHASE 4: Append new Bricks 7, 8, 9
# ============================================================
new_cells = [
    # --- BRICK 7 ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "---\n",
            "# BRICK 7: Trunk Extraction\n",
            "\n",
            "**Pipeline** (dendromatics/3DFin):\n",
            "1. Extract horizontal stripe at breast height\n",
            "2. Voxelise and compute verticality (pgeof C++)\n",
            "3. DBSCAN clustering in 2D (XY)\n",
            "4. Iterative peeling\n",
            "5. PCA for tree axes\n",
            "6. Assign all points to the nearest axis\n",
            "\n",
            "**Field parameters** (measured before/after scanning):\n",
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================\n",
            "# BRICK 7: TRUNK EXTRACTION\n",
            "# ==============================================================\n",
            "\n",
            "from src.core.trunk_extraction import extract_trunks, TrunkExtractionConfig\n",
            "\n",
            "trunk_config = TrunkExtractionConfig(\n",
            "    # Field-measured parameters\n",
            "    dbh_min=0.10,        # metres\n",
            "    dbh_max=0.80,        # metres\n",
            "    tree_height_min=5.0, # metres\n",
            "    tree_height_max=35.0,# metres\n",
            "    # Algorithm parameters\n",
            "    stripe_lower=1.0,    # metres (breast height band lower)\n",
            "    stripe_upper=2.0,    # metres (breast height band upper)\n",
            "    voxel_size=0.05,     # metres\n",
            "    verticality_threshold=0.7,\n",
            ")\n",
            "\n",
            "print(f\"Trunk extraction config: {trunk_config}\")\n",
            "print(f\"Input points: {len(veg_normalized):,}\")\n",
            "\n",
            "trunk_result = extract_trunks(veg_normalized, trunk_config, verbose=True)\n",
            "\n",
            "print(f\"\\nTrunks found: {trunk_result.n_trunks}\")\n",
            "print(f\"Trunk points: {len(trunk_result.trunk_points):,}\")\n",
        ],
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================\n",
            "# BRICK 7 (cont): ASSIGN ALL POINTS TO TRUNKS\n",
            "# ==============================================================\n",
            "\n",
            "# Points not assigned to any trunk\n",
            "non_trunk_mask = trunk_result.labels == -1\n",
            "non_trunk_points = veg_normalized[non_trunk_mask]\n",
            "print(f\"Non-trunk points: {len(non_trunk_points):,}\")\n",
        ],
        "outputs": [],
        "execution_count": None,
    },
    # --- BRICK 8 ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# BRICK 8: Branch Extraction\n",
            "\n",
            "**Pipeline:**\n",
            "1. Compute linearity (pgeof) on non-trunk points\n",
            "2. Filter by linearity threshold\n",
            "3. 26-neighbour connectivity graph\n",
            "4. Keep only components connected to trunks\n",
            "5. Filter by maximum branch length\n",
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================\n",
            "# BRICK 8: BRANCH EXTRACTION\n",
            "# ==============================================================\n",
            "\n",
            "from src.core.branch_extraction import extract_branches, BranchExtractionConfig\n",
            "\n",
            "branch_config = BranchExtractionConfig(\n",
            "    linearity_threshold=0.5,\n",
            "    max_branch_length=8.0,  # metres\n",
            "    voxel_size=0.05,         # metres\n",
            "    k_neighbors=15,\n",
            ")\n",
            "\n",
            "print(f\"Branch extraction config: {branch_config}\")\n",
            "print(f\"Input non-trunk points: {len(non_trunk_points):,}\")\n",
            "\n",
            "branch_result = extract_branches(\n",
            "    non_trunk_points,\n",
            "    trunk_result,\n",
            "    branch_config,\n",
            "    verbose=True\n",
            ")\n",
            "\n",
            "print(f\"\\nBranch points: {len(branch_result.branch_points):,}\")\n",
        ],
        "outputs": [],
        "execution_count": None,
    },
    # --- BRICK 9 ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# BRICK 9: Export & Visualisation\n",
            "\n",
            "Exports separated clouds for validation in CloudCompare.\n",
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================\n",
            "# BRICK 9: EXPORT & VISUALISATION\n",
            "# ==============================================================\n",
            "\n",
            "from src.core.io import export_point_cloud\n",
            "from pathlib import Path\n",
            "\n",
            "output_dir = Path(\"output\")\n",
            "output_dir.mkdir(exist_ok=True)\n",
            "\n",
            "# Export trunks\n",
            "export_point_cloud(output_dir / \"trunks.laz\", trunk_result.trunk_points)\n",
            "print(f\"Exported trunks: {len(trunk_result.trunk_points):,} points\")\n",
            "\n",
            "# Export branches\n",
            "export_point_cloud(output_dir / \"branches.laz\", branch_result.branch_points)\n",
            "print(f\"Exported branches: {len(branch_result.branch_points):,} points\")\n",
            "\n",
            "# Export combined wood structure\n",
            "import numpy as np\n",
            "wood = np.vstack([trunk_result.trunk_points, branch_result.branch_points])\n",
            "export_point_cloud(output_dir / \"wood_structure.laz\", wood)\n",
            "print(f\"Exported wood structure: {len(wood):,} points\")\n",
        ],
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Visualisation (Open3D)\n",
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================\n",
            "# BRICK 9 (cont): 3D VISUALISATION\n",
            "# ==============================================================\n",
            "\n",
            "import open3d as o3d\n",
            "import numpy as np\n",
            "\n",
            "# Colour trunks brown, branches green\n",
            "trunk_colours = np.tile([0.6, 0.4, 0.2], (len(trunk_result.trunk_points), 1))\n",
            "branch_colours = np.tile([0.2, 0.7, 0.3], (len(branch_result.branch_points), 1))\n",
            "\n",
            "pcd = o3d.geometry.PointCloud()\n",
            "pcd.points = o3d.utility.Vector3dVector(wood)\n",
            "pcd.colors = o3d.utility.Vector3dVector(np.vstack([trunk_colours, branch_colours]))\n",
            "\n",
            "o3d.visualization.draw_geometries([pcd], window_name=\"Wood Structure\")\n",
        ],
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Next Steps\n",
            "\n",
            "- **Brick 10:** Per-tree analysis (DBH, height, sweep)\n",
            "- **Brick 11:** Fork detection\n",
            "- **Brick 12:** HQP classification of branches and spikes\n",
        ]
    },
]

cells.extend(new_cells)

nb["cells"] = cells

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nFinal cell count: {len(cells)}")
print("Done!")
