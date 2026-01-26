# ═══════════════════════════════════════════════════════════════
# TEST: Feature Extraction Pipeline
# ═══════════════════════════════════════════════════════════════
import os
import sys
import time
import laspy
import numpy as np
from pathlib import Path

# Agregar módulo al path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.core import compute_all_features_fast, classify_understory, validate_tree_connectivity

# Cargar checkpoint
CHECKPOINT_FILE = Path("C:/Users/geoal/Documents/Work/LidarProcessing/LiDAR Cline/Data/processed/HQP079_01_veg_normalized.laz")
print(f"Cargando: {CHECKPOINT_FILE.name}")
las = laspy.read(str(CHECKPOINT_FILE))
veg_normalized = np.column_stack([las.x, las.y, las.z])
print(f"✓ Cargados {len(veg_normalized):,} puntos")

# Fase 1: Features (optimizado)
print("\n" + "="*60)
print("FASE 1: Extracción de features (optimizado)")
print("="*60)
t0 = time.perf_counter()
features, dist_to_ground, dist_to_top = compute_all_features_fast(
    veg_normalized,
    voxel_size=0.1,
    k_neighbors=20,
    cylinder_radius=0.5,
    verbose=True
)
print(f"\n✓ Fase 1 completada en {time.perf_counter() - t0:.1f}s")

# Fase 2: Clasificación
print("\n" + "="*60)
print("FASE 2: Clasificación de understory")
print("="*60)
classification = classify_understory(
    veg_normalized,
    features.verticality,
    features.linearity,
    features.sphericity,
    dist_to_ground,
    dist_to_top,
    verticality_threshold=0.7,
    sphericity_threshold=0.5,
    verbose=True
)

# Fase 3: Conectividad
print("\n" + "="*60)
print("FASE 3: Validación de conectividad")
print("="*60)
is_valid_tree = validate_tree_connectivity(
    veg_normalized,
    classification.is_stem,
    horizontal_radius=0.3,
    vertical_radius=0.8,
    verbose=True
)

veg_for_segmentation = veg_normalized[is_valid_tree]
print(f"\n✓ Puntos para segmentación: {len(veg_for_segmentation):,}")

# Exportar checkpoint
print("\n" + "="*60)
print("EXPORTAR CHECKPOINT")
print("="*60)
veg_filtered_file = Path("C:/Users/geoal/Documents/Work/LidarProcessing/LiDAR Cline/Data/processed/HQP079_01_veg_filtered.laz")

header = laspy.LasHeader(version="1.4", point_format=0)
las_out = laspy.LasData(header)
las_out.x = veg_for_segmentation[:, 0]
las_out.y = veg_for_segmentation[:, 1]
las_out.z = veg_for_segmentation[:, 2]

las_out.add_extra_dim(laspy.ExtraBytesParams(name="verticality", type="float32"))
las_out.add_extra_dim(laspy.ExtraBytesParams(name="linearity", type="float32"))
las_out.add_extra_dim(laspy.ExtraBytesParams(name="sphericity", type="float32"))

las_out.verticality = features.verticality[is_valid_tree]
las_out.linearity = features.linearity[is_valid_tree]
las_out.sphericity = features.sphericity[is_valid_tree]

las_out.write(str(veg_filtered_file))
print(f"✓ Checkpoint exportado: {veg_filtered_file.name}")
print(f"  {len(veg_for_segmentation):,} puntos con features")
