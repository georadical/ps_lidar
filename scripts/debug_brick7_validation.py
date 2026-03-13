"""
Debug script to analyze why specific trunks are rejected in Brick 7.
It runs the extract_trunks function with verbose=True to print
the exact rejection reason for each cluster to the console.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import laspy
from src.core.trunk_extraction import extract_trunks, TrunkExtractionConfig

# 1. Load the point cloud from Brick 6 (veg_normalized)
VEG_CHECKPOINT = Path("D:/OUTPUTS/T460298A_VEG_NORM.laz")
print(f"Loading: {VEG_CHECKPOINT.name}...")
las = laspy.read(str(VEG_CHECKPOINT))
xyz = np.column_stack([
    np.array(las.x, dtype=np.float32),
    np.array(las.y, dtype=np.float32),
    np.array(las.z, dtype=np.float32),
])

# 2. Configure the extraction parameters (same as Brick 7)
DBH_MIN = 0.20
DBH_MAX = 0.80
HEIGHT_MAX = 36.0
CROWN_DISTANCE_A = 8.0
CROWN_DISTANCE_B = 7.0
CROWN_DISTANCE_C = 6.5
CROWN_DISTANCE_D = 7.5
MAX_AXIS_DISTANCE = max(CROWN_DISTANCE_A, CROWN_DISTANCE_B, CROWN_DISTANCE_C, CROWN_DISTANCE_D)
STEM_SEARCH_RADIUS = (DBH_MAX / 2) + 0.10

config = TrunkExtractionConfig(
    stripe_lower_limit=2.0,
    stripe_upper_limit=6.0,
    dbh_min=DBH_MIN,
    dbh_max=DBH_MAX,
    height_max=HEIGHT_MAX,
    max_axis_distance=MAX_AXIS_DISTANCE,
    stem_search_radius=STEM_SEARCH_RADIUS,
    
    # Validation params
    cluster_circularity_min=0.3,
    cluster_diameter_max_factor=1.5,
    cluster_min_height=2.0,
    cluster_min_diameter=0.05,
    
    voxel_resolution_xy=0.05,
    voxel_resolution_z=0.05,
    verticality_threshold=0.7,
    peeling_iterations=2,
    min_cluster_points=500,
)

print("\nRunning Trunk Extraction with VERBOSE=True...")
print("Look for the 'REJECTED' lines to see why trunks are disappearing.")
print("-" * 60)

# 3. Run extraction and print logs
result = extract_trunks(xyz, config=config, verbose=True)

print("-" * 60)
print(f"Extraction complete: {result.n_trees} valid trees found.")
print("\nIf valid trees are being rejected, you can tweak these parameters in the notebook's BRICK 7:")
print("- Decrease 'cluster_circularity_min' (e.g., from 0.3 to 0.15) if trees have buttress roots or vines.")
print("- Increase 'cluster_diameter_max_factor' (e.g., from 1.5 to 2.0) if tree bases are very wide.")
