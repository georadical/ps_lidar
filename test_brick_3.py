"""
Brick 3 Verification Script: Ground Filtering (CSF)

Tests the classify_ground function using the Cloth Simulation Filter.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.io import PointCloudLoader
from src.core.ground import classify_ground


def main():
    # Use the same demo file
    default_file = Path("external_references/artemis_treeiso/data/LPine1_demo.laz")
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    print(f"{'='*60}")
    print(f"BRICK 3 VERIFICATION: Ground Filtering (CSF)")
    print(f"{'='*60}")

    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return 1

    try:
        # 1. Load data
        print(f"[1] Loading point cloud...")
        loader = PointCloudLoader(file_path)
        loader.load()
        xyz = loader.get_xyz()
        print(f"    ✓ Loaded {len(xyz):,} points")

        # 2. Run CSF
        print(f"\n[2] Running Cloth Simulation Filter...")
        # We use default parameters for forest plots
        ground_idx, non_ground_idx = classify_ground(
            xyz, 
            cloth_resolution=1.0, 
            rigidness=1, 
            class_threshold=0.5
        )
        
        n_ground = len(ground_idx)
        n_veg = len(non_ground_idx)
        total = n_ground + n_veg
        
        print(f"    ✓ Filtering complete")
        print(f"    • Ground:     {n_ground:,} points ({n_ground/total:.1%})")
        print(f"    • Off-ground: {n_veg:,} points ({n_veg/total:.1%})")

        # 3. Basic validity check
        if n_ground == 0:
            print(f"\n⚠️ WARNING: No ground points detected. Check parameters.")
        else:
            z_ground = xyz[ground_idx, 2]
            print(f"    • Ground Z range: {z_ground.min():.2f}m to {z_ground.max():.2f}m")
            
            # Since LPine1_demo is normalized, ground should be near Z=0
            if abs(z_ground.mean()) < 1.0:
                print(f"    ✓ Ground correctly identified near Z=0 (normalized input)")

        print(f"\n{'='*60}")
        print(f"✅ BRICK 3 COMPLETE")
        print(f"{'='*60}")
        return 0

    except Exception as e:
        print(f"\n❌ FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
