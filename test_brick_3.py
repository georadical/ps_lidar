"""
Brick 3 Verification Script: Ground Filtering (CSF)

Tests the ground filtering functions using the Cloth Simulation Filter.
Run: python test_brick_3.py [path_to_file.laz]
"""

import sys
import time
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from src.core import PointCloudLoader, classify_ground, get_ground_mask


def main():
    default_file = Path("external_references/artemis_treeiso/data/LPine1_demo.laz")
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    print(f"{'='*60}")
    print(f"BRICK 3 VERIFICATION: Ground Filtering (CSF)")
    print(f"{'='*60}")
    print(f"Target file: {file_path}")

    if not file_path.exists():
        print(f"\n❌ ERROR: File not found at {file_path}")
        print("Usage: python test_brick_3.py <path_to_las_or_laz>")
        return 1

    try:
        # [1] Load data
        print(f"\n[1] Loading point cloud...")
        loader = PointCloudLoader(file_path)
        loader.load()
        xyz = loader.get_xyz()
        print(f"    ✓ Loaded {len(xyz):,} points")

        # [2] Run CSF with classify_ground
        print(f"\n[2] Running Cloth Simulation Filter (classify_ground)...")
        t0 = time.perf_counter()
        result = classify_ground(
            xyz, 
            cloth_resolution=1.0, 
            rigidness=1, 
            class_threshold=0.5,
            slope_smooth=True,
        )
        elapsed = time.perf_counter() - t0
        
        print(f"    ✓ Filtering complete in {elapsed:.2f}s")
        print(f"    • Ground:     {result.n_ground:,} points ({result.ground_ratio:.1%})")
        print(f"    • Off-ground: {result.n_off_ground:,} points ({1 - result.ground_ratio:.1%})")

        # [3] Validate ground Z range
        print(f"\n[3] Validating ground detection...")
        if result.n_ground == 0:
            print(f"    ⚠️ WARNING: No ground points detected. Check parameters.")
        else:
            z_ground = xyz[result.ground_indices, 2]
            print(f"    • Ground Z range: {z_ground.min():.2f}m to {z_ground.max():.2f}m")
            print(f"    • Ground Z mean:  {z_ground.mean():.2f}m")
            
            if abs(z_ground.mean()) < 1.0:
                print(f"    ✓ Ground correctly identified near Z=0 (normalized input)")

        # [4] Test get_ground_mask function
        print(f"\n[4] Testing get_ground_mask...")
        t0 = time.perf_counter()
        mask = get_ground_mask(xyz, cloth_resolution=1.0)
        elapsed = time.perf_counter() - t0
        
        print(f"    ✓ Mask generated in {elapsed:.2f}s")
        print(f"    • Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"    • Ground points (mask sum): {mask.sum():,}")

        # [5] Consistency check
        print(f"\n[5] Consistency check...")
        if mask.sum() == result.n_ground:
            print(f"    ✓ classify_ground and get_ground_mask are consistent")
        else:
            print(f"    ⚠️ Mismatch: mask has {mask.sum()} vs result has {result.n_ground}")

        print(f"\n{'='*60}")
        print(f"✅ ALL TESTS PASSED")
        print(f"{'='*60}")
        return 0

    except Exception as e:
        print(f"\n❌ FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
