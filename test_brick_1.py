"""
Brick 1 Verification Script

Tests the PointCloudLoader to ensure LAS/LAZ files can be read correctly.
Run: python test_brick_1.py [path_to_file.laz]
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from src.core.io import PointCloudLoader


def main():
    # Default test file (found via search in external_references)
    default_file = Path("external_references/artemis_treeiso/data/LPine1_demo.laz")
    
    # Allow command line override
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    print(f"{'='*60}")
    print(f"BRICK 1 VERIFICATION: Point Cloud I/O")
    print(f"{'='*60}")
    print(f"Target file: {file_path}")

    if not file_path.exists():
        print(f"\n❌ ERROR: File not found at {file_path}")
        print("Usage: python test_brick_1.py <path_to_las_or_laz>")
        return 1

    try:
        # Test 1: Load file
        print(f"\n[1] Loading file...")
        loader = PointCloudLoader(file_path)
        loader.load()
        print(f"    ✓ File loaded successfully")
        print(f"    ✓ Loader repr: {loader}")
        
        # Test 2: Get metadata
        print(f"\n[2] Extracting metadata...")
        meta = loader.get_metadata()
        for key, value in meta.items():
            print(f"    {key}: {value}")
        
        # Test 3: Get available dimensions
        print(f"\n[3] Available dimensions...")
        dims = loader.get_available_dimensions()
        print(f"    {dims}")
        
        # Test 4: Get XYZ coordinates
        print(f"\n[4] Getting XYZ coordinates...")
        xyz = loader.get_xyz()
        print(f"    Shape: {xyz.shape}")
        print(f"    Dtype: {xyz.dtype}")
        print(f"    Memory: {xyz.nbytes / (1024*1024):.2f} MB")
        
        # Test 5: Get an attribute (if intensity exists)
        print(f"\n[5] Testing attribute access...")
        if "intensity" in dims:
            intensity = loader.get_attribute("intensity")
            print(f"    Intensity range: {intensity.min()} - {intensity.max()}")
        else:
            print(f"    (intensity not available, skipping)")
        
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
