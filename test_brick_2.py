"""
Brick 2 Verification Script

Tests the NormalizationAnalyzer to detect if point clouds are height-normalized.
Run: python test_brick_2.py [path_to_file.laz]
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from src.core.io import PointCloudLoader
from src.core.normalization import (
    NormalizationAnalyzer,
    NormalizationStatus,
    detect_normalization,
)


def main():
    # Default test file
    default_file = Path("external_references/artemis_treeiso/data/LPine1_demo.laz")
    
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    print(f"{'='*60}")
    print(f"BRICK 2 VERIFICATION: Normalization Detection")
    print(f"{'='*60}")
    print(f"Target file: {file_path}")

    if not file_path.exists():
        print(f"\n❌ ERROR: File not found at {file_path}")
        print("Usage: python test_brick_2.py <path_to_las_or_laz>")
        return 1

    try:
        # Step 1: Load the point cloud
        print(f"\n[1] Loading point cloud...")
        loader = PointCloudLoader(file_path)
        loader.load()
        xyz = loader.get_xyz()
        print(f"    ✓ Loaded {len(xyz):,} points")
        
        # Step 2: Analyze normalization
        print(f"\n[2] Analyzing normalization status...")
        result = detect_normalization(xyz)
        
        # Step 3: Display results
        print(f"\n[3] Analysis Results:")
        print(f"    Status:     {result.status.value.upper()}")
        print(f"    Confidence: {result.confidence:.1%}")
        print(f"    Z Range:    {result.z_min:.2f}m to {result.z_max:.2f}m ({result.z_range:.2f}m)")
        print(f"    Z Mean:     {result.z_mean:.2f}m (σ = {result.z_std:.2f}m)")
        print(f"    5th Pctl:   {result.percentile_5:.2f}m")
        print(f"    Ground:     {'Detected' if result.ground_plane_detected else 'Not detected'}")
        print(f"\n[4] Reasoning:")
        for reason in result.reasoning.split("; "):
            print(f"    • {reason}")
        
        # Step 4: Interpretation
        print(f"\n[5] Interpretation:")
        if result.status == NormalizationStatus.NORMALIZED:
            print(f"    → Point cloud appears to be HEIGHT-NORMALIZED.")
            print(f"    → Z values represent height above ground.")
            print(f"    → Ready for direct tree segmentation.")
        elif result.status == NormalizationStatus.NOT_NORMALIZED:
            print(f"    → Point cloud appears to use ABSOLUTE ELEVATION.")
            print(f"    → Z values need normalization before processing.")
            print(f"    → Will need CSF ground filtering + DTM interpolation.")
        else:
            print(f"    → UNCERTAIN - manual verification recommended.")
            print(f"    → Consider checking data source documentation.")
        
        print(f"\n{'='*60}")
        print(f"✅ NORMALIZATION DETECTION COMPLETE")
        print(f"{'='*60}")
        return 0
        
    except Exception as e:
        print(f"\n❌ FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
