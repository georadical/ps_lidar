"""
Test script for optimized understory classification.

Demonstrates the new height-dependent thresholds and diameter-based filtering.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    compute_all_features_fast,
    classify_understory,
    estimate_local_radius,
)

def test_height_adaptive_thresholds():
    """Test height-dependent threshold adaptation."""
    print("=" * 70)
    print("TEST 1: Height-Adaptive Thresholds")
    print("=" * 70)
    
    # Create synthetic data: low points (understory) and high points (canopy)
    np.random.seed(42)
    n_low = 1000
    n_high = 1000
    
    # Low points (0-2m) - should have stricter thresholds
    xyz_low = np.column_stack([
        np.random.uniform(-5, 5, n_low),
        np.random.uniform(-5, 5, n_low),
        np.random.uniform(0, 2, n_low)
    ])
    
    # High points (4-8m) - should have relaxed thresholds
    xyz_high = np.column_stack([
        np.random.uniform(-5, 5, n_high),
        np.random.uniform(-5, 5, n_high),
        np.random.uniform(4, 8, n_high)
    ])
    
    xyz = np.vstack([xyz_low, xyz_high])
    
    print(f"\nSynthetic data: {len(xyz):,} points")
    print(f"  Low (0-2m): {n_low:,} points")
    print(f"  High (4-8m): {n_high:,} points")
    
    # Compute features
    print("\nComputing features...")
    features, dist_to_ground, dist_to_top = compute_all_features_fast(xyz, verbose=True)
    
    # Test with adaptive thresholds
    print("\n" + "-" * 70)
    print("Classification WITH height-adaptive thresholds:")
    print("-" * 70)
    result_adaptive = classify_understory(
        xyz,
        features.verticality,
        features.linearity,
        features.sphericity,
        dist_to_ground,
        dist_to_top,
        use_height_adaptive=True,
        verbose=True
    )
    
    # Test without adaptive thresholds
    print("\n" + "-" * 70)
    print("Classification WITHOUT height-adaptive thresholds:")
    print("-" * 70)
    result_fixed = classify_understory(
        xyz,
        features.verticality,
        features.linearity,
        features.sphericity,
        dist_to_ground,
        dist_to_top,
        use_height_adaptive=False,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    print(f"Adaptive:  {result_adaptive.n_understory:,} understory, {result_adaptive.n_tree:,} tree")
    print(f"Fixed:     {result_fixed.n_understory:,} understory, {result_fixed.n_tree:,} tree")
    print(f"Difference: {abs(result_adaptive.n_understory - result_fixed.n_understory):,} points")


def test_diameter_filtering():
    """Test diameter-based filtering."""
    print("\n\n" + "=" * 70)
    print("TEST 2: Diameter-Based Filtering")
    print("=" * 70)
    
    # Create synthetic stems: thin (understory) and thick (trees)
    np.random.seed(42)
    
    # Thin stems (radius ~0.02m)
    n_thin = 500
    thin_centers = np.random.uniform(-5, 5, (n_thin // 10, 2))
    xyz_thin = []
    for center in thin_centers:
        # Create vertical stem with small radius
        n_pts = 50
        angles = np.random.uniform(0, 2*np.pi, n_pts)
        radii = np.random.normal(0.02, 0.005, n_pts)
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        z = np.random.uniform(0, 3, n_pts)
        xyz_thin.append(np.column_stack([x, y, z]))
    xyz_thin = np.vstack(xyz_thin)
    
    # Thick stems (radius ~0.15m)
    n_thick = 500
    thick_centers = np.random.uniform(-5, 5, (n_thick // 10, 2))
    xyz_thick = []
    for center in thick_centers:
        # Create vertical stem with large radius
        n_pts = 50
        angles = np.random.uniform(0, 2*np.pi, n_pts)
        radii = np.random.normal(0.15, 0.03, n_pts)
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        z = np.random.uniform(0, 3, n_pts)
        xyz_thick.append(np.column_stack([x, y, z]))
    xyz_thick = np.vstack(xyz_thick)
    
    xyz = np.vstack([xyz_thin, xyz_thick])
    
    print(f"\nSynthetic stems: {len(xyz):,} points")
    print(f"  Thin stems (~0.02m): {len(xyz_thin):,} points")
    print(f"  Thick stems (~0.15m): {len(xyz_thick):,} points")
    
    # Compute features
    print("\nComputing features...")
    features, dist_to_ground, dist_to_top = compute_all_features_fast(xyz, verbose=False)
    
    # Estimate local radius
    print("\nEstimating local radius...")
    local_radius = estimate_local_radius(xyz, verbose=True)
    
    # Test with diameter filtering
    print("\n" + "-" * 70)
    print("Classification WITH diameter filtering:")
    print("-" * 70)
    result_diameter = classify_understory(
        xyz,
        features.verticality,
        features.linearity,
        features.sphericity,
        dist_to_ground,
        dist_to_top,
        local_radius=local_radius,
        min_stem_radius=0.05,  # Threshold between thin and thick
        use_height_adaptive=False,
        verbose=True
    )
    
    # Test without diameter filtering
    print("\n" + "-" * 70)
    print("Classification WITHOUT diameter filtering:")
    print("-" * 70)
    result_no_diameter = classify_understory(
        xyz,
        features.verticality,
        features.linearity,
        features.sphericity,
        dist_to_ground,
        dist_to_top,
        local_radius=None,
        use_height_adaptive=False,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    print(f"With diameter:    {result_diameter.n_understory:,} understory, {result_diameter.n_tree:,} tree")
    print(f"Without diameter: {result_no_diameter.n_understory:,} understory, {result_no_diameter.n_tree:,} tree")
    print(f"Thin stems filtered: {result_no_diameter.n_tree - result_diameter.n_tree:,} points")


if __name__ == "__main__":
    test_height_adaptive_thresholds()
    test_diameter_filtering()
    print("\n" + "=" * 70)
    print("✓ All tests completed")
    print("=" * 70)
