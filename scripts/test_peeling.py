"""
Test script for iterative peeling understory separation.

Validates the algorithm on synthetic and real data.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    compute_all_features_fast,
    iterative_peeling_understory,
)


def test_synthetic_trunks():
    """Test on synthetic vertical trunks + scattered understory."""
    print("=" * 70)
    print("TEST 1: Synthetic Trunks + Understory")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create 5 vertical trunks (cylinders)
    trunks = []
    trunk_centers = [(0, 0), (5, 0), (0, 5), (5, 5), (2.5, 2.5)]
    
    for cx, cy in trunk_centers:
        # Vertical cylinder: radius ~0.15m, height 0-8m
        n_pts = 2000
        angles = np.random.uniform(0, 2*np.pi, n_pts)
        radii = np.random.normal(0.15, 0.03, n_pts)
        x = cx + radii * np.cos(angles)
        y = cy + radii * np.sin(angles)
        z = np.random.uniform(0, 8, n_pts)
        trunks.append(np.column_stack([x, y, z]))
    
    trunks_xyz = np.vstack(trunks)
    
    # Create scattered understory (random points, low height)
    n_understory = 5000
    understory_xyz = np.column_stack([
        np.random.uniform(-2, 7, n_understory),
        np.random.uniform(-2, 7, n_understory),
        np.random.uniform(0, 2, n_understory)
    ])
    
    # Combine
    xyz = np.vstack([trunks_xyz, understory_xyz])
    n_trunks = len(trunks_xyz)
    n_understory_true = len(understory_xyz)
    
    print(f"\nSynthetic data: {len(xyz):,} points")
    print(f"  True trunks: {n_trunks:,}")
    print(f"  True understory: {n_understory_true:,}")
    
    # Compute features
    print("\nComputing features...")
    features, dist_to_ground, dist_to_top = compute_all_features_fast(xyz, verbose=False)
    
    # Run iterative peeling
    print("\n" + "-" * 70)
    print("Running iterative peeling...")
    print("-" * 70)
    result = iterative_peeling_understory(
        xyz,
        features.verticality,
        features.linearity,
        features.sphericity,
        dist_to_ground,
        seed_verticality=0.85,
        seed_linearity=0.5,
        seed_height_min=1.0,
        seed_height_max=3.0,
        expansion_verticality=0.4,
        expansion_radius=0.3,
        verbose=True
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION:")
    print("=" * 70)
    
    # Ground truth labels
    true_trunk_mask = np.zeros(len(xyz), dtype=bool)
    true_trunk_mask[:n_trunks] = True
    
    # Compute metrics
    true_positives = np.sum(result.is_tree & true_trunk_mask)
    false_positives = np.sum(result.is_tree & ~true_trunk_mask)
    false_negatives = np.sum(~result.is_tree & true_trunk_mask)
    true_negatives = np.sum(~result.is_tree & ~true_trunk_mask)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.3f} (TP={true_positives}, FP={false_positives})")
    print(f"Recall:    {recall:.3f} (TP={true_positives}, FN={false_negatives})")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Accuracy:  {(true_positives + true_negatives) / len(xyz):.3f}")
    
    # Success criteria
    success = precision > 0.95 and recall > 0.90
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Precision > 0.95 and Recall > 0.90")
    
    return success


def test_parameter_sensitivity():
    """Test sensitivity to seed parameters."""
    print("\n\n" + "=" * 70)
    print("TEST 2: Parameter Sensitivity")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Simple synthetic data
    n_trunk = 1000
    trunk_xyz = np.column_stack([
        np.random.normal(0, 0.1, n_trunk),
        np.random.normal(0, 0.1, n_trunk),
        np.random.uniform(0, 5, n_trunk)
    ])
    
    n_understory = 500
    understory_xyz = np.column_stack([
        np.random.uniform(-2, 2, n_understory),
        np.random.uniform(-2, 2, n_understory),
        np.random.uniform(0, 1, n_understory)
    ])
    
    xyz = np.vstack([trunk_xyz, understory_xyz])
    
    print(f"\nData: {len(xyz):,} points ({n_trunk} trunk, {n_understory} understory)")
    
    # Compute features
    features, dist_to_ground, dist_to_top = compute_all_features_fast(xyz, verbose=False)
    
    # Test different seed_verticality values
    test_values = [0.85, 0.90, 0.95]
    
    print("\nTesting seed_verticality values:")
    for seed_vert in test_values:
        result = iterative_peeling_understory(
            xyz,
            features.verticality,
            features.linearity,
            features.sphericity,
            dist_to_ground,
            seed_verticality=seed_vert,
            expansion_verticality=0.5,
            verbose=False
        )
        print(f"  seed_verticality={seed_vert}: {result.n_seeds} seeds → "
              f"{result.n_tree} tree pts ({100*result.n_tree/len(xyz):.1f}%)")
    
    print("\n✓ Parameter sensitivity test complete")
    return True


if __name__ == "__main__":
    success1 = test_synthetic_trunks()
    success2 = test_parameter_sensitivity()
    
    print("\n" + "=" * 70)
    if success1 and success2:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
    print("=" * 70)
