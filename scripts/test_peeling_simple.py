"""Quick test of iterative peeling."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import compute_all_features_fast, iterative_peeling_understory

# Simple test: 1 trunk + scattered understory
np.random.seed(42)

# Trunk (vertical cylinder)
n_trunk = 500
trunk_xyz = np.column_stack([
    np.random.normal(0, 0.1, n_trunk),
    np.random.normal(0, 0.1, n_trunk),
    np.random.uniform(0, 5, n_trunk)
])

# Understory (scattered low points)
n_understory = 200
understory_xyz = np.column_stack([
    np.random.uniform(-2, 2, n_understory),
    np.random.uniform(-2, 2, n_understory),
    np.random.uniform(0, 1, n_understory)
])

xyz = np.vstack([trunk_xyz, understory_xyz])

print(f"Data: {len(xyz)} points ({n_trunk} trunk, {n_understory} understory)")

# Compute features
print("\nComputing features...")
features, dist_to_ground, dist_to_top = compute_all_features_fast(xyz, verbose=False)

print(f"Verticality range: {features.verticality.min():.2f} - {features.verticality.max():.2f}")
print(f"Linearity range: {features.linearity.min():.2f} - {features.linearity.max():.2f}")

# Run peeling
print("\nRunning iterative peeling...")
result = iterative_peeling_understory(
    xyz,
    features.verticality,
    features.linearity,
    features.sphericity,
    dist_to_ground,
    seed_verticality=0.8,
    seed_linearity=0.5,
    expansion_verticality=0.4,
    verbose=True
)

# Evaluate
true_trunk_mask = np.zeros(len(xyz), dtype=bool)
true_trunk_mask[:n_trunk] = True

tp = np.sum(result.is_tree & true_trunk_mask)
fp = np.sum(result.is_tree & ~true_trunk_mask)
fn = np.sum(~result.is_tree & true_trunk_mask)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nResults:")
print(f"  Precision: {precision:.3f} (TP={tp}, FP={fp})")
print(f"  Recall: {recall:.3f} (TP={tp}, FN={fn})")
print(f"  {'✓ PASS' if precision > 0.9 and recall > 0.9 else '✗ FAIL'}")
