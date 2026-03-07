"""Quick test of trunk scrubbing with synthetic data."""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from src.core.trunk_validation import scrub_trunks, TrunkScrubConfig
from src.core.trunk_extraction import TrunkExtractionResult, TrunkExtractionConfig

np.random.seed(42)
N = 5000

# === Tree 0: Clean cylinder (real trunk) ===
# Radius ~0.2m, height 5-25m
z0 = np.random.uniform(5, 25, N)
theta0 = np.random.uniform(0, 2*np.pi, N)
r0 = 0.2 + np.random.normal(0, 0.01, N)
good_trunk = np.column_stack([
    10 + r0 * np.cos(theta0),
    10 + r0 * np.sin(theta0),
    z0,
])

# === Tree 1: Trunk + attached understory (mixed) ===
# Real trunk core (~0.15m radius) + understory blob attached to one side
n_trunk = 3000
n_under = 2000
z1a = np.random.uniform(3, 20, n_trunk)
theta1 = np.random.uniform(0, 2*np.pi, n_trunk)
r1 = 0.15 + np.random.normal(0, 0.01, n_trunk)
trunk_core = np.column_stack([
    20 + r1 * np.cos(theta1),
    20 + r1 * np.sin(theta1),
    z1a,
])

# Understory blob offset from trunk center
z1b = np.random.uniform(3, 8, n_under)
understory = np.column_stack([
    20 + 0.3 + np.random.uniform(-0.5, 0.5, n_under),  # offset 0.3m + wide
    20 + np.random.uniform(-0.5, 0.5, n_under),
    z1b,
])
mixed_tree = np.vstack([trunk_core, understory])

# === Tree 2: Pure understory mass (no trunk) ===
z2 = np.random.uniform(2, 6, N)
understory_only = np.column_stack([
    30 + np.random.uniform(-1.0, 1.0, N),
    30 + np.random.uniform(-0.8, 0.8, N),
    z2,
])

# Combine all
xyz = np.vstack([good_trunk, mixed_tree, understory_only])
n0, n1, n2 = len(good_trunk), len(mixed_tree), len(understory_only)
trunk_mask = np.ones(len(xyz), dtype=bool)
tree_ids = np.concatenate([
    np.zeros(n0, dtype=int),
    np.ones(n1, dtype=int),
    np.full(n2, 2, dtype=int),
])

trunk_result = TrunkExtractionResult(
    trunk_mask=trunk_mask,
    tree_ids=tree_ids,
    n_trees=3,
    tree_axes=[
        {"tree_id": 0, "centroid": good_trunk.mean(axis=0),
         "direction": np.array([0,0,1.0]), "n_points": n0,
         "z_min": 5.0, "z_max": 25.0},
        {"tree_id": 1, "centroid": trunk_core.mean(axis=0),
         "direction": np.array([0,0,1.0]), "n_points": n1,
         "z_min": 3.0, "z_max": 20.0},
        {"tree_id": 2, "centroid": understory_only.mean(axis=0),
         "direction": np.array([0,0,1.0]), "n_points": n2,
         "z_min": 2.0, "z_max": 6.0},
    ],
    cluster_points=xyz[:10],
    config=TrunkExtractionConfig(),
)

config = TrunkScrubConfig(
    section_height=1.0,
    radius_offset=0.03,
    min_points_per_section=30,
    min_trunk_points_after=200,
    min_trunk_height=5.0,
    dbh_max=0.80,
    percentile=75.0,
)

result = scrub_trunks(xyz, trunk_result, config)

print(f"\n=== Test Results ===")
for tr in result.tree_results:
    print(f"  Tree {tr.tree_id}: {tr.points_before} → {tr.points_after} pts "
          f"(removed={tr.points_removed}, entirely={tr.removed_entirely})")

# Tree 0 (clean trunk): should keep most points
t0 = result.tree_results[0]
assert not t0.removed_entirely, "Clean trunk should NOT be removed"
assert t0.points_after > 0.8 * t0.points_before, \
    f"Clean trunk lost too many points: {t0.points_after}/{t0.points_before}"

# Tree 1 (trunk + understory): should keep trunk core, scrub understory
t1 = result.tree_results[1]
assert not t1.removed_entirely, "Mixed tree should NOT be removed entirely"
assert t1.points_removed > 500, \
    f"Mixed tree should have scrubbed understory, only removed {t1.points_removed}"

# Tree 2 (pure understory): should be removed or heavily scrubbed
t2 = result.tree_results[2]
# Either removed entirely OR heavily scrubbed
if not t2.removed_entirely:
    assert t2.points_after < 0.3 * t2.points_before, \
        f"Pure understory should be heavily scrubbed: {t2.points_after}/{t2.points_before}"

print("\n✓ ALL TESTS PASSED")
