"""Unit tests for the piecewise centerline construction (Improvement 1, Phase 1A).

The function under test is ``_build_tree_centerlines`` in
``src.core.trunk_extraction``. It enriches ``tree_axes`` with a ``centerline``
field but does not yet affect point assignment or downstream sectioning, so
tests focus on the computed polyline itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.trunk_extraction import _build_tree_centerlines


def _make_tree_axes(tree_ids):
    """Helper: produce minimal tree_axes records for the given ids."""
    return [
        {
            "tree_id": tid,
            "centroid": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "n_points": 0,
            "z_min": 0.0,
            "z_max": 0.0,
        }
        for tid in tree_ids
    ]


# ---------------------------------------------------------------------------
# Happy-path geometry
# ---------------------------------------------------------------------------

def test_centerline_simple_vertical_line():
    """A perfect vertical column at (5, 7) should yield a vertical centerline."""
    n = 1000
    rng = np.random.default_rng(0)
    z = np.linspace(0.0, 10.0, n)
    x = np.full(n, 5.0) + rng.normal(0.0, 0.005, n)  # tiny noise
    y = np.full(n, 7.0) + rng.normal(0.0, 0.005, n)
    xyz = np.column_stack([x, y, z])
    tree_ids = np.zeros(n, dtype=np.int32)

    axes = _make_tree_axes([0])
    _build_tree_centerlines(xyz, tree_ids, axes, slab_step=1.0, min_points_per_slab=5)

    cl = axes[0]["centerline"]
    assert cl is not None
    assert cl.shape[1] == 3
    # ~10 control points, ordered by Z
    assert 8 <= len(cl) <= 11
    assert np.all(np.diff(cl[:, 2]) > 0), "control points must be sorted by Z"
    # XY of every control point ≈ (5, 7)
    np.testing.assert_allclose(cl[:, 0], 5.0, atol=0.05)
    np.testing.assert_allclose(cl[:, 1], 7.0, atol=0.05)


def test_centerline_curved_stem_follows_curve():
    """A column whose XY position drifts with height should produce a polyline
    whose XY follows the same drift."""
    n = 2000
    rng = np.random.default_rng(1)
    z = np.linspace(0.0, 20.0, n)
    # Stem leans 0.05 m per metre of height in +X
    x_centre = 0.05 * z
    x = x_centre + rng.normal(0.0, 0.01, n)
    y = np.zeros(n) + rng.normal(0.0, 0.01, n)
    xyz = np.column_stack([x, y, z])
    tree_ids = np.zeros(n, dtype=np.int32)

    axes = _make_tree_axes([0])
    _build_tree_centerlines(xyz, tree_ids, axes, slab_step=1.0, min_points_per_slab=5)

    cl = axes[0]["centerline"]
    assert cl is not None
    # X of each control point should roughly equal 0.05 * Z of that point
    expected_x = 0.05 * cl[:, 2]
    np.testing.assert_allclose(cl[:, 0], expected_x, atol=0.05)
    np.testing.assert_allclose(cl[:, 1], 0.0, atol=0.05)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_centerline_robust_to_outliers():
    """The use of median for XY centroids should resist a few wild outliers."""
    n_main = 600
    n_outliers = 30
    rng = np.random.default_rng(2)
    z_main = np.linspace(0.0, 10.0, n_main)
    main = np.column_stack([
        np.full(n_main, 3.0) + rng.normal(0.0, 0.01, n_main),
        np.full(n_main, 4.0) + rng.normal(0.0, 0.01, n_main),
        z_main,
    ])
    # Outliers at far XY positions but within the same Z range
    outliers = np.column_stack([
        rng.uniform(-50.0, 50.0, n_outliers),
        rng.uniform(-50.0, 50.0, n_outliers),
        rng.uniform(0.0, 10.0, n_outliers),
    ])
    xyz = np.vstack([main, outliers])
    tree_ids = np.zeros(len(xyz), dtype=np.int32)

    axes = _make_tree_axes([0])
    _build_tree_centerlines(xyz, tree_ids, axes, slab_step=1.0, min_points_per_slab=10)

    cl = axes[0]["centerline"]
    assert cl is not None
    # Median should hold the centerline near the true (3, 4) despite outliers.
    np.testing.assert_allclose(cl[:, 0], 3.0, atol=0.5)
    np.testing.assert_allclose(cl[:, 1], 4.0, atol=0.5)


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------

def test_centerline_no_points_assigned():
    """A tree with zero assigned points should get centerline=None."""
    xyz = np.zeros((100, 3))
    tree_ids = np.full(100, -1, dtype=np.int32)  # nothing assigned
    axes = _make_tree_axes([0])
    _build_tree_centerlines(xyz, tree_ids, axes, slab_step=0.5, min_points_per_slab=5)
    assert axes[0]["centerline"] is None


def test_centerline_too_short_z_range():
    """A tree shorter than slab_step should get centerline=None."""
    n = 100
    xyz = np.column_stack([
        np.zeros(n),
        np.zeros(n),
        np.linspace(0.0, 0.3, n),  # 0.3 m range, less than slab_step
    ])
    tree_ids = np.zeros(n, dtype=np.int32)
    axes = _make_tree_axes([0])
    _build_tree_centerlines(xyz, tree_ids, axes, slab_step=0.5, min_points_per_slab=5)
    assert axes[0]["centerline"] is None


def test_centerline_min_points_filters_sparse_slabs():
    """Slabs with fewer than min_points_per_slab points should be skipped.
    The remaining control points still form a polyline if there are >= 2."""
    # Dense lower half, sparse upper half (only 2 points per upper slab).
    rng = np.random.default_rng(3)
    z_dense = np.linspace(0.0, 5.0, 600)
    dense = np.column_stack([
        np.zeros(600) + rng.normal(0.0, 0.01, 600),
        np.zeros(600) + rng.normal(0.0, 0.01, 600),
        z_dense,
    ])
    z_sparse = np.linspace(5.5, 10.0, 8)
    sparse = np.column_stack([np.zeros(8), np.zeros(8), z_sparse])
    xyz = np.vstack([dense, sparse])
    tree_ids = np.zeros(len(xyz), dtype=np.int32)

    axes = _make_tree_axes([0])
    _build_tree_centerlines(xyz, tree_ids, axes, slab_step=1.0, min_points_per_slab=20)
    cl = axes[0]["centerline"]
    assert cl is not None
    # All control points should be inside the dense range (Z <= 5.0 + slab tolerance)
    assert cl[:, 2].max() <= 5.5


# ---------------------------------------------------------------------------
# Multi-tree handling
# ---------------------------------------------------------------------------

def test_centerline_multiple_trees_independent():
    """Each tree's centerline is computed from its own assigned points only."""
    rng = np.random.default_rng(4)
    n_per = 500
    # Tree 0 at XY = (0, 0)
    z0 = np.linspace(0.0, 10.0, n_per)
    t0 = np.column_stack([
        rng.normal(0.0, 0.01, n_per),
        rng.normal(0.0, 0.01, n_per),
        z0,
    ])
    # Tree 1 at XY = (20, 20)
    z1 = np.linspace(0.0, 10.0, n_per)
    t1 = np.column_stack([
        20.0 + rng.normal(0.0, 0.01, n_per),
        20.0 + rng.normal(0.0, 0.01, n_per),
        z1,
    ])
    xyz = np.vstack([t0, t1])
    tree_ids = np.concatenate([np.zeros(n_per, dtype=np.int32),
                               np.ones(n_per, dtype=np.int32)])

    axes = _make_tree_axes([0, 1])
    _build_tree_centerlines(xyz, tree_ids, axes, slab_step=1.0, min_points_per_slab=5)
    assert axes[0]["centerline"] is not None
    assert axes[1]["centerline"] is not None
    np.testing.assert_allclose(axes[0]["centerline"][:, 0], 0.0, atol=0.05)
    np.testing.assert_allclose(axes[1]["centerline"][:, 0], 20.0, atol=0.05)
