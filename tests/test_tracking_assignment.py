"""Tests for ``src.core.tracking_assignment`` — Phase 1B real (GS block).

GS.1 (``filter_by_verticality``) is exercised here with two synthetic
clouds:

  * a vertical pillar (stem-like) → must be kept,
  * a horizontal slab (ground / canopy-like) → must be rejected.

The function is a thin wrapper over the well-tested
``compute_verticality_mask_early_exit``, so we don't repeat the
exhaustive numerical coverage that lives in
``tests/test_segmentation.py`` and the feature tests. What we validate
here is the **API contract** at the GS.1 surface and the qualitative
keep/reject behaviour on synthetic geometric extremes.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.ellipse_fitting import EllipseFitConfig
from src.core.tracking_assignment import (
    Cluster2D,
    ClusterEllipse,
    HorizontalSlice,
    cluster_slice,
    filter_by_verticality,
    fit_ellipses_in_slice,
    slice_horizontal_global,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _vertical_pillar(
    n: int = 4000,
    radius: float = 0.15,
    height: float = 8.0,
    centre_xy=(0.0, 0.0),
    rng_seed: int = 0,
) -> np.ndarray:
    """Cylindrical column aligned with +z — high verticality everywhere."""
    rng = np.random.default_rng(seed=rng_seed)
    z = rng.uniform(0.0, height, size=n)
    a = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = centre_xy[0] + radius * np.cos(a)
    y = centre_xy[1] + radius * np.sin(a)
    return np.column_stack([x, y, z]).astype(np.float64)


def _horizontal_slab(
    n: int = 4000,
    side: float = 4.0,
    z: float = 0.05,
    rng_seed: int = 1,
) -> np.ndarray:
    """Flat layer at fixed z (ground-like) — verticality near zero."""
    rng = np.random.default_rng(seed=rng_seed)
    x = rng.uniform(-side / 2.0, side / 2.0, size=n)
    y = rng.uniform(-side / 2.0, side / 2.0, size=n)
    z_arr = np.full(n, z, dtype=np.float64) + rng.normal(0, 0.005, size=n)
    return np.column_stack([x, y, z_arr]).astype(np.float64)


# ===========================================================================
# API-contract tests
# ===========================================================================

class TestFilterByVerticalityContract:

    def test_returns_filtered_cloud_and_mask_aligned(self):
        xyz = _vertical_pillar(n=2000)
        filt, mask = filter_by_verticality(xyz, verbose=False)

        assert isinstance(filt, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (xyz.shape[0],)
        # Subset relationship: filtered = xyz[mask]
        np.testing.assert_array_equal(filt, xyz[mask])

    def test_rejects_invalid_xyz_shape(self):
        with pytest.raises(ValueError):
            filter_by_verticality(np.zeros((10, 2)))
        with pytest.raises(ValueError):
            filter_by_verticality(np.zeros((10,)))

    def test_rejects_out_of_range_threshold(self):
        xyz = _vertical_pillar(n=100)
        with pytest.raises(ValueError):
            filter_by_verticality(xyz, threshold=-0.1)
        with pytest.raises(ValueError):
            filter_by_verticality(xyz, threshold=1.5)


# ===========================================================================
# Qualitative keep/reject behaviour
# ===========================================================================

class TestFilterByVerticalityBehaviour:

    def test_pillar_mostly_kept(self):
        # A near-perfect vertical pillar must retain the vast majority
        # of its points at the default threshold.
        xyz = _vertical_pillar(n=4000, radius=0.15, height=8.0)
        _filt, mask = filter_by_verticality(xyz, threshold=0.7)
        keep_ratio = float(mask.mean())
        assert keep_ratio >= 0.90, f"pillar keep ratio {keep_ratio:.2%} too low"

    def test_horizontal_slab_mostly_rejected(self):
        # A horizontal slab should have very low verticality and be
        # mostly rejected. We accept up to ~20% keep rate to absorb
        # edge-of-slab voxels whose pgeof neighbourhood includes the
        # vertical jitter in z noise (σ=5 mm). The strong qualitative
        # test is the asymmetry vs the pillar, exercised below.
        xyz = _horizontal_slab(n=4000, side=4.0)
        _filt, mask = filter_by_verticality(xyz, threshold=0.7)
        keep_ratio = float(mask.mean())
        assert keep_ratio <= 0.20, f"slab keep ratio {keep_ratio:.2%} too high"

    def test_mixed_cloud_separates_components(self):
        # Combined pillar + slab: the pillar portion of the output mask
        # should be mostly True, the slab portion mostly False.
        pillar = _vertical_pillar(n=3000, radius=0.15, height=8.0)
        slab = _horizontal_slab(n=3000, side=4.0, z=0.05)
        xyz = np.concatenate([pillar, slab], axis=0)

        _filt, mask = filter_by_verticality(xyz, threshold=0.7)

        pillar_keep = float(mask[: pillar.shape[0]].mean())
        slab_keep = float(mask[pillar.shape[0]:].mean())

        # Asymmetry: the pillar side must keep far more points than the
        # slab side — that's the whole point of the filter.
        assert pillar_keep > slab_keep + 0.5, (
            f"pillar keep {pillar_keep:.2%} vs slab keep {slab_keep:.2%} "
            "— not enough separation"
        )

    def test_threshold_monotonicity(self):
        # Higher threshold → fewer points kept (monotonically).
        xyz = _vertical_pillar(n=4000, radius=0.15, height=8.0)
        _f1, m1 = filter_by_verticality(xyz, threshold=0.5)
        _f2, m2 = filter_by_verticality(xyz, threshold=0.85)
        assert m1.sum() >= m2.sum(), (
            f"raising threshold did not shrink the kept set: "
            f"{m1.sum()} vs {m2.sum()}"
        )


# ===========================================================================
# GS.2 — slice_horizontal_global
# ===========================================================================

def _uniform_pillar(n: int, height: float, rng_seed: int = 0) -> np.ndarray:
    """Pillar with z uniform on [0, height] — used to test slice membership."""
    rng = np.random.default_rng(seed=rng_seed)
    z = rng.uniform(0.0, height, size=n)
    a = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = 0.15 * np.cos(a)
    y = 0.15 * np.sin(a)
    return np.column_stack([x, y, z]).astype(np.float64)


class TestSliceHorizontalGlobalContract:

    def test_returns_list_of_horizontal_slices(self):
        xyz = _uniform_pillar(n=500, height=5.0)
        slices = slice_horizontal_global(xyz, z_min=0.5, z_max=4.5)
        assert isinstance(slices, list)
        assert all(isinstance(s, HorizontalSlice) for s in slices)

    def test_empty_cloud_still_returns_slabs(self):
        # Empty input → every slab is still listed, but with empty
        # indices arrays. Lets downstream callers track coverage gaps.
        xyz = np.empty((0, 3), dtype=np.float64)
        slices = slice_horizontal_global(xyz, z_min=0.0, z_max=5.0, slab_step=1.0)
        assert len(slices) == 5
        for s in slices:
            assert s.indices.size == 0

    def test_rejects_invalid_shape(self):
        with pytest.raises(ValueError):
            slice_horizontal_global(np.zeros((10, 2)))

    def test_rejects_invalid_z_range(self):
        xyz = _uniform_pillar(n=100, height=5.0)
        with pytest.raises(ValueError):
            slice_horizontal_global(xyz, z_min=5.0, z_max=5.0)
        with pytest.raises(ValueError):
            slice_horizontal_global(xyz, z_min=5.0, z_max=2.0)

    def test_rejects_non_positive_step_or_thickness(self):
        xyz = _uniform_pillar(n=100, height=5.0)
        with pytest.raises(ValueError):
            slice_horizontal_global(xyz, slab_step=0.0)
        with pytest.raises(ValueError):
            slice_horizontal_global(xyz, slab_step=-1.0)
        with pytest.raises(ValueError):
            slice_horizontal_global(xyz, slab_half_thickness=0.0)


class TestSliceHorizontalGlobalBehaviour:

    def test_slab_centres_match_arange(self):
        # Slab centres are np.arange(z_min, z_max, step).
        xyz = _uniform_pillar(n=500, height=10.0)
        slices = slice_horizontal_global(
            xyz, z_min=0.5, z_max=9.5, slab_step=1.0,
        )
        expected_centres = np.arange(0.5, 9.5, 1.0)
        actual_centres = np.array([s.z_centre for s in slices])
        np.testing.assert_array_equal(actual_centres, expected_centres)

    def test_slab_bounds_are_centre_plus_minus_half(self):
        xyz = _uniform_pillar(n=100, height=5.0)
        slices = slice_horizontal_global(
            xyz, z_min=1.0, z_max=3.0, slab_step=1.0, slab_half_thickness=0.3,
        )
        # Two slabs at z=1.0, 2.0
        s0 = slices[0]
        assert s0.z_centre == 1.0
        assert s0.z_low == pytest.approx(0.7)
        assert s0.z_high == pytest.approx(1.3)

    def test_indices_actually_in_z_range(self):
        # Every index returned must point to a point whose z is in the
        # slab's [z_low, z_high) range.
        rng = np.random.default_rng(seed=42)
        n = 5000
        z = rng.uniform(0.0, 10.0, size=n)
        x = rng.uniform(-1, 1, size=n)
        y = rng.uniform(-1, 1, size=n)
        xyz = np.column_stack([x, y, z])

        slices = slice_horizontal_global(
            xyz, z_min=0.5, z_max=9.5, slab_step=1.0,
        )
        for s in slices:
            assigned_z = xyz[s.indices, 2]
            assert np.all(assigned_z >= s.z_low)
            assert np.all(assigned_z < s.z_high)

    def test_no_overlap_no_gap_when_half_equals_step_over_two(self):
        # The canonical config: step=1.0, half=0.5 → with slab centres
        # at np.arange(z_min, z_max, step) = [0.5, 1.5, ..., 8.5], the
        # slabs cover the half-open range [0.0, 9.0). Every point in
        # that range must be assigned to exactly one slab.
        rng = np.random.default_rng(seed=7)
        n = 4000
        # Restrict the point cloud to the coverage range so the test
        # actually probes the no-gap-no-overlap property.
        z = rng.uniform(0.0, 9.0, size=n)
        x = rng.uniform(-1, 1, size=n)
        y = rng.uniform(-1, 1, size=n)
        xyz = np.column_stack([x, y, z])

        slices = slice_horizontal_global(
            xyz, z_min=0.5, z_max=9.5, slab_step=1.0, slab_half_thickness=0.5,
        )
        total_assigned = sum(s.indices.size for s in slices)
        assert total_assigned == n, (
            f"expected {n} assignments, got {total_assigned} "
            "— either gap or overlap"
        )

        # No duplicates: union of all index arrays has n unique elements.
        all_idx = np.concatenate([s.indices for s in slices])
        assert np.unique(all_idx).size == n

    def test_overlap_when_half_greater_than_step_over_two(self):
        # Slabs overlap: a point near a slab boundary may appear in two.
        rng = np.random.default_rng(seed=7)
        n = 1000
        z = rng.uniform(0.0, 5.0, size=n)
        x = np.zeros(n)
        y = np.zeros(n)
        xyz = np.column_stack([x, y, z])

        slices = slice_horizontal_global(
            xyz, z_min=0.5, z_max=4.5, slab_step=1.0, slab_half_thickness=0.7,
        )
        total_assigned = sum(s.indices.size for s in slices)
        # Some points are in two slabs → total assigned exceeds n
        # (provided the centre z range has decent coverage).
        assert total_assigned > int(0.8 * n) and total_assigned <= 2 * n

    def test_pillar_distributes_points_across_slabs(self):
        # A uniform-z pillar should put roughly equal counts in each slab.
        xyz = _uniform_pillar(n=10000, height=10.0)
        slices = slice_horizontal_global(
            xyz, z_min=0.5, z_max=9.5, slab_step=1.0, slab_half_thickness=0.5,
        )
        counts = np.array([s.indices.size for s in slices])
        # 9 slabs covering 9 m of a 10 m uniform pillar → ~1000 each.
        # Allow ±25 % variance for the uniform draw.
        assert counts.min() > 700
        assert counts.max() < 1300


# ===========================================================================
# GS.3 — cluster_slice (DBSCAN on slice XY)
# ===========================================================================

def _ring_at_z(
    cx: float, cy: float, radius: float, z: float,
    n: int = 200, rng_seed: int = 0,
) -> np.ndarray:
    """Generate `n` points forming an XY ring at a fixed height."""
    rng = np.random.default_rng(seed=rng_seed)
    a = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = cx + radius * np.cos(a)
    y = cy + radius * np.sin(a)
    z_arr = np.full(n, z, dtype=np.float64) + rng.normal(0, 0.01, size=n)
    return np.column_stack([x, y, z_arr]).astype(np.float64)


def _whole_slice(xyz: np.ndarray, z_centre: float) -> HorizontalSlice:
    """Build a HorizontalSlice that selects all of `xyz`."""
    return HorizontalSlice(
        z_centre=z_centre,
        z_low=z_centre - 100.0,
        z_high=z_centre + 100.0,
        indices=np.arange(xyz.shape[0], dtype=np.int64),
    )


class TestClusterSliceContract:

    def test_returns_list_of_cluster2d(self):
        xyz = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=200)
        sl = _whole_slice(xyz, z_centre=1.5)
        clusters = cluster_slice(xyz, sl)
        assert isinstance(clusters, list)
        assert all(isinstance(c, Cluster2D) for c in clusters)

    def test_empty_slice_returns_empty_list(self):
        xyz = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=200)
        empty = HorizontalSlice(
            z_centre=10.0, z_low=9.5, z_high=10.5,
            indices=np.empty(0, dtype=np.int64),
        )
        assert cluster_slice(xyz, empty) == []

    def test_indices_refer_to_input_xyz_not_slice_subset(self):
        # Build a cloud where the slice selects a non-contiguous subset
        # and verify the returned cluster indices are in the input-cloud
        # coordinate system.
        ring = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=120)
        noise_before = np.random.default_rng(7).normal(size=(50, 3))
        noise_before[:, 2] = -10.0  # well below slab
        noise_after = np.random.default_rng(8).normal(size=(50, 3))
        noise_after[:, 2] = 100.0   # well above slab
        xyz = np.vstack([noise_before, ring, noise_after])

        # Slice spans only the ring region.
        slice_obj = HorizontalSlice(
            z_centre=1.5, z_low=1.0, z_high=2.0,
            indices=np.arange(50, 50 + 120, dtype=np.int64),
        )
        clusters = cluster_slice(xyz, slice_obj, eps=0.05, min_samples=5)
        assert len(clusters) >= 1
        # All cluster indices must be inside [50, 170)
        for c in clusters:
            assert c.indices.min() >= 50
            assert c.indices.max() < 50 + 120

    def test_rejects_invalid_shape(self):
        with pytest.raises(ValueError):
            cluster_slice(np.zeros((10, 2)), _whole_slice(np.zeros((10, 3)), 0.0))

    def test_rejects_non_positive_eps_or_min_samples(self):
        xyz = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=50)
        sl = _whole_slice(xyz, z_centre=1.5)
        with pytest.raises(ValueError):
            cluster_slice(xyz, sl, eps=0.0)
        with pytest.raises(ValueError):
            cluster_slice(xyz, sl, eps=-0.05)
        with pytest.raises(ValueError):
            cluster_slice(xyz, sl, min_samples=0)


class TestClusterSliceBehaviour:

    def test_single_well_formed_ring_makes_one_cluster(self):
        xyz = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=200)
        sl = _whole_slice(xyz, z_centre=1.5)
        clusters = cluster_slice(xyz, sl, eps=0.10, min_samples=5)
        assert len(clusters) == 1
        c = clusters[0]
        assert c.n_points == 200
        # Centroid near (0, 0)
        np.testing.assert_allclose(c.centroid_xy, [0.0, 0.0], atol=0.02)

    def test_two_well_separated_rings_make_two_clusters(self):
        # Two rings centred 2 m apart — way more than eps=0.10
        ring_a = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=200, rng_seed=1)
        ring_b = _ring_at_z(2.0, 0.0, 0.15, 1.5, n=200, rng_seed=2)
        xyz = np.vstack([ring_a, ring_b])
        sl = _whole_slice(xyz, z_centre=1.5)
        clusters = cluster_slice(xyz, sl, eps=0.10, min_samples=5)
        assert len(clusters) == 2
        # Each cluster has ~200 points (no cross-contamination).
        sizes = sorted(c.n_points for c in clusters)
        assert sizes == [200, 200]

    def test_pure_noise_returns_no_clusters(self):
        rng = np.random.default_rng(seed=42)
        # Scatter sparse random points so DBSCAN labels everything as noise
        # at the default min_samples=10.
        xyz = rng.uniform(-5.0, 5.0, size=(80, 3))
        xyz[:, 2] = 1.5
        sl = _whole_slice(xyz, z_centre=1.5)
        clusters = cluster_slice(xyz, sl, eps=0.05, min_samples=10)
        assert clusters == []

    def test_eps_monotonicity(self):
        # Two rings 0.4 m apart. Small eps separates them; large eps
        # merges them.
        ring_a = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=200, rng_seed=1)
        ring_b = _ring_at_z(0.4, 0.0, 0.15, 1.5, n=200, rng_seed=2)
        xyz = np.vstack([ring_a, ring_b])
        sl = _whole_slice(xyz, z_centre=1.5)

        small_eps = cluster_slice(xyz, sl, eps=0.05, min_samples=5)
        big_eps = cluster_slice(xyz, sl, eps=0.30, min_samples=5)
        assert len(small_eps) >= len(big_eps), (
            f"raising eps did not reduce cluster count: "
            f"{len(small_eps)} → {len(big_eps)}"
        )

    def test_centroid_matches_mean_of_cluster_points(self):
        ring = _ring_at_z(1.0, 2.5, 0.15, 1.5, n=200, rng_seed=9)
        sl = _whole_slice(ring, z_centre=1.5)
        clusters = cluster_slice(ring, sl, eps=0.10, min_samples=5)
        assert len(clusters) == 1
        c = clusters[0]
        expected = ring[c.indices, :2].mean(axis=0)
        np.testing.assert_allclose(c.centroid_xy, expected, atol=1e-9)


# ===========================================================================
# GS.4 — fit_ellipses_in_slice
# ===========================================================================

def _stem_ellipse_config(**overrides) -> EllipseFitConfig:
    """Loose-ish EllipseFitConfig sized for synthetic test rings
    (radius ~0.15 m, ~150 points per ring). Mirrors the helper used
    in `test_ellipse_fitting.py`."""
    base = dict(
        min_points_section=40,
        r_min=0.05,
        r_max=0.40,
        inner_ratio=0.5,
        max_inner_points=5,
        n_sectors=16,
        min_sectors=9,
        sector_width=0.02,
        ransac_n_iters=200,
        ransac_tau_sampson=0.005,
        min_inlier_fraction=0.6,
        min_aspect_ratio=0.5,
        cluster_eps=0.02,
    )
    base.update(overrides)
    return EllipseFitConfig(**base)


class TestFitEllipsesInSliceContract:

    def test_empty_input_returns_empty_list(self):
        xyz = np.empty((0, 3), dtype=np.float64)
        cfg = _stem_ellipse_config()
        assert fit_ellipses_in_slice(xyz, [], cfg) == []

    def test_returns_list_of_cluster_ellipse(self):
        ring = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=150, rng_seed=0)
        sl = _whole_slice(ring, z_centre=1.5)
        clusters = cluster_slice(ring, sl, eps=0.10, min_samples=5)
        cfg = _stem_ellipse_config()
        rng = np.random.default_rng(seed=42)

        results = fit_ellipses_in_slice(ring, clusters, cfg, rng=rng)
        assert isinstance(results, list)
        assert all(isinstance(r, ClusterEllipse) for r in results)


class TestFitEllipsesInSliceBehaviour:

    def test_single_ring_fitted_within_tolerance(self):
        ring = _ring_at_z(0.5, -0.5, 0.15, 1.5, n=200, rng_seed=0)
        sl = _whole_slice(ring, z_centre=1.5)
        clusters = cluster_slice(ring, sl, eps=0.10, min_samples=5)
        cfg = _stem_ellipse_config()
        rng = np.random.default_rng(seed=42)

        results = fit_ellipses_in_slice(ring, clusters, cfg, rng=rng)
        assert len(results) == 1
        r = results[0]
        # Indices preserved
        np.testing.assert_array_equal(r.indices, clusters[0].indices)
        # Geometry recovered
        assert abs(r.xc - 0.5) < 5e-3
        assert abs(r.yc - (-0.5)) < 5e-3
        assert abs(r.a - 0.15) < 5e-3
        assert abs(r.b - 0.15) < 5e-3
        assert r.a >= r.b  # convention
        assert r.check_status in (0, 1)
        assert r.n_points == clusters[0].n_points

    def test_two_clusters_both_fitted(self):
        ring_a = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=200, rng_seed=1)
        ring_b = _ring_at_z(2.0, 0.0, 0.15, 1.5, n=200, rng_seed=2)
        xyz = np.vstack([ring_a, ring_b])
        sl = _whole_slice(xyz, z_centre=1.5)
        clusters = cluster_slice(xyz, sl, eps=0.10, min_samples=5)
        assert len(clusters) == 2  # sanity from GS.3
        cfg = _stem_ellipse_config()
        rng = np.random.default_rng(seed=42)

        results = fit_ellipses_in_slice(xyz, clusters, cfg, rng=rng)
        assert len(results) == 2
        # Centres match the two ring centroids (order may vary).
        centres = sorted([(r.xc, r.yc) for r in results])
        np.testing.assert_allclose(centres[0], (0.0, 0.0), atol=5e-3)
        np.testing.assert_allclose(centres[1], (2.0, 0.0), atol=5e-3)

    def test_undersized_cluster_dropped(self):
        # 10 points → below min_points_section=40 → _fit_ellipse_check
        # returns status 2 with zeros → dropped by the filter.
        rng_geom = np.random.default_rng(seed=0)
        n = 10
        a = rng_geom.uniform(0.0, 2.0 * np.pi, size=n)
        x = 0.15 * np.cos(a)
        y = 0.15 * np.sin(a)
        z = np.full(n, 1.5)
        xyz = np.column_stack([x, y, z])
        cluster = Cluster2D(
            indices=np.arange(n, dtype=np.int64),
            centroid_xy=np.array([0.0, 0.0]),
            n_points=n,
        )
        cfg = _stem_ellipse_config()
        rng = np.random.default_rng(seed=42)

        results = fit_ellipses_in_slice(xyz, [cluster], cfg, rng=rng)
        assert results == []

    def test_pure_noise_cluster_dropped(self):
        # Random points spread over a 1 m square at fixed z → no ellipse
        # passes the quality checks → cluster dropped.
        rng_geom = np.random.default_rng(seed=99)
        n = 200
        x = rng_geom.uniform(-0.5, 0.5, size=n)
        y = rng_geom.uniform(-0.5, 0.5, size=n)
        z = np.full(n, 1.5)
        xyz = np.column_stack([x, y, z])
        cluster = Cluster2D(
            indices=np.arange(n, dtype=np.int64),
            centroid_xy=np.array([0.0, 0.0]),
            n_points=n,
        )
        cfg = _stem_ellipse_config()
        rng = np.random.default_rng(seed=42)

        results = fit_ellipses_in_slice(xyz, [cluster], cfg, rng=rng)
        assert results == []

    def test_a_at_least_b_invariant(self):
        # Even on a slightly elongated ellipse, the wrapper must respect
        # the a ≥ b convention.
        rng_geom = np.random.default_rng(seed=3)
        n = 200
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = 0.18 * np.cos(t)
        y = 0.12 * np.sin(t)
        z = np.full(n, 1.5)
        xyz = np.column_stack([x, y, z])
        cluster = Cluster2D(
            indices=np.arange(n, dtype=np.int64),
            centroid_xy=np.array([0.0, 0.0]),
            n_points=n,
        )
        cfg = _stem_ellipse_config()
        rng = np.random.default_rng(seed=42)

        results = fit_ellipses_in_slice(xyz, [cluster], cfg, rng=rng)
        assert len(results) == 1
        r = results[0]
        assert r.a >= r.b
        assert abs(r.a - 0.18) < 5e-3
        assert abs(r.b - 0.12) < 5e-3

    def test_reproducible_with_seed(self):
        ring = _ring_at_z(0.0, 0.0, 0.15, 1.5, n=200, rng_seed=11)
        sl = _whole_slice(ring, z_centre=1.5)
        clusters = cluster_slice(ring, sl, eps=0.10, min_samples=5)
        cfg = _stem_ellipse_config()

        rng_a = np.random.default_rng(seed=999)
        rng_b = np.random.default_rng(seed=999)
        ra = fit_ellipses_in_slice(ring, clusters, cfg, rng=rng_a)
        rb = fit_ellipses_in_slice(ring, clusters, cfg, rng=rng_b)
        assert ra == rb
