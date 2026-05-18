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

from src.core.tracking_assignment import filter_by_verticality


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
