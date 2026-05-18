"""Smoke + sanity tests for ``compute_stem_sections_ellipse`` (Mejora 1
Phase 1C — EF.1).

Integration coverage is intentionally light here: the per-section fit
correctness is exhaustively covered in ``test_ellipse_fitting.py``
(47 unit tests on the EL.1–EL.5 primitives). What we validate at this
level is the **drop-in contract** with ``compute_stem_sections``:

  - Same return type (``SectionResult``) and field shapes.
  - Same loop semantics over trees and horizontal sections.
  - ``R`` stores the equivalent radius √(a·b) of each section's ellipse,
    consumed unchanged by ``build_centerline_from_sections`` downstream.

The visual gate for this sub-fase lives in the notebook (Module 7B
flag ``USE_ELLIPSE_FIT``) and compares circle- vs ellipse-derived
centerlines in CloudCompare. Production-scale correctness is a visual
exercise, not a unit-test one.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from src.core.trunk_validation import (
    SectionResult,
    StemCleaningConfig,
    compute_stem_sections,
    compute_stem_sections_ellipse,
    build_centerline_from_sections,
)


# ===========================================================================
# Synthetic-cylinder fixture
# ===========================================================================

def _synthetic_cylinder(
    radius: float = 0.15,
    height: float = 8.0,
    n_points_per_m_of_height: int = 800,
    centre_xy: Tuple[float, float] = (1.0, 2.0),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a perfectly cylindrical stem aligned with +z.

    Returns ``(xyz, stem_mask, tree_ids)`` ready to feed
    ``compute_stem_sections_ellipse``. Every point is on the cylinder
    surface; no noise, no outliers — the gate here is *contract*, not
    fit quality.
    """
    rng = np.random.default_rng(seed=seed)
    n_total = int(n_points_per_m_of_height * height)
    z = rng.uniform(0.0, height, size=n_total)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n_total)
    x = centre_xy[0] + radius * np.cos(angle)
    y = centre_xy[1] + radius * np.sin(angle)
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    stem_mask = np.ones(n_total, dtype=bool)
    tree_ids = np.zeros(n_total, dtype=np.int32)
    return xyz, stem_mask, tree_ids


def _section_config(
    minimum_height: float = 0.3,
    maximum_height: float = 8.0,
    section_len: float = 0.2,
    section_wid: float = 0.05,
) -> StemCleaningConfig:
    """A minimal StemCleaningConfig for sectioning-only tests (we never
    call clean_stems here so the 2nd-pass-verticality fields are inert).
    """
    return StemCleaningConfig(
        mode="global",
        verticality_threshold=0.6,
        verticality_scale=0.1,
        voxel_resolution_xy=0.02,
        voxel_resolution_z=0.02,
        section_len=section_len,
        section_wid=section_wid,
        min_points_section=40,
        r_min=0.05,
        r_max=0.40,
        n_sectors=16,
        min_sectors=9,
        sector_width=0.02,
        inner_circle_ratio=0.5,
        max_inner_points=5,
        minimum_height=minimum_height,
        maximum_height=maximum_height,
        cluster_eps=0.02,
    )


# ===========================================================================
# Contract tests
# ===========================================================================

class TestComputeStemSectionsEllipse:

    def test_returns_section_result_with_correct_shape(self):
        xyz, mask, tids = _synthetic_cylinder()
        cfg = _section_config()
        rng = np.random.default_rng(seed=42)

        result = compute_stem_sections_ellipse(
            xyz, mask, tids, cfg, verbose=False, rng=rng,
        )

        assert isinstance(result, SectionResult)
        n_sections_expected = len(
            np.arange(cfg.minimum_height, cfg.maximum_height, cfg.section_len)
        )
        assert result.X_c.shape == (1, n_sections_expected)
        assert result.Y_c.shape == (1, n_sections_expected)
        assert result.R.shape == (1, n_sections_expected)
        assert result.check.shape == (1, n_sections_expected)
        assert result.sector_pct.shape == (1, n_sections_expected)
        assert result.sections.shape == (n_sections_expected,)
        assert result.tree_ids == [0]

    def test_perfect_cylinder_recovers_radius_and_centre(self):
        xyz, mask, tids = _synthetic_cylinder(
            radius=0.15, height=8.0, centre_xy=(1.0, 2.0),
        )
        cfg = _section_config()
        rng = np.random.default_rng(seed=42)

        result = compute_stem_sections_ellipse(
            xyz, mask, tids, cfg, verbose=False, rng=rng,
        )

        valid = result.R[0] > 0
        n_valid = int(valid.sum())
        # On a perfect cylinder with dense sampling the vast majority
        # of sections must fit cleanly.
        assert n_valid >= int(0.9 * valid.size)

        # Equivalent radius √(a·b) ≈ 0.15 m on a circular cross-section.
        np.testing.assert_allclose(result.R[0, valid], 0.15, atol=2e-3)
        np.testing.assert_allclose(result.X_c[0, valid], 1.0, atol=2e-3)
        np.testing.assert_allclose(result.Y_c[0, valid], 2.0, atol=2e-3)

    def test_drop_in_with_build_centerline(self):
        # The whole point of EF.1: ellipse-derived SectionResult must
        # flow into build_centerline_from_sections unchanged.
        xyz, mask, tids = _synthetic_cylinder()
        cfg = _section_config()
        rng = np.random.default_rng(seed=42)

        ellipse_result = compute_stem_sections_ellipse(
            xyz, mask, tids, cfg, verbose=False, rng=rng,
        )
        centerlines = build_centerline_from_sections(ellipse_result)

        assert len(centerlines) == 1
        cl = centerlines[0]
        assert cl is not None
        assert cl.shape[1] == 3  # (X_c, Y_c, z_section)
        # All control points should sit near the true axis (1.0, 2.0, *).
        np.testing.assert_allclose(cl[:, 0], 1.0, atol=2e-3)
        np.testing.assert_allclose(cl[:, 1], 2.0, atol=2e-3)
        # And Z must be monotonically ascending.
        assert np.all(np.diff(cl[:, 2]) > 0)

    def test_circle_vs_ellipse_consistent_on_circular_cross_section(self):
        # On a perfectly circular cross-section the ellipse fit must
        # collapse to a circle, so R-values should agree with the
        # circle pipeline within a small tolerance.
        xyz, mask, tids = _synthetic_cylinder(radius=0.15)
        cfg = _section_config()
        rng = np.random.default_rng(seed=42)

        r_circle = compute_stem_sections(xyz, mask, tids, cfg, verbose=False)
        r_ellipse = compute_stem_sections_ellipse(
            xyz, mask, tids, cfg, verbose=False, rng=rng,
        )

        # Take sections that are valid in both.
        both_valid = (r_circle.R[0] > 0) & (r_ellipse.R[0] > 0)
        assert both_valid.sum() >= int(0.8 * len(both_valid))

        np.testing.assert_allclose(
            r_ellipse.R[0, both_valid],
            r_circle.R[0, both_valid],
            atol=2e-3,
        )
        np.testing.assert_allclose(
            r_ellipse.X_c[0, both_valid],
            r_circle.X_c[0, both_valid],
            atol=2e-3,
        )
        np.testing.assert_allclose(
            r_ellipse.Y_c[0, both_valid],
            r_circle.Y_c[0, both_valid],
            atol=2e-3,
        )

    def test_empty_stem_mask_returns_empty_result(self):
        # No stem points → no trees → empty SectionResult arrays.
        xyz = np.zeros((50, 3), dtype=np.float64)
        mask = np.zeros(50, dtype=bool)
        tids = np.full(50, -1, dtype=np.int32)
        cfg = _section_config()

        result = compute_stem_sections_ellipse(xyz, mask, tids, cfg, verbose=False)

        assert result.X_c.shape[0] == 0
        assert result.tree_ids == []
