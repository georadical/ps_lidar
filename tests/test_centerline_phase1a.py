"""Unit tests for the piecewise centerline construction (Improvement 1).

The active centerline builder is ``build_centerline_from_sections`` in
``src.core.trunk_validation`` (Phase 1A.REAL): for each tree, it concatenates
the circle centres produced by ``compute_stem_sections`` into a polyline,
then applies a moving-median smoothing (F1) across adjacent sections to
suppress per-fit jitter. This is the centerline source the notebook exports
via the ``EXPORT_CENTERLINES`` flag in Module 9.

(The earlier naive median-XY builder ``_build_tree_centerlines`` was removed
along with Phase 1A.REAL — its tests have been deleted; the section-based
builder fully replaces it.)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.trunk_validation import (
    SectionResult,
    StemCleaningConfig,
    build_centerline_from_sections,
)


# ===========================================================================
# Phase 1A.REAL — build_centerline_from_sections
# ===========================================================================

def _make_section_result(
    tree_ids,
    sections_z,
    X_c,
    Y_c,
    R,
    check=None,
    sector_pct=None,
):
    """Helper: build a minimal SectionResult for unit tests.

    Inputs may be lists; they are converted to arrays of the correct shape
    (n_trees, n_sections) for the X_c/Y_c/R/check/sector_pct fields.
    """
    X_c = np.asarray(X_c, dtype=np.float64)
    Y_c = np.asarray(Y_c, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n_trees, n_sections = X_c.shape
    if check is None:
        check = np.where(R > 0, 0, 2).astype(np.float64)
    else:
        check = np.asarray(check, dtype=np.float64)
    if sector_pct is None:
        sector_pct = np.where(R > 0, 100.0, 0.0)
    else:
        sector_pct = np.asarray(sector_pct, dtype=np.float64)
    return SectionResult(
        X_c=X_c,
        Y_c=Y_c,
        R=R,
        check=check,
        sector_pct=sector_pct,
        sections=np.asarray(sections_z, dtype=np.float64),
        tree_ids=list(tree_ids),
        config=StemCleaningConfig(),
    )


def test_section_centerline_happy_path():
    """All sections valid → polyline reproduces (X_c, Y_c, z_section) per row."""
    sections_z = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    X_c = np.array([[1.0, 1.1, 1.2, 1.3, 1.4]])
    Y_c = np.array([[2.0, 2.1, 2.2, 2.3, 2.4]])
    R = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
    sr = _make_section_result([0], sections_z, X_c, Y_c, R)

    centerlines = build_centerline_from_sections(sr)
    assert len(centerlines) == 1
    cl = centerlines[0]
    assert cl is not None
    assert cl.shape == (5, 3)
    np.testing.assert_allclose(cl[:, 0], X_c[0])
    np.testing.assert_allclose(cl[:, 1], Y_c[0])
    np.testing.assert_allclose(cl[:, 2], sections_z)
    assert np.all(np.diff(cl[:, 2]) > 0), "polyline must be sorted by Z"


def test_section_centerline_curved_stem():
    """Section centres that drift with height yield a polyline that follows."""
    sections_z = np.arange(0.3, 10.0, 0.5)
    # Lean: X drifts 0.05 m per metre of height; Y stays at 3.0
    X_c = (0.05 * sections_z)[None, :]
    Y_c = np.full((1, len(sections_z)), 3.0)
    R = np.full((1, len(sections_z)), 0.12)
    sr = _make_section_result([0], sections_z, X_c, Y_c, R)

    cl = build_centerline_from_sections(sr)[0]
    assert cl is not None
    np.testing.assert_allclose(cl[:, 0], 0.05 * cl[:, 2])
    np.testing.assert_allclose(cl[:, 1], 3.0)


def test_section_centerline_skips_invalid_sections():
    """Sections with R == 0 (invalid fit) are excluded from the polyline."""
    sections_z = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    X_c = np.array([[1.0, 99.0, 1.2, 99.0, 1.4]])
    Y_c = np.array([[2.0, 99.0, 2.2, 99.0, 2.4]])
    R = np.array([[0.1, 0.0, 0.1, 0.0, 0.1]])  # 3 valid, 2 invalid
    sr = _make_section_result([0], sections_z, X_c, Y_c, R)

    cl = build_centerline_from_sections(sr)[0]
    assert cl is not None
    assert cl.shape == (3, 3)
    np.testing.assert_allclose(cl[:, 0], [1.0, 1.2, 1.4])
    np.testing.assert_allclose(cl[:, 1], [2.0, 2.2, 2.4])
    np.testing.assert_allclose(cl[:, 2], [0.3, 0.7, 1.1])


def test_section_centerline_below_threshold_returns_none():
    """Tree with fewer than min_valid_sections valid sections → None."""
    sections_z = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    X_c = np.array([[1.0, 99.0, 1.2, 99.0, 99.0]])
    Y_c = np.array([[2.0, 99.0, 2.2, 99.0, 99.0]])
    R = np.array([[0.1, 0.0, 0.1, 0.0, 0.0]])  # only 2 valid
    sr = _make_section_result([0], sections_z, X_c, Y_c, R)

    # Default min_valid_sections=3 — 2 valid is below threshold.
    assert build_centerline_from_sections(sr)[0] is None
    # min_valid_sections=2 — 2 valid is sufficient.
    cl = build_centerline_from_sections(sr, min_valid_sections=2)[0]
    assert cl is not None
    assert cl.shape == (2, 3)


def test_section_centerline_all_invalid_returns_none():
    """Tree with no valid sections returns None."""
    sections_z = np.array([0.3, 0.5, 0.7])
    X_c = np.zeros((1, 3))
    Y_c = np.zeros((1, 3))
    R = np.zeros((1, 3))
    sr = _make_section_result([0], sections_z, X_c, Y_c, R)
    assert build_centerline_from_sections(sr)[0] is None


def test_section_centerline_multiple_trees_indexed_by_position():
    """The output list is indexed positionally to match section_result.tree_ids.

    Trees may have different numbers of valid sections; each gets its own
    polyline (or None) independently of the others.
    """
    sections_z = np.array([0.3, 0.5, 0.7, 0.9])
    # Tree 7: all 4 sections valid at XY=(5, 5)
    # Tree 12: only 2 sections valid → below default threshold → None
    # Tree 30: 3 sections valid at XY=(10, 10) → just meets threshold
    X_c = np.array([
        [5.0, 5.0, 5.0, 5.0],
        [99.0, 99.0, 99.0, 99.0],
        [10.0, 10.0, 99.0, 10.0],
    ])
    Y_c = np.array([
        [5.0, 5.0, 5.0, 5.0],
        [99.0, 99.0, 99.0, 99.0],
        [10.0, 10.0, 99.0, 10.0],
    ])
    R = np.array([
        [0.1, 0.1, 0.1, 0.1],
        [0.0, 0.1, 0.0, 0.1],
        [0.1, 0.1, 0.0, 0.1],
    ])
    sr = _make_section_result([7, 12, 30], sections_z, X_c, Y_c, R)

    centerlines = build_centerline_from_sections(sr)
    assert len(centerlines) == 3
    # Position 0 → tree_id 7
    assert centerlines[0] is not None and centerlines[0].shape == (4, 3)
    np.testing.assert_allclose(centerlines[0][:, 0], 5.0)
    # Position 1 → tree_id 12 → below threshold
    assert centerlines[1] is None
    # Position 2 → tree_id 30 → exactly threshold
    assert centerlines[2] is not None and centerlines[2].shape == (3, 3)
    np.testing.assert_allclose(centerlines[2][:, 0], 10.0)


def test_section_centerline_ignores_ambiguous_check_field():
    """Sections with check==1 are kept if R>0 (valid retry) and dropped if R==0
    (failed retry). This pins the contract that ``check`` is NOT consulted —
    only ``R > 0`` decides validity."""
    sections_z = np.array([0.3, 0.5, 0.7, 0.9])
    X_c = np.array([[1.0, 2.0, 3.0, 4.0]])
    Y_c = np.array([[1.0, 2.0, 3.0, 4.0]])
    R = np.array([[0.1, 0.1, 0.0, 0.1]])
    # check==1 for both R>0 sections (valid retry) AND for the R==0 section
    # (failed retry). The function should keep only the three R>0 rows.
    check = np.array([[1.0, 1.0, 1.0, 1.0]])
    sr = _make_section_result([0], sections_z, X_c, Y_c, R, check=check)

    cl = build_centerline_from_sections(sr)[0]
    assert cl is not None
    np.testing.assert_allclose(cl[:, 0], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(cl[:, 2], [0.3, 0.5, 0.9])


def test_section_centerline_rejects_invalid_min_valid_sections():
    """A polyline needs at least 2 control points by definition."""
    sr = _make_section_result(
        [0], np.array([0.3]), np.array([[0.0]]), np.array([[0.0]]), np.array([[0.1]])
    )
    with pytest.raises(ValueError):
        build_centerline_from_sections(sr, min_valid_sections=1)


def test_section_centerline_empty_tree_list():
    """SectionResult with zero trees → empty list, no error."""
    sr = SectionResult(
        X_c=np.empty((0, 5)),
        Y_c=np.empty((0, 5)),
        R=np.empty((0, 5)),
        check=np.empty((0, 5)),
        sector_pct=np.empty((0, 5)),
        sections=np.linspace(0.3, 1.1, 5),
        tree_ids=[],
        config=StemCleaningConfig(),
    )
    assert build_centerline_from_sections(sr) == []


# ---------------------------------------------------------------------------
# Smoothing (F1) — moving median across adjacent sections
# ---------------------------------------------------------------------------

def test_smoothing_default_kills_isolated_spike():
    """An isolated XY spike surrounded by quiet neighbours is removed by the
    default window=3 moving median. This is the core F1 use case: an LS
    circle fit that mis-fired on one section should not contaminate the
    centerline.
    """
    sections_z = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    X_c = np.array([[0.0, 0.0, 5.0, 0.0, 0.0]])   # spike at section 2
    Y_c = np.array([[0.0, 0.0, 5.0, 0.0, 0.0]])
    R = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])     # all sections valid
    sr = _make_section_result([0], sections_z, X_c, Y_c, R)

    cl = build_centerline_from_sections(sr)[0]
    assert cl is not None
    # Default smoothing (window=3) should obliterate the spike: median of any
    # contiguous triplet including the spike is 0 since the other two are 0.
    np.testing.assert_allclose(cl[:, 0], 0.0)
    np.testing.assert_allclose(cl[:, 1], 0.0)
    # Z grid is unchanged.
    np.testing.assert_allclose(cl[:, 2], sections_z)


def test_smoothing_disabled_window_one_returns_raw_centers():
    """smooth_window=1 disables smoothing — the polyline reproduces X_c, Y_c
    bit-for-bit as fitted (no median filter applied).
    """
    sections_z = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    X_c = np.array([[0.0, 0.0, 5.0, 0.0, 0.0]])
    Y_c = np.array([[0.0, 0.0, 5.0, 0.0, 0.0]])
    R = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
    sr = _make_section_result([0], sections_z, X_c, Y_c, R)

    cl = build_centerline_from_sections(sr, smooth_window=1)[0]
    assert cl is not None
    np.testing.assert_allclose(cl[:, 0], [0.0, 0.0, 5.0, 0.0, 0.0])
    np.testing.assert_allclose(cl[:, 1], [0.0, 0.0, 5.0, 0.0, 0.0])


def test_smoothing_preserves_leaning_trunk():
    """A monotonic XY drift (perfectly leaning trunk) must NOT be flattened by
    smoothing. mode='nearest' on a monotonic series is the identity transform
    everywhere except (potentially) at endpoints, where edge replication still
    preserves the endpoint values exactly.
    """
    sections_z = np.linspace(0.3, 10.0, 25)
    # Lean: X drifts 0.05 m per metre of Z; Y constant
    X_c = (0.05 * sections_z)[None, :]
    Y_c = np.full((1, len(sections_z)), 3.0)
    R = np.full((1, len(sections_z)), 0.12)
    sr = _make_section_result([0], sections_z, X_c, Y_c, R)

    cl_smooth = build_centerline_from_sections(sr, smooth_window=3)[0]
    cl_raw = build_centerline_from_sections(sr, smooth_window=1)[0]
    # Median of [a,b,c] for monotonic ascending is b → smoothing should be
    # the identity for this input.
    np.testing.assert_allclose(cl_smooth[:, 0], cl_raw[:, 0])
    np.testing.assert_allclose(cl_smooth[:, 1], cl_raw[:, 1])


def test_smoothing_rejects_even_window():
    """smooth_window must be a positive odd integer (even windows have no
    well-defined centre point for a median filter on integer indices).
    """
    sr = _make_section_result(
        [0], np.array([0.3, 0.5]), np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0]]), np.array([[0.1, 0.1]])
    )
    with pytest.raises(ValueError):
        build_centerline_from_sections(sr, smooth_window=2)
    with pytest.raises(ValueError):
        build_centerline_from_sections(sr, smooth_window=0)
