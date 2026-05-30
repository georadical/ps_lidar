"""Tests for ``src.core.dendrometry`` — per-tree DBH, height, ovality.

Synthetic-only tests. The dendrometry layer is a pure consumer of
``TrunkExtractionResult`` and ``SectionResult``; it doesn't fit anything
itself. So the tests build minimal in-memory result objects with known
ground-truth values and verify the arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest

from src.core.dendrometry import (
    TreeMetrics,
    compute_tree_metrics,
    tree_metrics_to_dataframe,
)
from src.core.trunk_validation import SectionResult, StemCleaningConfig


# ===========================================================================
# Minimal stand-in for TrunkExtractionResult
# ===========================================================================
# The dendrometry layer reads only `.tree_ids` from TrunkExtractionResult,
# so a tiny duck-typed stub keeps the tests independent of the full
# extraction pipeline.

@dataclass
class _StubTrunkResult:
    tree_ids: np.ndarray


# ===========================================================================
# Helpers to build synthetic per-tree fixtures
# ===========================================================================

def _make_section_result(
    tree_ids: List[int],
    sections: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    theta: np.ndarray = None,
) -> SectionResult:
    """Build a SectionResult with the (a, b, theta) fields populated.

    R is derived as √(a · b) — consistent with the ellipse path's
    invariant — and `check` is 0 where R > 0, 1 elsewhere.
    """
    R = np.where((a > 0) & (b > 0), np.sqrt(a * b), 0.0)
    if theta is None:
        theta = np.zeros_like(a)
    check = np.where(R > 0.0, 0.0, 1.0)
    sector_pct = np.where(R > 0.0, 100.0, 0.0)
    return SectionResult(
        X_c=np.zeros_like(R),
        Y_c=np.zeros_like(R),
        R=R,
        check=check,
        sector_pct=sector_pct,
        sections=sections,
        tree_ids=list(tree_ids),
        config=StemCleaningConfig(),
        a=a,
        b=b,
        theta=theta,
    )


def _synthetic_cloud_for_trees(
    per_tree_heights: List[tuple],  # list of (tree_id, z_base, z_top)
    points_per_tree: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (xyz, tree_ids) pair with given z-extent per tree."""
    xyz_parts = []
    ids_parts = []
    rng = np.random.default_rng(seed=0)
    for tid, z_base, z_top in per_tree_heights:
        z = rng.uniform(z_base, z_top, size=points_per_tree)
        xy = rng.uniform(-0.2, 0.2, size=(points_per_tree, 2))
        xyz_parts.append(np.column_stack([xy[:, 0], xy[:, 1], z]))
        ids_parts.append(np.full(points_per_tree, tid, dtype=np.int32))
    return np.vstack(xyz_parts), np.concatenate(ids_parts)


# ===========================================================================
# Tests
# ===========================================================================

class TestComputeTreeMetrics:

    def test_perfect_cylinder_circle_path_dbh_and_height(self):
        # Tree 0: cylinder r=0.15m, z in [0.2, 10.5] → height 10.3, DBH 0.30,
        # ovality 1.0 (since a = b = R = 0.15 in the circle convention).
        sections = np.arange(0.3, 10.0, 0.2)
        n_secs = len(sections)
        a = np.full((1, n_secs), 0.15)  # circle: a = R
        b = np.full((1, n_secs), 0.15)
        sr = _make_section_result([0], sections, a, b)

        xyz, tree_ids = _synthetic_cloud_for_trees([(0, 0.2, 10.5)])
        tr = _StubTrunkResult(tree_ids=tree_ids)

        ms = compute_tree_metrics(xyz, tr, sr)
        assert len(ms) == 1
        m = ms[0]
        assert m.tree_id == 0
        assert m.valid_at_dbh is True
        assert abs(m.dbh - 0.30) < 1e-9
        assert abs(m.dbh_major - 0.30) < 1e-9
        assert abs(m.dbh_minor - 0.30) < 1e-9
        assert abs(m.ovality_at_dbh - 1.0) < 1e-9
        assert m.is_oval_at_dbh is False
        # Height: range of uniform z samples → close to but not exactly 10.3
        assert m.height_total == pytest.approx(10.3, abs=0.05)
        assert m.n_points == 1000

    def test_oval_section_flagged(self):
        # Tree 0: ellipse a=0.20, b=0.10 → DBH = 2·√(0.02) ≈ 0.2828,
        # ovality = 2.0, is_oval = True.
        sections = np.arange(0.3, 10.0, 0.2)
        n_secs = len(sections)
        a = np.full((1, n_secs), 0.20)
        b = np.full((1, n_secs), 0.10)
        sr = _make_section_result([0], sections, a, b)

        xyz, tree_ids = _synthetic_cloud_for_trees([(0, 0.0, 10.0)])
        tr = _StubTrunkResult(tree_ids=tree_ids)

        ms = compute_tree_metrics(xyz, tr, sr)
        m = ms[0]
        assert abs(m.dbh - 2.0 * np.sqrt(0.02)) < 1e-9
        assert abs(m.dbh_major - 0.40) < 1e-9
        assert abs(m.dbh_minor - 0.20) < 1e-9
        assert abs(m.ovality_at_dbh - 2.0) < 1e-9
        assert m.is_oval_at_dbh is True

    def test_dbh_section_chosen_is_closest_to_target(self):
        # Sections at 0.3, 0.5, ..., 1.3 m. Target 1.3 m → index 5.
        sections = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
        a = np.array([[0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]])
        b = a.copy()
        sr = _make_section_result([0], sections, a, b)
        xyz, tree_ids = _synthetic_cloud_for_trees([(0, 0.0, 5.0)])
        tr = _StubTrunkResult(tree_ids=tree_ids)

        m = compute_tree_metrics(xyz, tr, sr)[0]
        assert m.dbh_section_height == 1.3
        assert abs(m.dbh - 0.30) < 1e-9  # 2 · 0.15

    def test_dbh_invalid_when_no_section_near_target(self):
        # All sections too high (lowest at 5.0 m) → no section near 1.30 m.
        sections = np.array([5.0, 6.0, 7.0])
        a = np.array([[0.15, 0.15, 0.15]])
        b = a.copy()
        sr = _make_section_result([0], sections, a, b)
        xyz, tree_ids = _synthetic_cloud_for_trees([(0, 0.0, 10.0)])
        tr = _StubTrunkResult(tree_ids=tree_ids)

        m = compute_tree_metrics(xyz, tr, sr)[0]
        assert m.valid_at_dbh is False
        assert m.dbh == 0.0
        assert m.ovality_at_dbh == 0.0
        # Height is still computed (independent of section fit).
        assert m.height_total == pytest.approx(10.0, abs=0.05)

    def test_dbh_invalid_when_section_at_target_failed(self):
        # Section at 1.3 m exists but its fit failed (R = a = b = 0).
        sections = np.array([0.5, 1.3, 2.1])
        a = np.array([[0.15, 0.00, 0.15]])  # 1.3m section is invalid
        b = a.copy()
        sr = _make_section_result([0], sections, a, b)
        xyz, tree_ids = _synthetic_cloud_for_trees([(0, 0.0, 5.0)])
        tr = _StubTrunkResult(tree_ids=tree_ids)

        m = compute_tree_metrics(xyz, tr, sr)[0]
        assert m.valid_at_dbh is False
        assert m.n_valid_sections == 2  # the other two sections are valid

    def test_height_unaffected_by_section_fit_status(self):
        # Even if EVERY section fit fails, total height should still be
        # computed from the cloud extent.
        sections = np.array([0.5, 1.3, 2.1])
        a = np.zeros((1, 3))  # all invalid
        b = np.zeros((1, 3))
        sr = _make_section_result([0], sections, a, b)
        xyz, tree_ids = _synthetic_cloud_for_trees([(0, 0.1, 18.5)])
        tr = _StubTrunkResult(tree_ids=tree_ids)

        m = compute_tree_metrics(xyz, tr, sr)[0]
        assert m.valid_at_dbh is False
        assert m.height_total == pytest.approx(18.4, abs=0.05)
        assert m.z_base == pytest.approx(0.1, abs=0.05)
        assert m.z_top == pytest.approx(18.5, abs=0.05)
        assert m.n_valid_sections == 0

    def test_multiple_trees_processed_independently(self):
        sections = np.array([0.5, 1.3, 2.1, 3.0])
        # Tree 0: oval at 1.3m. Tree 1: circle at 1.3m. Tree 2: invalid at 1.3m.
        a = np.array([
            [0.10, 0.20, 0.18, 0.16],  # tree 0
            [0.15, 0.15, 0.14, 0.12],  # tree 1
            [0.12, 0.00, 0.10, 0.08],  # tree 2 (invalid at 1.3m)
        ])
        b = np.array([
            [0.10, 0.10, 0.10, 0.10],  # tree 0: ovality 2.0
            [0.15, 0.15, 0.14, 0.12],  # tree 1: ovality 1.0
            [0.12, 0.00, 0.10, 0.08],  # tree 2
        ])
        sr = _make_section_result([0, 1, 2], sections, a, b)
        xyz, tree_ids = _synthetic_cloud_for_trees([
            (0, 0.0, 12.0), (1, 0.0, 18.0), (2, 0.0, 25.0),
        ])
        tr = _StubTrunkResult(tree_ids=tree_ids)

        ms = compute_tree_metrics(xyz, tr, sr)
        assert [m.tree_id for m in ms] == [0, 1, 2]
        # Tree 0: oval, ovality 2.0
        assert ms[0].is_oval_at_dbh is True
        assert abs(ms[0].ovality_at_dbh - 2.0) < 1e-9
        # Tree 1: not oval
        assert ms[1].is_oval_at_dbh is False
        assert abs(ms[1].ovality_at_dbh - 1.0) < 1e-9
        # Tree 2: invalid at DBH but height still computed
        assert ms[2].valid_at_dbh is False
        assert ms[2].height_total == pytest.approx(25.0, abs=0.05)

    def test_zero_trees_returns_empty_list(self):
        # Empty section result → no metrics.
        sr = SectionResult(
            X_c=np.empty((0, 5)), Y_c=np.empty((0, 5)),
            R=np.empty((0, 5)), check=np.empty((0, 5)),
            sector_pct=np.empty((0, 5)), sections=np.linspace(0.3, 1.1, 5),
            tree_ids=[], config=StemCleaningConfig(),
            a=np.empty((0, 5)), b=np.empty((0, 5)), theta=np.empty((0, 5)),
        )
        xyz = np.empty((0, 3))
        tr = _StubTrunkResult(tree_ids=np.empty(0, dtype=np.int32))
        assert compute_tree_metrics(xyz, tr, sr) == []

    def test_works_when_section_result_lacks_ellipse_fields(self):
        # Legacy SectionResult without (a, b, theta) — e.g. an old mock
        # in some other test file — should produce metrics with
        # valid_at_dbh=False (no ellipse info to compute DBH from), but
        # heights still come through.
        sections = np.array([0.5, 1.3, 2.1])
        R = np.array([[0.15, 0.15, 0.14]])
        sr = SectionResult(
            X_c=np.zeros_like(R), Y_c=np.zeros_like(R),
            R=R, check=np.zeros_like(R), sector_pct=np.full_like(R, 100.0),
            sections=sections, tree_ids=[0], config=StemCleaningConfig(),
        )  # a, b, theta default to None
        xyz, tree_ids = _synthetic_cloud_for_trees([(0, 0.0, 15.0)])
        tr = _StubTrunkResult(tree_ids=tree_ids)

        m = compute_tree_metrics(xyz, tr, sr)[0]
        assert m.valid_at_dbh is False
        assert m.height_total == pytest.approx(15.0, abs=0.05)


class TestTreeMetricsToDataframe:

    def test_dataframe_round_trip(self):
        m = TreeMetrics(
            tree_id=7,
            valid_at_dbh=True,
            dbh_section_height=1.3,
            dbh=0.302,
            dbh_major=0.40,
            dbh_minor=0.20,
            ovality_at_dbh=2.0,
            is_oval_at_dbh=True,
            z_base=0.1,
            z_top=18.5,
            height_total=18.4,
            n_points=1234,
            n_valid_sections=27,
        )
        df = tree_metrics_to_dataframe([m])
        assert len(df) == 1
        row = df.iloc[0]
        assert int(row["Tree_ID"]) == 7
        assert bool(row["Valid_at_DBH"]) is True
        assert bool(row["Is_oval_at_DBH"]) is True
        assert float(row["DBH"]) == pytest.approx(0.302, abs=1e-4)
        assert float(row["Height_total"]) == pytest.approx(18.4, abs=1e-3)
