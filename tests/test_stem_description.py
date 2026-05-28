"""Tests for ``src.core.stem_description`` — Coverage sheet + Interpine
HQP Stem_Description schema.

Synthetic-only tests; this module is a pure consumer of upstream
dataclass outputs (``TrunkExtractionResult``, ``SectionResult``,
``TreeMetrics``, per-tree centerlines), so each test wires up a minimal
in-memory stub instead of running the full pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from src.core.dendrometry import TreeMetrics
from src.core.stem_description import (
    CoverageMetrics,
    _max_deviation_from_baseline,
    _polyline_length_z,
    _topmost_valid_section_index,
    build_stem_description_rows,
    classify_sweep,
    compute_coverage_metrics,
    coverage_metrics_to_dataframe,
    stem_description_to_dataframe,
)
from src.core.trunk_validation import SectionResult, StemCleaningConfig


# ===========================================================================
# Stubs and helpers
# ===========================================================================

@dataclass
class _StubTrunkResult:
    """Duck-typed stand-in for TrunkExtractionResult; this module only
    reads ``tree_ids`` and ``tree_axes``."""
    tree_ids: np.ndarray
    tree_axes: list


def _make_section_result(
    tree_ids: List[int],
    sections: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> SectionResult:
    """Build a SectionResult with the ellipse fields populated."""
    R = np.where((a > 0) & (b > 0), np.sqrt(a * b), 0.0)
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


def _make_tree_metrics(
    tree_id: int,
    *,
    height_total: float,
    dbh: float = 0.30,
    z_base: float = 0.0,
    z_top: float = None,
    is_oval_at_dbh: bool = False,
) -> TreeMetrics:
    z_top = z_top if z_top is not None else z_base + height_total
    return TreeMetrics(
        tree_id=tree_id,
        valid_at_dbh=True,
        dbh_section_height=1.30,
        dbh=dbh,
        dbh_major=dbh,
        dbh_minor=dbh,
        ovality_at_dbh=1.0,
        is_oval_at_dbh=is_oval_at_dbh,
        z_base=z_base,
        z_top=z_top,
        height_total=height_total,
        n_points=1000,
        n_valid_sections=10,
    )


# ===========================================================================
# Unit tests — internal helpers
# ===========================================================================

class TestTopmostValidSectionIndex:

    def test_empty_returns_minus_one(self):
        assert _topmost_valid_section_index(np.array([])) == -1

    def test_all_invalid_returns_minus_one(self):
        assert _topmost_valid_section_index(np.array([0.0, 0.0, 0.0])) == -1

    def test_mixed_returns_highest_valid_index(self):
        # Valid at index 1, 3, 4 (4 is the topmost).
        assert _topmost_valid_section_index(
            np.array([0.0, 0.15, 0.0, 0.20, 0.18, 0.0])
        ) == 4

    def test_all_valid_returns_last(self):
        assert _topmost_valid_section_index(np.array([0.10, 0.12, 0.14])) == 2


class TestMaxDeviationFromBaseline:

    def test_too_few_points_returns_zero(self):
        # 2 points cannot have any deviation by definition.
        assert _max_deviation_from_baseline(np.zeros((2, 3))) == 0.0

    def test_perfectly_straight_polyline_zero_deviation(self):
        # Three points on the same z axis → all on the base-top line.
        cl = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ])
        assert _max_deviation_from_baseline(cl) == pytest.approx(0.0, abs=1e-12)

    def test_bow_deviation_matches_geometry(self):
        # Vertical axis from (0, 0, 0) to (0, 0, 2). A middle node at
        # (0.5, 0, 1) sits 0.5 m perpendicular to that axis.
        cl = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ])
        assert _max_deviation_from_baseline(cl) == pytest.approx(0.5, abs=1e-9)

    def test_invariant_under_axis_swap(self):
        # Same bow but oriented along Y instead of X — still 0.5 m amplitude.
        cl = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.0, 2.0],
        ])
        assert _max_deviation_from_baseline(cl) == pytest.approx(0.5, abs=1e-9)


class TestPolylineLengthZ:

    def test_z_extent(self):
        cl = np.array([
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 3.5],
            [0.0, 0.0, 8.5],
        ])
        assert _polyline_length_z(cl) == pytest.approx(7.0)


# ===========================================================================
# Unit tests — classify_sweep
# ===========================================================================

class TestClassifySweep:

    @staticmethod
    def _bow_centerline(amplitude_m: float, z_top_m: float = 10.0):
        """Three-node polyline: base at origin, top at (0, 0, z_top), middle
        node displaced by ``amplitude_m`` in +x at z = z_top / 2."""
        return np.array([
            [0.0, 0.0, 0.0],
            [amplitude_m, 0.0, z_top_m / 2.0],
            [0.0, 0.0, z_top_m],
        ])

    def test_none_centerline_returns_none(self):
        assert classify_sweep(None, sed_obs_m=0.20) is None

    def test_zero_sed_returns_none(self):
        cl = self._bow_centerline(0.01)
        assert classify_sweep(cl, sed_obs_m=0.0) is None

    def test_too_few_nodes_returns_none(self):
        cl = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        assert classify_sweep(cl, sed_obs_m=0.20) is None

    def test_gun_barrel_8_for_tiny_amplitude(self):
        # SED 0.20 m → SED/8 = 0.025 m. Amplitude 0.01 m → ratio = 0.05 < 1/8.
        cl = self._bow_centerline(0.01)
        assert classify_sweep(cl, sed_obs_m=0.20) == "8"

    def test_long_log_L_for_consistent_sweep_above_threshold(self):
        # SED 0.20 → SED/5 = 0.04. Amplitude 0.035 → ratio 0.175, between
        # 1/8 (0.125) and 1/5 (0.20). Length 10 m ≥ 6.1 m → "L".
        cl = self._bow_centerline(0.035, z_top_m=10.0)
        assert classify_sweep(cl, sed_obs_m=0.20) == "L"

    def test_short_log_S_when_below_log_length_threshold(self):
        # Same amplitude/SED but z_top 5 m < 6.1 → "S".
        cl = self._bow_centerline(0.035, z_top_m=5.0)
        assert classify_sweep(cl, sed_obs_m=0.20) == "S"

    def test_moderate_sweep_3(self):
        # SED 0.20 → SED/3 ≈ 0.0667. Amplitude 0.05 → ratio 0.25, in
        # (1/5, 1/3] → "3".
        cl = self._bow_centerline(0.05)
        assert classify_sweep(cl, sed_obs_m=0.20) == "3"

    def test_excessive_sweep_1(self):
        # SED 0.20 → SED/1 = 0.20. Amplitude 0.10 → ratio 0.5, in (1/3, 1] → "1".
        cl = self._bow_centerline(0.10)
        assert classify_sweep(cl, sed_obs_m=0.20) == "1"

    def test_severe_sweep_X_for_amplitude_above_sed(self):
        # Amplitude 0.30 > SED 0.20 → ratio 1.5 → "X".
        cl = self._bow_centerline(0.30)
        assert classify_sweep(cl, sed_obs_m=0.20) == "X"


# ===========================================================================
# Integration tests — compute_coverage_metrics
# ===========================================================================

class TestComputeCoverageMetrics:

    def test_basic_single_tree_with_canopy(self):
        # Tree 0: stem labelled to z ∈ [0.1, 18.0], but canopy extends to
        # z = 22.0 with points in a 1 m radius column around the basal
        # centroid (5.0, 5.0). HT_cloud should be 22.0.
        sections = np.arange(0.3, 18.0, 0.2)
        n_secs = len(sections)
        a = np.full((1, n_secs), 0.15)
        b = np.full((1, n_secs), 0.15)
        sr = _make_section_result([0], sections, a, b)

        # Cloud: 1000 stem points around (5, 5) up to z=18; 500 canopy
        # points around (5, 5) at z up to 22.
        rng = np.random.default_rng(seed=0)
        stem_xy = rng.uniform(-0.2, 0.2, size=(1000, 2)) + np.array([5.0, 5.0])
        stem_z = rng.uniform(0.1, 18.0, size=1000)
        canopy_xy = rng.uniform(-0.8, 0.8, size=(500, 2)) + np.array([5.0, 5.0])
        canopy_z = rng.uniform(18.0, 22.0, size=500)
        xyz = np.vstack([
            np.column_stack([stem_xy[:, 0], stem_xy[:, 1], stem_z]),
            np.column_stack([canopy_xy[:, 0], canopy_xy[:, 1], canopy_z]),
        ])
        # tree_ids: stem points labelled 0, canopy points unlabelled (-1)
        tree_ids = np.concatenate([
            np.full(1000, 0, dtype=np.int32),
            np.full(500, -1, dtype=np.int32),
        ])

        trunk_result = _StubTrunkResult(
            tree_ids=tree_ids,
            tree_axes=[{"tree_id": 0, "centroid": (5.0, 5.0)}],
        )

        tm = _make_tree_metrics(
            tree_id=0,
            height_total=17.9,  # 18.0 - 0.1
            dbh=0.30,
            z_base=0.1,
            z_top=18.0,
        )

        cov = compute_coverage_metrics(
            xyz, trunk_result, sr, [tm], crown_buffer_radius=2.5,
        )
        assert len(cov) == 1
        c = cov[0]
        assert c.tree_id == 0
        # HT_cloud should reach the canopy region (z up to 22).
        assert c.ht_cloud_m == pytest.approx(22.0, abs=0.1)
        assert c.ht_stem_m == pytest.approx(17.9)
        # RH ≈ 17.9 / 22 ≈ 0.81
        assert c.rh_obs == pytest.approx(17.9 / 22.0, abs=0.01)
        assert c.dbh_cm == pytest.approx(30.0)
        # SED at topmost valid section: a = b = 0.15 → diameter 0.30 m = 30 cm.
        assert c.valid_sed is True
        assert c.sed_obs_cm == pytest.approx(30.0)

    def test_no_valid_section_sets_valid_sed_false(self):
        # All sections invalid (a = b = 0) → no SED to report.
        sections = np.arange(0.3, 5.0, 0.2)
        n_secs = len(sections)
        a = np.zeros((1, n_secs))
        b = np.zeros((1, n_secs))
        sr = _make_section_result([0], sections, a, b)

        xyz = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])
        tree_ids = np.array([0, 0], dtype=np.int32)
        trunk_result = _StubTrunkResult(
            tree_ids=tree_ids,
            tree_axes=[{"tree_id": 0, "centroid": (0.0, 0.0)}],
        )
        tm = _make_tree_metrics(tree_id=0, height_total=4.0, z_top=5.0)

        cov = compute_coverage_metrics(xyz, trunk_result, sr, [tm])
        assert cov[0].valid_sed is False
        assert cov[0].sed_obs_cm == 0.0


# ===========================================================================
# Integration tests — build_stem_description_rows
# ===========================================================================

class TestBuildStemDescriptionRows:

    def test_per_section_position_rows_and_base_marker(self):
        # Single tree with 3 valid sections at z = 0.3, 1.3, 2.3
        # plus 1 invalid section at z = 3.3.
        sections = np.array([0.3, 1.3, 2.3, 3.3])
        a = np.array([[0.15, 0.14, 0.13, 0.0]])  # 0.0 marks the invalid one
        b = np.array([[0.15, 0.14, 0.13, 0.0]])
        sr = _make_section_result([42], sections, a, b)

        tm = _make_tree_metrics(tree_id=42, height_total=2.5)
        cov = CoverageMetrics(
            tree_id=42,
            ht_cloud_m=3.0, ht_stem_m=2.5, rh_obs=0.83,
            dbh_cm=28.0, sed_obs_cm=26.0,
            sed_obs_height_m=2.3, valid_sed=True,
        )
        # Centerline: 3 nodes, perfectly straight → sweep "8".
        cl = np.array([
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 1.3],
            [0.0, 0.0, 2.3],
        ])

        rows = build_stem_description_rows(
            plot_id="T_TEST",
            tree_metrics=[tm],
            coverage_metrics=[cov],
            section_result=sr,
            tree_centerlines=[cl],
        )

        # Expected rows: 1 base marker + 3 position rows + 1 sweep row.
        assert len(rows) == 5

        # Base marker
        assert rows[0]["Position"] == 0.0
        assert np.isnan(rows[0]["Diameter"])
        assert rows[0]["PlotId"] == "T_TEST"
        assert rows[0]["TreeNumber"] == 42
        assert rows[0]["StemNo"] == 0
        assert rows[0]["Level"] == 0

        # Position rows: 3 valid sections, diameters in mm
        position_rows = [r for r in rows if not np.isnan(r["Position"])][1:]
        diameters = sorted(r["Diameter"] for r in position_rows)
        # Sections 0.3, 1.3, 2.3 with a=b=0.15, 0.14, 0.13 → diameters 300, 280, 260 mm
        assert diameters == pytest.approx([260.0, 280.0, 300.0], abs=0.1)

        # Sweep row
        sw_rows = [r for r in rows if not isinstance(r["Sw"], float)
                   or not np.isnan(r["Sw"])]
        assert len(sw_rows) == 1
        assert sw_rows[0]["Sw"] == "8"
        assert np.isnan(sw_rows[0]["Position"])
        assert np.isnan(sw_rows[0]["Diameter"])

    def test_oval_tree_emits_ovality_feature_row(self):
        sections = np.array([1.3])
        a = np.array([[0.20]])
        b = np.array([[0.20]])  # circle-style; oval flag comes from tree_metrics
        sr = _make_section_result([1], sections, a, b)

        tm = _make_tree_metrics(
            tree_id=1, height_total=10.0, is_oval_at_dbh=True,
        )
        cov = CoverageMetrics(
            tree_id=1, ht_cloud_m=12.0, ht_stem_m=10.0, rh_obs=0.83,
            dbh_cm=40.0, sed_obs_cm=40.0, sed_obs_height_m=1.3, valid_sed=True,
        )
        # No centerline (so no sweep row)
        rows = build_stem_description_rows(
            plot_id="T_OVAL",
            tree_metrics=[tm],
            coverage_metrics=[cov],
            section_result=sr,
            tree_centerlines=[None],
        )
        f_rows = [r for r in rows
                  if not (isinstance(r["F"], float) and np.isnan(r["F"]))]
        assert len(f_rows) == 1
        assert f_rows[0]["F"] == "O1.2+"


class TestDataframeConverters:

    def test_coverage_dataframe_columns_in_expected_order(self):
        c = CoverageMetrics(
            tree_id=7,
            ht_cloud_m=20.0, ht_stem_m=15.0, rh_obs=0.75,
            dbh_cm=32.0, sed_obs_cm=12.0, sed_obs_height_m=14.5,
            valid_sed=True,
        )
        df = coverage_metrics_to_dataframe([c])
        assert list(df.columns) == [
            "Tree_ID", "HT_cloud_m", "HT_stem_m", "RH_obs", "DBH_cm",
            "SED_obs_cm", "SED_obs_height_m",
        ]
        assert df.iloc[0]["Tree_ID"] == 7
        assert df.iloc[0]["RH_obs"] == 0.75

    def test_stem_description_dataframe_uses_interpine_column_order(self):
        rows = [{
            "PlotId": "P1", "TreeNumber": 1, "StemNo": 0, "Level": 0,
            "Position": 1.3, "Diameter": 300.0,
            "Br": float("nan"), "Sw": float("nan"), "F": float("nan"),
        }]
        df = stem_description_to_dataframe(rows)
        assert list(df.columns) == [
            "PlotId", "TreeNumber", "StemNo", "Level",
            "Position", "Diameter", "Br", "Sw", "F",
        ]
