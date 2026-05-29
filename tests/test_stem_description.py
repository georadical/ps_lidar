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
    build_taper_rows,
    classify_sweep,
    classify_sweep_zones,
    compute_coverage_metrics,
    coverage_metrics_to_dataframe,
    stem_description_to_dataframe,
    taper_to_dataframe,
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

class TestClassifySweepZones:
    """Zone-aware sliding-window classifier (F1.1)."""

    @staticmethod
    def _straight_polyline(z_top: float = 10.0, n: int = 21) -> np.ndarray:
        z = np.linspace(0.0, z_top, n)
        return np.column_stack([np.zeros(n), np.zeros(n), z])

    def test_empty_inputs_return_empty(self):
        assert classify_sweep_zones(None, 0.2) == []
        assert classify_sweep_zones(self._straight_polyline(), 0.0) == []
        assert classify_sweep_zones(np.zeros((2, 3)), 0.2) == []

    def test_straight_polyline_single_8_zone(self):
        cl = self._straight_polyline(z_top=10.0, n=21)
        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        assert len(zones) == 1
        assert zones[0][2] == "8"
        # Spans the whole polyline
        assert zones[0][0] == pytest.approx(0.0)
        assert zones[0][1] == pytest.approx(10.0)

    def test_localised_bow_creates_worse_central_zone(self):
        # Polyline ~21 m long, endpoints on the stem axis (x=0), with a
        # localised bow at z ∈ [10, 14] where x drifts to 0.2 m.
        # Distance to base-top chord: 0 outside the bow, 0.2 inside.
        # ratio = 0.2 / SED 0.2 = 1.0 → "1" inside, "8" outside.
        z = np.linspace(0.0, 21.0, 106)  # 0.2 m spacing
        x = np.zeros_like(z)
        bow_mask = (z >= 10.0) & (z <= 14.0)
        x[bow_mask] = 0.20
        cl = np.column_stack([x, np.zeros_like(z), z])

        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        codes = [c for (_, _, c) in zones]
        assert "1" in codes
        assert "8" in codes
        # First zone covers the lower clean section
        assert zones[0][0] == pytest.approx(0.0, abs=0.5)
        assert zones[0][2] == "8"
        # Last zone covers the upper clean section
        assert zones[-1][1] == pytest.approx(21.0, abs=0.5)
        assert zones[-1][2] == "8"

    def test_min_length_absorbs_short_better_zone_between_worse(self):
        # F1.3: a 1 m "8" gap between two "1" bows is below its 4 m min
        # and absorbs into the surrounding "1". Boundary "8" zones in
        # this layout are kept long enough (5 m each) to exceed the
        # "8" minimum and survive.
        z = np.linspace(0.0, 18.0, 181)  # 0.1 m spacing
        x = np.zeros_like(z)
        x[(z >= 5.0) & (z < 8.0)] = 0.20   # first bow → "1" (3 m)
        x[(z > 9.0) & (z <= 13.0)] = 0.20  # second bow → "1" (4 m)
        cl = np.column_stack([x, np.zeros_like(z), z])

        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        codes = [c for (_, _, c) in zones]
        # Central "8" gap (1 m) absorbed into "1"; boundary "8" zones
        # (5 m each) exceed the 4 m min and survive.
        assert codes == ["8", "1", "8"]
        assert zones[1][2] == "1"
        # Central "1" zone now spans both bows + the absorbed gap
        assert zones[1][0] == pytest.approx(5.0, abs=0.2)
        assert zones[1][1] == pytest.approx(13.0, abs=0.2)

    def test_short_sed5_zone_reclassified_to_S(self):
        # SED/5 amplitude (ratio 0.175) over a 4 m plateau → "S"
        # (4 m < l_min_length_m=5 m, so not "L").
        z = np.linspace(0.0, 6.0, 121)  # 0.05 m spacing
        x = np.where((z >= 1.0) & (z <= 5.0), 0.035, 0.0)  # 0.035/0.20 = 0.175
        cl = np.column_stack([x, np.zeros_like(z), z])
        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        codes = [c for (_, _, c) in zones]
        assert "S" in codes
        assert "L" not in codes

    def test_long_sed5_zone_is_L(self):
        # Same SED/5 amplitude but a 6 m plateau (≥ 5 m) → "L".
        z = np.linspace(0.0, 8.0, 161)  # 0.05 m spacing
        x = np.where((z >= 1.0) & (z <= 7.0), 0.035, 0.0)
        cl = np.column_stack([x, np.zeros_like(z), z])
        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        codes = [c for (_, _, c) in zones]
        assert "L" in codes
        assert "S" not in codes

    def test_short_worse_zone_absorbed_symmetrically(self):
        # F1.3: a 2 m "3" between two ample "8" zones (5 m each, above
        # the 4 m min) is below its 3 m min and absorbs into the
        # surrounding "8", collapsing to a single "8" over the stem.
        # Supersedes the F1.2 asymmetric preserve-defects behaviour.
        z = np.linspace(0.0, 12.0, 241)  # 0.05 m spacing
        x = np.where((z >= 5.0) & (z <= 7.0), 0.05, 0.0)  # 0.05/0.20 = 0.25 → "3"
        cl = np.column_stack([x, np.zeros_like(z), z])
        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        codes = [c for (_, _, c) in zones]
        assert codes == ["8"]
        assert zones[0][0] == pytest.approx(0.0, abs=0.1)
        assert zones[0][1] == pytest.approx(12.0, abs=0.1)

    def test_short_zone_at_boundary_absorbed(self):
        # F1.3: a short zone at the polyline boundary (single neighbour)
        # is absorbed into that neighbour. Polyline has tiny "8" caps
        # at base and top with a long L plateau in between; the caps
        # should disappear, leaving a single L zone over the polyline.
        z = np.linspace(0.0, 8.0, 161)
        x = np.where((z >= 1.0) & (z <= 7.0), 0.040, 0.0)  # 0.04/0.20 = 0.20 → "LS" → "L" (6 m)
        cl = np.column_stack([x, np.zeros_like(z), z])
        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        codes = [c for (_, _, c) in zones]
        # Boundary "8" caps (1 m each, < 4 m min, single neighbour L)
        # absorbed → single L zone over the whole polyline.
        assert codes == ["L"]
        assert zones[0][0] == pytest.approx(0.0, abs=0.1)
        assert zones[0][1] == pytest.approx(8.0, abs=0.1)

    def test_short_X_zone_preserved(self):
        # X stays exempt from the length floor — its 0.3-1 m
        # short-severe definition is intrinsic.
        z = np.linspace(0.0, 10.0, 201)
        x = np.where((z >= 4.75) & (z <= 5.25), 0.25, 0.0)  # 0.25/0.20 = 1.25 → "X"
        cl = np.column_stack([x, np.zeros_like(z), z])
        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        codes = [c for (_, _, c) in zones]
        assert "X" in codes

    def test_zones_cover_polyline_contiguously(self):
        cl = self._straight_polyline(z_top=15.0, n=76)
        zones = classify_sweep_zones(cl, sed_obs_m=0.20)
        # No gaps between zones
        for k in range(len(zones) - 1):
            assert zones[k][1] == pytest.approx(zones[k + 1][0], abs=0.3)


class TestBuildStemDescriptionRows:
    """Sparse-row schema: base + DBH + end-of-zone Sw + optional F.
    Dense diameter samples live on the separate Taper sheet."""

    def test_sparse_rows_base_dbh_and_endofzone_sweep(self):
        # Single tree with valid sections at z = 0.3, 1.3, 2.3 (Δz_top=2.3).
        sections = np.array([0.3, 1.3, 2.3, 3.3])
        a = np.array([[0.15, 0.14, 0.13, 0.0]])  # 0.0 marks invalid
        b = np.array([[0.15, 0.14, 0.13, 0.0]])
        sr = _make_section_result([42], sections, a, b)

        tm = _make_tree_metrics(tree_id=42, height_total=2.5, dbh=0.28)
        cov = CoverageMetrics(
            tree_id=42,
            ht_cloud_m=3.0, ht_stem_m=2.5, rh_obs=0.83,
            dbh_cm=28.0, sed_obs_cm=26.0,
            sed_obs_height_m=2.3, valid_sed=True,
        )
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

        # Expected: base + DBH + Sw end-of-zone = 3 rows (no F since not oval).
        assert len(rows) == 3
        assert all(r["PlotId"] == "T_TEST" for r in rows)
        assert all(r["TreeNumber"] == 42 for r in rows)

        # Base marker
        base = rows[0]
        assert base["Position"] == 0.0
        assert np.isnan(base["Diameter"])

        # DBH row
        dbh_row = rows[1]
        assert dbh_row["Position"] == pytest.approx(1.30)
        assert dbh_row["Diameter"] == pytest.approx(280.0, abs=0.1)
        assert np.isnan(dbh_row["Sw"]) and np.isnan(dbh_row["F"])

        # End-of-zone Sw row
        sw_row = rows[2]
        assert sw_row["Position"] == pytest.approx(2.3)
        assert sw_row["Sw"] == "8"
        assert np.isnan(sw_row["Diameter"])

    def test_oval_tree_emits_ovality_feature_row(self):
        sections = np.array([1.3])
        a = np.array([[0.20]])
        b = np.array([[0.20]])
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

    def test_invalid_dbh_skips_dbh_row(self):
        sections = np.array([2.3, 3.3])
        a = np.array([[0.18, 0.17]])
        b = np.array([[0.18, 0.17]])
        sr = _make_section_result([2], sections, a, b)

        tm = _make_tree_metrics(tree_id=2, height_total=3.0)
        tm.valid_at_dbh = False
        tm.dbh = 0.0
        cov = CoverageMetrics(
            tree_id=2, ht_cloud_m=5.0, ht_stem_m=3.0, rh_obs=0.60,
            dbh_cm=0.0, sed_obs_cm=34.0, sed_obs_height_m=3.3, valid_sed=True,
        )
        cl = np.array([
            [0.0, 0.0, 2.3],
            [0.0, 0.0, 2.8],
            [0.0, 0.0, 3.3],
        ])
        rows = build_stem_description_rows(
            plot_id="P", tree_metrics=[tm], coverage_metrics=[cov],
            section_result=sr, tree_centerlines=[cl],
        )
        # base + Sw end-of-zone (no DBH row, no F)
        assert len(rows) == 2
        positions = [r["Position"] for r in rows]
        assert positions[0] == 0.0
        assert positions[1] == pytest.approx(3.3)


class TestBuildTaperRows:

    def test_targets_dbh_then_integer_meters(self):
        # Sections every 0.2 m from 0.3 to 5.3 m. Topmost valid at z=5.3.
        sections = np.round(np.arange(0.3, 5.31, 0.2), 1)
        n = len(sections)
        a = np.full((1, n), 0.20)
        b = np.full((1, n), 0.20)
        sr = _make_section_result([7], sections, a, b)

        tm = _make_tree_metrics(tree_id=7, height_total=5.0, z_top=5.3)
        rows = build_taper_rows([tm], sr, z_start=1.30, z_step=1.0)

        positions = [r["Position_m"] for r in rows]
        # Expected: 1.3, 2.0, 3.0, 4.0, 5.0
        assert positions == pytest.approx([1.3, 2.0, 3.0, 4.0, 5.0])
        # All diameters from a=b=0.20 → 400 mm
        assert all(r["Diameter_mm"] == pytest.approx(400.0, abs=0.1) for r in rows)
        assert all(r["Tree_ID"] == 7 for r in rows)

    def test_skips_when_no_valid_section_near_target(self):
        # Only one valid section at 1.3; all others invalid.
        sections = np.array([0.3, 1.3, 2.3, 3.3])
        a = np.array([[0.0, 0.20, 0.0, 0.0]])
        b = np.array([[0.0, 0.20, 0.0, 0.0]])
        sr = _make_section_result([3], sections, a, b)
        tm = _make_tree_metrics(tree_id=3, height_total=3.0, z_top=3.3)
        rows = build_taper_rows([tm], sr, z_start=1.30, z_step=1.0)
        # Only the DBH target (1.3) finds a valid section.
        assert len(rows) == 1
        assert rows[0]["Position_m"] == pytest.approx(1.3)

    def test_no_valid_section_emits_no_rows(self):
        sections = np.array([0.3, 1.3, 2.3])
        a = np.zeros((1, 3))
        b = np.zeros((1, 3))
        sr = _make_section_result([4], sections, a, b)
        tm = _make_tree_metrics(tree_id=4, height_total=2.0, z_top=2.3)
        assert build_taper_rows([tm], sr) == []

    def test_z_step_2m(self):
        # z_step=2 → targets 1.3, 2.0, 4.0, 6.0 ...
        sections = np.round(np.arange(0.3, 7.51, 0.2), 1)
        n = len(sections)
        a = np.full((1, n), 0.18)
        b = np.full((1, n), 0.18)
        sr = _make_section_result([8], sections, a, b)
        tm = _make_tree_metrics(tree_id=8, height_total=7.0, z_top=7.3)
        rows = build_taper_rows([tm], sr, z_start=1.30, z_step=2.0)
        positions = [r["Position_m"] for r in rows]
        assert positions == pytest.approx([1.3, 2.0, 4.0, 6.0])


class TestTaperDataframe:

    def test_columns_in_canonical_order(self):
        rows = [
            {"Tree_ID": 1, "Position_m": 1.3, "Diameter_mm": 300.0},
            {"Tree_ID": 1, "Position_m": 2.0, "Diameter_mm": 290.0},
        ]
        df = taper_to_dataframe(rows)
        assert list(df.columns) == ["Tree_ID", "Position_m", "Diameter_mm"]
        assert len(df) == 2


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
