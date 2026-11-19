"""Tests for ``src.core.sweep_windowed`` — window-based sweep zones (F2).

Synthetic stems isolate the cases in the PlotSafe procedures manual
(Feb 2007, §4.1.4.2): straight, uniform sweep, kink, wobble, a stem that
ends before the assessment length (footnote 6), butt flare, the 150 mm
top, SED taken at each window's top, and the HQP upgrade rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.stem_description import (
    CoverageMetrics,
    build_stem_description_rows,
    classify_sweep_zones,
)
from src.core.sweep_windowed import (
    HQP_QUICKCARD,
    PLOTSAFE_RAD05,
    SEVERITY,
    WindowedSweepConfig,
    _apply_upgrade_rule,
    _isotonic_nonincreasing,
    classify_sweep_zones_windowed,
)
from tests.test_stem_description import _make_section_result, _make_tree_metrics

STEP = 0.2


def _z(z0: float = 0.3, z1: float = 20.3) -> np.ndarray:
    return np.round(np.arange(z0, z1 + 1e-9, STEP), 3)


def _taper(z: np.ndarray, d_base: float = 0.30, d_top: float = 0.12) -> np.ndarray:
    return d_base + (d_top - d_base) * (z - z[0]) / (z[-1] - z[0])


def _stem(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.zeros_like(z), z])


def _codes(zones):
    return [c for (_, _, c) in zones]


def _assert_contiguous(zones, z0, z1):
    assert zones[0][0] == pytest.approx(z0)
    assert zones[-1][1] == pytest.approx(z1)
    for (_, b, _), (a, _, _) in zip(zones[:-1], zones[1:]):
        assert a == pytest.approx(b)


class TestInputs:

    def test_empty_or_degenerate_inputs_return_empty(self):
        z = _z()
        cl = _stem(np.zeros_like(z), z)
        assert classify_sweep_zones_windowed(None, (z, _taper(z))) == []
        assert classify_sweep_zones_windowed(cl[:2], (z, _taper(z))) == []
        assert classify_sweep_zones_windowed(cl) == []                       # no diameter
        assert classify_sweep_zones_windowed(cl, sed_m=0.0) == []

    def test_constant_sed_fallback(self):
        z = _z()
        zones = classify_sweep_zones_windowed(_stem(np.zeros_like(z), z), sed_m=0.25)
        assert _codes(zones) == ["8"]

    def test_isotonic_profile_is_non_increasing(self):
        y = np.array([0.30, 0.31, 0.28, 0.29, 0.25, 0.26, 0.20])
        fit = _isotonic_nonincreasing(y)
        assert np.all(np.diff(fit) <= 1e-12)
        assert fit.mean() == pytest.approx(y.mean())


class TestStraightAndSweep:

    def test_noisy_straight_stem_is_single_8(self):
        z = _z()
        x = np.random.default_rng(0).normal(0.0, 0.005, z.size)
        zones = classify_sweep_zones_windowed(_stem(x, z), (z, _taper(z)))
        assert _codes(zones) == ["8"]
        _assert_contiguous(zones, z[0], z[-1])

    def test_uniform_gentle_sweep_is_not_penalised_by_stem_length(self):
        # R = 200 m over 26 m: every 6 m window has ≈ 2.3 cm sagitta (an
        # "8" at the butt), but the whole-stem chord sagitta is ≈ 42 cm.
        z = _z(0.3, 26.3)
        x = (z - z.mean()) ** 2 / (2 * 200.0)
        cl, prof = _stem(x, z), (z, _taper(z, 0.32, 0.12))
        old = classify_sweep_zones(cl, sed_obs_m=0.12)
        new = classify_sweep_zones_windowed(cl, prof)
        assert "X" in _codes(old)
        assert set(_codes(new)) <= {"8", "L"}
        assert new[0][2] == "8"

    def test_sed_is_taken_at_each_window_top(self):
        # Same R = 150 m sweep (6 m sagitta 3 cm): 8 with a thick stem,
        # L with a thin one — the code follows the diameter, not a
        # per-tree scalar.
        z = _z()
        cl = _stem((z - z.mean()) ** 2 / (2 * 150.0), z)
        assert _codes(classify_sweep_zones_windowed(cl, sed_m=0.35)) == ["8"]
        assert _codes(classify_sweep_zones_windowed(cl, sed_m=0.18)) == ["L"]

    def test_long_log_test_separates_L_from_S_in_rad05(self):
        # R = 100 m, D = 0.30: SED/4 over 6.1 m passes, SED/2 over 12 m
        # fails → RAD05 "S"; the HQP quickcard (SED/5 over 6 m) says "L".
        z = _z()
        cl = _stem((z - z.mean()) ** 2 / (2 * 100.0), z)
        hqp = classify_sweep_zones_windowed(cl, sed_m=0.30)
        rad = classify_sweep_zones_windowed(
            cl, sed_m=0.30, config=WindowedSweepConfig(dictionary=PLOTSAFE_RAD05))
        assert _codes(hqp) == ["L"]
        assert _codes(rad) == ["S"]


class TestDefects:

    def test_sharp_kink_becomes_short_K_between_straight_zones(self):
        z = _z()
        x = np.where(z > 10.0, (z - 10.0) * np.tan(np.radians(6.0)), 0.0)
        zones = classify_sweep_zones_windowed(_stem(x, z), (z, _taper(z)))
        assert _codes(zones) == ["8", "K", "8"]
        k0, k1, _ = zones[1]
        assert k1 - k0 <= HQP_QUICKCARD.k_max_length_m + 1e-9
        assert abs(0.5 * (k0 + k1) - 10.0) <= 0.3

    def test_wobble_becomes_bounded_W(self):
        z = _z()
        x = np.where((z > 8.0) & (z < 11.0), 0.06 * np.sin(2 * np.pi * (z - 8.0) / 3.0), 0.0)
        zones = classify_sweep_zones_windowed(_stem(x, z), (z, _taper(z)))
        assert _codes(zones) == ["8", "W", "8"]
        w0, w1, _ = zones[1]
        assert w0 == pytest.approx(8.0, abs=0.5)
        assert w1 == pytest.approx(11.0, abs=0.5)
        assert w1 - w0 <= HQP_QUICKCARD.w_max_length_m

    @pytest.mark.parametrize("dictionary", [HQP_QUICKCARD, PLOTSAFE_RAD05])
    def test_single_local_bow_is_not_wobble(self, dictionary):
        z = _z()
        x = np.where((z > 9.0) & (z < 13.0), 0.08 * np.sin(np.pi * (z - 9.0) / 4.0), 0.0)
        zones = classify_sweep_zones_windowed(
            _stem(x, z), (z, _taper(z)), config=WindowedSweepConfig(dictionary=dictionary))
        assert "W" not in _codes(zones)
        assert zones[0][2] == "8" and zones[-1][2] == "8"


class TestDomain:

    def test_stem_shorter_than_window_keeps_code_8(self):
        # Manual footnote 6: 3 m straight under a fork is still "8".
        z = _z(0.3, 3.3)
        zones = classify_sweep_zones_windowed(_stem(np.zeros_like(z), z), sed_m=0.30)
        assert _codes(zones) == ["8"]

    def test_butt_flare_below_exclusion_is_ignored(self):
        z = _z(0.1, 12.1)
        x = np.zeros_like(z)
        x[0] = 0.10                          # 10 cm swelling at 0.1 m
        zones = classify_sweep_zones_windowed(_stem(x, z), sed_m=0.25)
        assert _codes(zones) == ["8"]
        assert zones[0][0] == pytest.approx(0.1)

    def test_code_frozen_above_150mm(self):
        # 20° kink at 18 m where D ≈ 0.10 m: above the 150 mm point the
        # code does not change, so no K is reported.
        z = _z()
        x = np.where(z > 18.0, (z - 18.0) * np.tan(np.radians(20.0)), 0.0)
        zones = classify_sweep_zones_windowed(_stem(x, z), (z, _taper(z, 0.30, 0.08)))
        assert "K" not in _codes(zones)
        assert zones[-1][1] == pytest.approx(z[-1])
        # with the freeze disabled the kink is reported
        free = classify_sweep_zones_windowed(
            _stem(x, z), (z, _taper(z, 0.30, 0.08)),
            config=WindowedSweepConfig(min_diameter_m=0.0))
        assert "K" in _codes(free)


class TestUpgradeRule:

    def test_short_better_zone_between_worse_is_downgraded(self):
        z = np.round(np.arange(0.0, 12.0, STEP), 3)
        labels = ["S"] * 20 + ["8"] * 10 + ["S"] * (z.size - 30)   # 8 zone = 2 m
        assert set(_apply_upgrade_rule(z, labels, 3.0)) == {"S"}

    def test_long_better_zone_is_kept(self):
        z = np.round(np.arange(0.0, 14.0, STEP), 3)
        labels = ["S"] * 20 + ["8"] * 20 + ["S"] * (z.size - 40)   # 8 zone = 4 m
        assert _apply_upgrade_rule(z, labels, 3.0) == labels

    def test_point_defect_does_not_force_downgrade(self):
        z = np.round(np.arange(0.0, 12.0, STEP), 3)
        labels = ["8"] * 25 + ["K"] + ["8"] * (z.size - 26)
        assert _apply_upgrade_rule(z, labels, 3.0) == labels

    def test_severity_order_matches_quickcard(self):
        order = sorted(SEVERITY, key=SEVERITY.get)
        assert order.index("8") < order.index("L") < order.index("S") < order.index("3")
        assert SEVERITY["W"] == SEVERITY["K"] < SEVERITY["1"] < SEVERITY["X"]


class TestStemDescriptionWiring:

    @staticmethod
    def _inputs():
        sections = _z(0.3, 12.3)
        d = _taper(sections, 0.30, 0.20)
        a = (d / 2.0)[None, :]
        sr = _make_section_result([7], sections, a, a.copy())
        tm = _make_tree_metrics(7, height_total=12.0)
        cov = CoverageMetrics(
            tree_id=7, ht_cloud_m=14.0, ht_stem_m=12.0, rh_obs=0.86,
            dbh_cm=30.0, sed_obs_cm=20.0, sed_obs_height_m=12.3, valid_sed=True,
        )
        x = np.where(sections > 6.0, (sections - 6.0) * np.tan(np.radians(6.0)), 0.0)
        return sr, tm, cov, _stem(x, sections)

    def test_windowed_method_emits_zone_rows(self):
        sr, tm, cov, cl = self._inputs()
        rows = build_stem_description_rows(
            "P", [tm], [cov], sr, [cl], sweep_method="windowed")
        sw = [(r["Position"], r["Sw"]) for r in rows if isinstance(r["Sw"], str)]
        assert [c for _, c in sw] == ["8", "K", "8"]
        assert sw[-1][0] == pytest.approx(12.3)

    def test_default_method_unchanged(self):
        sr, tm, cov, cl = self._inputs()
        default = build_stem_description_rows("P", [tm], [cov], sr, [cl])
        explicit = build_stem_description_rows(
            "P", [tm], [cov], sr, [cl], sweep_method="global_chord")
        assert [r["Sw"] for r in default if isinstance(r["Sw"], str)] == \
               [r["Sw"] for r in explicit if isinstance(r["Sw"], str)]

    def test_unknown_method_raises(self):
        sr, tm, cov, cl = self._inputs()
        with pytest.raises(ValueError, match="sweep_method"):
            build_stem_description_rows("P", [tm], [cov], sr, [cl], sweep_method="nope")
