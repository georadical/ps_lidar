"""Unit tests for the ellipse-fitting primitives in ``src.core.ellipse_fitting``.

Sub-fase EL.1 — gate visual / numérico:
    Fit to synthetic noise-free ellipses must recover the input parameters
    to better than 1e-8 in centre, semi-axes and rotation angle (the gate
    threshold proposed in the Phase 1B/1C plan revised 2026-05-15).

This file grows with each sub-fase EL.2 through EL.5: scoring (Sampson),
RANSAC loop, geometric refit, and the ``_fit_ellipse_check`` wrapper.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.ellipse_fitting import (
    _fit_ellipse_algebraic,
    _conic_to_geometric,
    _sampson_distance,
    _fit_ellipse_ransac,
    _geometric_to_conic,
    _orthogonal_distance_to_ellipse,
    _refit_ellipse_geometric,
    EllipseFitConfig,
    _fit_ellipse_check,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_ellipse_points(
    xc: float, yc: float,
    a: float, b: float,
    theta: float,
    n: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` points on a perfect ellipse, uniformly in parameter.

    The parametric form  (a·cos t, b·sin t)  does NOT space points
    uniformly along arc length, but it is good enough for fit-recovery
    tests because the algebraic fit is a least-squares minimiser, not
    sensitive to the parameterisation of inputs.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x_local = a * np.cos(t)
    y_local = b * np.sin(t)
    c, s = np.cos(theta), np.sin(theta)
    x = xc + c * x_local - s * y_local
    y = yc + s * x_local + c * y_local
    return x, y


def _theta_diff_mod_pi(a: float, b: float) -> float:
    """Smallest absolute angular difference, modulo π (ellipse symmetry)."""
    d = (a - b) % np.pi
    return min(d, np.pi - d)


# ===========================================================================
# EL.1 — _fit_ellipse_algebraic + _conic_to_geometric, synthetic recovery
# ===========================================================================

class TestAlgebraicFitRecovery:
    """Noise-free ellipses → parameters recovered with error < 1e-8."""

    def test_axis_aligned_a_along_x(self):
        # Semi-major along +x (a > b, theta = 0)
        xc_t, yc_t, a_t, b_t, theta_t = 2.0, 3.0, 5.0, 2.0, 0.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t)

        coefs = _fit_ellipse_algebraic(X, Y)
        assert coefs is not None
        result = _conic_to_geometric(coefs)
        assert result is not None
        xc, yc, a, b, theta = result

        assert abs(xc - xc_t) < 1e-9
        assert abs(yc - yc_t) < 1e-9
        assert abs(a - a_t) < 1e-9
        assert abs(b - b_t) < 1e-9
        assert _theta_diff_mod_pi(theta, theta_t) < 1e-9

    def test_axis_aligned_a_along_y(self):
        # Semi-major along +y → theta = π/2
        xc_t, yc_t, a_t, b_t, theta_t = -1.0, 0.5, 7.0, 3.0, np.pi / 2.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t)

        coefs = _fit_ellipse_algebraic(X, Y)
        result = _conic_to_geometric(coefs)
        assert result is not None
        xc, yc, a, b, theta = result

        assert abs(xc - xc_t) < 1e-9
        assert abs(yc - yc_t) < 1e-9
        assert abs(a - a_t) < 1e-9
        assert abs(b - b_t) < 1e-9
        assert _theta_diff_mod_pi(theta, theta_t) < 1e-9

    def test_rotated_30deg(self):
        xc_t, yc_t, a_t, b_t, theta_t = -1.0, 4.0, 7.0, 3.0, np.pi / 6.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t)

        coefs = _fit_ellipse_algebraic(X, Y)
        result = _conic_to_geometric(coefs)
        assert result is not None
        xc, yc, a, b, theta = result

        assert abs(xc - xc_t) < 1e-9
        assert abs(yc - yc_t) < 1e-9
        assert abs(a - a_t) < 1e-9
        assert abs(b - b_t) < 1e-9
        assert _theta_diff_mod_pi(theta, theta_t) < 1e-9

    def test_rotated_negative_angle(self):
        # Negative input angle should still recover its equivalent in [0, π)
        xc_t, yc_t, a_t, b_t, theta_t = 0.0, 0.0, 4.0, 2.5, -np.pi / 4.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t)

        coefs = _fit_ellipse_algebraic(X, Y)
        result = _conic_to_geometric(coefs)
        assert result is not None
        xc, yc, a, b, theta = result

        assert abs(a - a_t) < 1e-9
        assert abs(b - b_t) < 1e-9
        assert _theta_diff_mod_pi(theta, theta_t) < 1e-9

    def test_near_circle(self):
        # A near-circle (a ≈ b) must still recover the centre and radius
        # within tolerance; the rotation angle is essentially arbitrary
        # in this limit so we do not test it.
        xc_t, yc_t, r_t = 5.0, -2.0, 3.0
        X, Y = _make_ellipse_points(xc_t, yc_t, r_t, r_t, 0.0, n=50)

        coefs = _fit_ellipse_algebraic(X, Y)
        result = _conic_to_geometric(coefs)
        assert result is not None
        xc, yc, a, b, _theta = result

        assert abs(xc - xc_t) < 1e-8
        assert abs(yc - yc_t) < 1e-8
        assert abs(a - r_t) < 1e-8
        assert abs(b - r_t) < 1e-8
        assert abs(a - b) < 1e-8

    def test_minimum_points_5(self):
        # Exactly 5 points on a known ellipse: degrees-of-freedom-matched
        # case, the fit must still recover parameters (no overdetermination
        # margin, but no underdetermination either).
        xc_t, yc_t, a_t, b_t, theta_t = 1.0, 1.0, 4.0, 2.0, np.pi / 8.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t, n=5)

        coefs = _fit_ellipse_algebraic(X, Y)
        result = _conic_to_geometric(coefs)
        assert result is not None
        xc, yc, a, b, theta = result

        assert abs(xc - xc_t) < 1e-7
        assert abs(yc - yc_t) < 1e-7
        assert abs(a - a_t) < 1e-7
        assert abs(b - b_t) < 1e-7
        assert _theta_diff_mod_pi(theta, theta_t) < 1e-7


# ===========================================================================
# EL.1 — degenerate-input handling
# ===========================================================================

class TestAlgebraicFitDegenerate:

    def test_too_few_points_returns_none(self):
        X = np.array([0.0, 1.0, 2.0, 3.0])
        Y = np.array([0.0, 1.0, 0.5, 1.5])
        assert _fit_ellipse_algebraic(X, Y) is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            _fit_ellipse_algebraic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_collinear_points_returns_none(self):
        # Five collinear points: scatter matrix S3 is singular.
        X = np.linspace(0.0, 4.0, 5)
        Y = 2.0 * X + 1.0
        result = _fit_ellipse_algebraic(X, Y)
        # The fit should either return None or yield a non-ellipse on
        # conversion. Both outcomes signal "no valid ellipse" — we accept
        # either, but at least one must reject.
        if result is not None:
            assert _conic_to_geometric(result) is None


class TestConicToGeometricDegenerate:

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            _conic_to_geometric(np.array([1.0, 2.0, 3.0]))

    def test_hyperbola_returns_none(self):
        # x² − y² = 1: A=1, B=0, C=−1, F=−1; discriminant = 4 > 0.
        coefs = np.array([1.0, 0.0, -1.0, 0.0, 0.0, -1.0])
        assert _conic_to_geometric(coefs) is None

    def test_parabola_returns_none(self):
        # y² − x = 0:  A=0, B=0, C=1, D=−1, E=0, F=0; discriminant = 0.
        coefs = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
        assert _conic_to_geometric(coefs) is None


# ===========================================================================
# EL.2 — _sampson_distance, scale-corrected point-to-conic distance
# ===========================================================================

def _axis_aligned_ellipse_coefs(a: float, b: float) -> np.ndarray:
    """Conic coefficients for the canonical axis-aligned ellipse
    centred at the origin: ``(x/a)² + (y/b)² − 1 = 0``."""
    return np.array([1.0 / (a * a), 0.0, 1.0 / (b * b), 0.0, 0.0, -1.0])


class TestSampsonDistance:

    def test_points_exactly_on_ellipse_have_zero_distance(self):
        # Points on a known ellipse → Sampson must be ~0 (numerical noise)
        xc, yc, a, b, theta = 1.0, 2.0, 5.0, 3.0, np.pi / 6.0
        X, Y = _make_ellipse_points(xc, yc, a, b, theta, n=30)
        coefs = _fit_ellipse_algebraic(X, Y)
        assert coefs is not None

        d = _sampson_distance(X, Y, coefs)
        assert np.all(d < 1e-9)

    def test_known_orthogonal_offset_recovered_to_first_order(self):
        # Generate points exactly on an axis-aligned ellipse, then offset
        # each one along the outward unit normal by a small known distance.
        # Sampson distance should recover the offset to first order in d.
        a, b = 4.0, 2.5
        coefs = _axis_aligned_ellipse_coefs(a, b)
        X_on, Y_on = _make_ellipse_points(0.0, 0.0, a, b, 0.0, n=20)

        # Outward unit normal = ∇F / |∇F|  (∇F points outward when F is
        # negative inside the ellipse, which it is for our convention).
        A, B, C, D, E, F = coefs
        fx = 2.0 * A * X_on + B * Y_on + D
        fy = B * X_on + 2.0 * C * Y_on + E
        gnorm = np.sqrt(fx * fx + fy * fy)
        nx, ny = fx / gnorm, fy / gnorm

        d_known = 0.01  # 1 cm — small relative to min curvature radius b²/a = 1.56
        X_off = X_on + d_known * nx
        Y_off = Y_on + d_known * ny

        d_sampson = _sampson_distance(X_off, Y_off, coefs)
        # First-order error is O(d²·κ); with d=0.01 and κ ≈ b/a² ≈ 0.156 at
        # the major-axis end, expected error per point is ≲ 1e-4. Use 1e-3
        # to absorb worst-case curvature on the minor-axis end too.
        assert np.all(np.abs(d_sampson - d_known) < 1e-3)

    def test_corrects_scale_bias_of_algebraic_distance(self):
        # Two similar ellipses differing only in scale, evaluated at a
        # point at the same physical offset (d=0.01) along the +x axis.
        # Algebraic |F| should differ by ≈ scale ratio; Sampson should not.
        d_known = 0.01

        # Small ellipse: a=1, b=0.6
        a_s, b_s = 1.0, 0.6
        coefs_small = _axis_aligned_ellipse_coefs(a_s, b_s)
        # Large ellipse: a=10, b=6
        a_l, b_l = 10.0, 6.0
        coefs_large = _axis_aligned_ellipse_coefs(a_l, b_l)

        pt_small_x = np.array([a_s + d_known])
        pt_small_y = np.array([0.0])
        pt_large_x = np.array([a_l + d_known])
        pt_large_y = np.array([0.0])

        # Raw algebraic value |F|: should reveal the bias
        def _F(x, y, c):
            A, B, C, D, E, F = c
            return abs(A * x * x + B * x * y + C * y * y + D * x + E * y + F)

        f_small = float(_F(pt_small_x[0], pt_small_y[0], coefs_small))
        f_large = float(_F(pt_large_x[0], pt_large_y[0], coefs_large))
        # Algebraic |F| ratio should be roughly (a_l / a_s) = 10
        # because |F| ≈ 2d/a for an axis-aligned ellipse at its major-axis tip.
        assert f_small / f_large > 5.0  # well above the Sampson ratio (≈1)

        # Sampson should give ≈ d for both
        d_samp_small = float(_sampson_distance(pt_small_x, pt_small_y, coefs_small)[0])
        d_samp_large = float(_sampson_distance(pt_large_x, pt_large_y, coefs_large)[0])
        assert abs(d_samp_small - d_known) < 1e-3
        assert abs(d_samp_large - d_known) < 1e-3

    def test_circle_unit_offset_matches_closed_form(self):
        # Unit circle F = x² + y² − 1; point at (1+d, 0) has closed-form
        # Sampson = d·(2+d)/(2+2d). Verify exact agreement.
        coefs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        d_known = 0.01
        X = np.array([1.0 + d_known])
        Y = np.array([0.0])
        expected = d_known * (2.0 + d_known) / (2.0 + 2.0 * d_known)

        d_samp = float(_sampson_distance(X, Y, coefs)[0])
        assert abs(d_samp - expected) < 1e-12

    def test_vectorised_returns_per_point_distances(self):
        # Unit circle, four points: shape (4,) output, all non-negative,
        # point at (1,0) on the curve has zero distance.
        coefs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        X = np.array([1.0, 2.0, 0.5, 1.5])
        Y = np.array([0.0, 0.0, 0.5, 1.5])

        d = _sampson_distance(X, Y, coefs)
        assert d.shape == (4,)
        assert np.all(d >= 0.0)
        assert d[0] < 1e-12  # exactly on the unit circle

    def test_mismatched_input_lengths_raises(self):
        coefs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        with pytest.raises(ValueError):
            _sampson_distance(np.array([1.0, 2.0]), np.array([1.0]), coefs)

    def test_invalid_coefs_length_raises(self):
        with pytest.raises(ValueError):
            _sampson_distance(np.array([1.0]), np.array([1.0]), np.array([1.0, 2.0]))


# ===========================================================================
# EL.3 — _fit_ellipse_ransac, robust ellipse fit via consensus sampling
# ===========================================================================

class TestRansacEllipseFit:

    def test_clean_data_recovers_parameters(self):
        # No outliers, no noise → RANSAC should match the algebraic fit.
        xc_t, yc_t, a_t, b_t, theta_t = 1.0, 2.0, 5.0, 3.0, np.pi / 6.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t, n=60)

        rng = np.random.default_rng(seed=42)
        result = _fit_ellipse_ransac(
            X, Y, n_iters=50, tau_sampson=1e-6, rng=rng,
        )
        assert result is not None
        coefs, inliers, n_in = result
        assert n_in == 60  # all points are inliers on noise-free data

        geom = _conic_to_geometric(coefs)
        assert geom is not None
        xc, yc, a, b, theta = geom
        assert abs(xc - xc_t) < 1e-6
        assert abs(yc - yc_t) < 1e-6
        assert abs(a - a_t) < 1e-6
        assert abs(b - b_t) < 1e-6
        assert _theta_diff_mod_pi(theta, theta_t) < 1e-6

    def test_30_percent_outliers_rejected(self):
        # 70 inliers on an ellipse + 30 outliers in a lateral cluster.
        # RANSAC must lock onto the ellipse, not be pulled by the cluster.
        # Pure LS on the same data would shift the centre toward the cluster.
        rng = np.random.default_rng(seed=42)
        xc_t, yc_t, a_t, b_t = 0.0, 0.0, 1.0, 0.6
        X_in, Y_in = _make_ellipse_points(xc_t, yc_t, a_t, b_t, 0.0, n=70)

        # Tight outlier cluster centred at (1.5, 0) — well outside the
        # ellipse (right edge at x=1.0) so outliers cannot be near-inliers.
        outlier_xy = rng.normal(loc=[1.5, 0.0], scale=0.05, size=(30, 2))
        X = np.concatenate([X_in, outlier_xy[:, 0]])
        Y = np.concatenate([Y_in, outlier_xy[:, 1]])

        result = _fit_ellipse_ransac(
            X, Y, n_iters=200, tau_sampson=0.005, rng=rng,
        )
        assert result is not None
        coefs, inliers, n_in = result

        geom = _conic_to_geometric(coefs)
        assert geom is not None
        xc, yc, a, b, _theta = geom

        # Centre must stay at the true centre — not pulled to (1.5, 0).
        assert abs(xc - xc_t) < 0.005, f"xc drifted: got {xc}"
        assert abs(yc - yc_t) < 0.005, f"yc drifted: got {yc}"

        # Inliers should be (almost) all 70 true points and (almost) no
        # outliers. Allow a bit of slack for borderline-sample noise.
        true_inlier_recall = inliers[:70].sum() / 70
        outlier_false_positive = inliers[70:].sum() / 30
        assert true_inlier_recall >= 0.95
        assert outlier_false_positive <= 0.05

    def test_reproducible_with_seed(self):
        rng_a = np.random.default_rng(seed=123)
        rng_b = np.random.default_rng(seed=123)
        X, Y = _make_ellipse_points(0.0, 0.0, 4.0, 2.0, np.pi / 8.0, n=40)

        res_a = _fit_ellipse_ransac(X, Y, n_iters=30, tau_sampson=1e-4, rng=rng_a)
        res_b = _fit_ellipse_ransac(X, Y, n_iters=30, tau_sampson=1e-4, rng=rng_b)
        assert res_a is not None and res_b is not None
        np.testing.assert_array_equal(res_a[1], res_b[1])
        np.testing.assert_allclose(res_a[0], res_b[0], rtol=0, atol=0)

    def test_too_few_points_returns_none(self):
        X = np.array([0.0, 1.0, 2.0])
        Y = np.array([0.0, 1.0, 0.5])
        assert _fit_ellipse_ransac(X, Y) is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            _fit_ellipse_ransac(np.array([1.0, 2.0]), np.array([1.0]))

    def test_invalid_iter_count_raises(self):
        X, Y = _make_ellipse_points(0, 0, 1, 0.5, 0, n=20)
        with pytest.raises(ValueError):
            _fit_ellipse_ransac(X, Y, n_iters=0)

    def test_invalid_tau_raises(self):
        X, Y = _make_ellipse_points(0, 0, 1, 0.5, 0, n=20)
        with pytest.raises(ValueError):
            _fit_ellipse_ransac(X, Y, tau_sampson=0.0)

    def test_min_inliers_floor_enforced(self):
        # Pure noise → no real consensus → min_inliers floor should bite.
        rng = np.random.default_rng(seed=7)
        X = rng.uniform(-1, 1, size=30)
        Y = rng.uniform(-1, 1, size=30)
        # Demand more inliers than the data could ever produce.
        result = _fit_ellipse_ransac(
            X, Y, n_iters=50, tau_sampson=1e-6, min_inliers=25, rng=rng,
        )
        assert result is None


# ===========================================================================
# EL.4 — _geometric_to_conic, _orthogonal_distance_to_ellipse,
#        _refit_ellipse_geometric (LM precision refit)
# ===========================================================================

class TestGeometricToConic:

    def test_unit_circle(self):
        coefs = _geometric_to_conic(0.0, 0.0, 1.0, 1.0, 0.0)
        # x² + y² − 1 = 0
        np.testing.assert_allclose(coefs, [1.0, 0.0, 1.0, 0.0, 0.0, -1.0])

    def test_axis_aligned_major_y(self):
        # x²/4 + y²/9 − 1 = 0  ⇒  A=1/4, C=1/9
        coefs = _geometric_to_conic(0.0, 0.0, 3.0, 2.0, np.pi / 2.0)
        np.testing.assert_allclose(coefs[0], 1.0 / 4.0, atol=1e-12)
        np.testing.assert_allclose(coefs[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(coefs[2], 1.0 / 9.0, atol=1e-12)
        np.testing.assert_allclose(coefs[3:5], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(coefs[5], -1.0, atol=1e-12)

    def test_round_trip_geometric_to_conic_to_geometric(self):
        # Random parameters → geometric_to_conic → conic_to_geometric
        # must recover the original parameters (within numerical noise).
        rng = np.random.default_rng(seed=11)
        for _ in range(8):
            xc_t = rng.uniform(-5, 5)
            yc_t = rng.uniform(-5, 5)
            a_t = rng.uniform(1.0, 5.0)
            b_t = rng.uniform(0.3, a_t)  # ensure b ≤ a
            theta_t = rng.uniform(0.0, np.pi)

            coefs = _geometric_to_conic(xc_t, yc_t, a_t, b_t, theta_t)
            geom = _conic_to_geometric(coefs)
            assert geom is not None
            xc, yc, a, b, theta = geom
            assert abs(xc - xc_t) < 1e-9
            assert abs(yc - yc_t) < 1e-9
            assert abs(a - a_t) < 1e-9
            assert abs(b - b_t) < 1e-9
            assert _theta_diff_mod_pi(theta, theta_t) < 1e-9


class TestOrthogonalDistanceToEllipse:

    def test_unit_circle_radial_offset(self):
        # Point at distance d outside the unit circle along +x: distance = d
        d = 0.05
        X = np.array([1.0 + d])
        Y = np.array([0.0])
        dist = _orthogonal_distance_to_ellipse(X, Y, 0.0, 0.0, 1.0, 1.0, 0.0)
        assert abs(float(dist[0]) - d) < 1e-9

    def test_axis_aligned_offsets_at_axis_ends(self):
        # Ellipse a=4, b=2 centred at origin. Points at (4+d, 0), (0, 2+d),
        # (−4−d, 0), (0, −2−d) — all at orthogonal distance d.
        d = 0.1
        a, b = 4.0, 2.0
        X = np.array([a + d, 0.0, -a - d, 0.0])
        Y = np.array([0.0, b + d, 0.0, -b - d])
        dist = _orthogonal_distance_to_ellipse(X, Y, 0.0, 0.0, a, b, 0.0)
        np.testing.assert_allclose(dist, d, atol=1e-9)

    def test_point_on_ellipse_returns_zero(self):
        # Points generated parametrically on the ellipse → distance ≈ 0
        a, b, theta = 5.0, 3.0, np.pi / 5.0
        X, Y = _make_ellipse_points(0.0, 0.0, a, b, theta, n=20)
        dist = _orthogonal_distance_to_ellipse(X, Y, 0.0, 0.0, a, b, theta)
        assert np.all(dist < 1e-9)

    def test_rotated_ellipse_known_normal_offset(self):
        # Generate points on ellipse, push each one outward along the
        # outward normal by a known d. Orthogonal distance must recover d
        # exactly (within Newton tolerance).
        xc, yc, a, b, theta = 1.0, -2.0, 4.0, 2.0, np.pi / 3.0
        X_on, Y_on = _make_ellipse_points(xc, yc, a, b, theta, n=24)

        # Compute outward normal at each point via the ellipse gradient
        # in canonical frame, then rotate back.
        cos_th, sin_th = np.cos(theta), np.sin(theta)
        p = (X_on - xc) * cos_th + (Y_on - yc) * sin_th
        q = -(X_on - xc) * sin_th + (Y_on - yc) * cos_th
        # ∇(p²/a² + q²/b²) = (2p/a², 2q/b²) ⇒ unit outward normal in canon
        nx_c = 2.0 * p / (a * a)
        ny_c = 2.0 * q / (b * b)
        norm = np.sqrt(nx_c * nx_c + ny_c * ny_c)
        nx_c /= norm
        ny_c /= norm
        # Rotate back to world frame
        nx = nx_c * cos_th - ny_c * sin_th
        ny = nx_c * sin_th + ny_c * cos_th

        d = 0.03
        X = X_on + d * nx
        Y = Y_on + d * ny

        dist = _orthogonal_distance_to_ellipse(X, Y, xc, yc, a, b, theta)
        np.testing.assert_allclose(dist, d, atol=1e-7)


class TestRefitEllipseGeometric:

    def test_clean_data_idempotent(self):
        # Starting from a perfect algebraic fit on noise-free data, the
        # geometric refit must not move the parameters appreciably.
        xc_t, yc_t, a_t, b_t, theta_t = 0.5, -0.3, 0.15, 0.12, np.pi / 7.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t, n=80)
        coefs_alg = _fit_ellipse_algebraic(X, Y)
        assert coefs_alg is not None

        coefs_geom = _refit_ellipse_geometric(X, Y, coefs_alg)
        assert coefs_geom is not None

        geom = _conic_to_geometric(coefs_geom)
        assert geom is not None
        xc, yc, a, b, theta = geom
        # Tolerance: scipy LM convergence + Newton inner-iter noise.
        assert abs(xc - xc_t) < 1e-6
        assert abs(yc - yc_t) < 1e-6
        assert abs(a - a_t) < 1e-6
        assert abs(b - b_t) < 1e-6
        assert _theta_diff_mod_pi(theta, theta_t) < 1e-6

    def test_noisy_data_refit_matches_or_improves_algebraic(self):
        # Noisy points on a known ellipse: the geometric refit's centre
        # error must be no worse than the algebraic fit's centre error.
        # On expectation the geometric refit removes the algebraic bias
        # toward the conic singular set, but on a single random draw it
        # may already be very close.
        rng = np.random.default_rng(seed=2026)
        xc_t, yc_t, a_t, b_t, theta_t = 1.0, -2.0, 0.5, 0.35, np.pi / 6.0
        X_on, Y_on = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t, n=120)
        noise = rng.normal(scale=0.005, size=(X_on.size, 2))  # σ = 5 mm
        X = X_on + noise[:, 0]
        Y = Y_on + noise[:, 1]

        coefs_alg = _fit_ellipse_algebraic(X, Y)
        assert coefs_alg is not None
        coefs_geom = _refit_ellipse_geometric(X, Y, coefs_alg)
        assert coefs_geom is not None

        g_alg = _conic_to_geometric(coefs_alg)
        g_geom = _conic_to_geometric(coefs_geom)
        assert g_alg is not None and g_geom is not None

        err_alg = np.hypot(g_alg[0] - xc_t, g_alg[1] - yc_t)
        err_geom = np.hypot(g_geom[0] - xc_t, g_geom[1] - yc_t)
        # Geometric refit must not make the centre estimate worse —
        # within numerical slack of 0.5 mm.
        assert err_geom <= err_alg + 5e-4

    def test_invalid_initial_returns_none(self):
        # Hyperbola coefs: the conic_to_geometric returns None, so refit
        # must too.
        X = np.linspace(-1, 1, 20)
        Y = np.linspace(-1, 1, 20)
        bad_coefs = np.array([1.0, 0.0, -1.0, 0.0, 0.0, -1.0])  # x²−y²−1=0
        assert _refit_ellipse_geometric(X, Y, bad_coefs) is None

    def test_too_few_points_returns_none(self):
        X = np.array([0.0, 1.0, 2.0, 3.0])
        Y = np.array([0.0, 1.0, 0.5, 1.5])
        coefs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])  # unit circle
        assert _refit_ellipse_geometric(X, Y, coefs) is None

    def test_mismatched_lengths_raises(self):
        coefs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        with pytest.raises(ValueError):
            _refit_ellipse_geometric(np.array([1.0, 2.0]), np.array([1.0]), coefs)

    def test_refit_max_inliers_subsamples_large_input(self):
        # Refit on 5000 clean inliers with max_inliers=200 must produce
        # essentially the same fit as refit on all 5000 — within tens
        # of microns. This validates that the subsampling cap is safe
        # at the precision relevant to forestry LiDAR.
        xc_t, yc_t, a_t, b_t, theta_t = 0.0, 0.0, 0.18, 0.12, np.pi / 6.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t, n=5000)
        coefs_alg = _fit_ellipse_algebraic(X, Y)
        assert coefs_alg is not None

        # Disable cap → fit on all 5000
        coefs_full = _refit_ellipse_geometric(X, Y, coefs_alg, max_inliers=0)
        # Cap at 200
        coefs_capped = _refit_ellipse_geometric(X, Y, coefs_alg, max_inliers=200)

        assert coefs_full is not None and coefs_capped is not None
        g_full = _conic_to_geometric(coefs_full)
        g_capped = _conic_to_geometric(coefs_capped)
        assert g_full is not None and g_capped is not None

        # Same xc, yc, a, b within 1e-5 m
        for i in range(4):
            assert abs(g_full[i] - g_capped[i]) < 1e-5, (
                f"param {i}: full={g_full[i]} capped={g_capped[i]}"
            )
        # Theta within 1e-5 rad
        assert _theta_diff_mod_pi(g_full[4], g_capped[4]) < 1e-5

    def test_refit_negative_max_inliers_raises(self):
        X = np.linspace(0, 1, 10)
        Y = np.linspace(0, 1, 10)
        coefs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        with pytest.raises(ValueError):
            _refit_ellipse_geometric(X, Y, coefs, max_inliers=-5)

    def test_refit_no_subsampling_when_below_cap(self):
        # 100 points and cap=500 → no subsampling happens. Result must
        # equal a refit with no cap.
        X, Y = _make_ellipse_points(0.0, 0.0, 0.15, 0.10, 0.0, n=100)
        coefs_alg = _fit_ellipse_algebraic(X, Y)
        coefs_a = _refit_ellipse_geometric(X, Y, coefs_alg, max_inliers=0)
        coefs_b = _refit_ellipse_geometric(X, Y, coefs_alg, max_inliers=500)
        assert coefs_a is not None and coefs_b is not None
        np.testing.assert_allclose(coefs_a, coefs_b, atol=0)

    def test_ransac_then_refit_pipeline(self):
        # Full pipeline integration: noisy ellipse + 20% outliers.
        # RANSAC selects inliers, geometric refit polishes. Final centre
        # error must be small.
        rng = np.random.default_rng(seed=2026)
        xc_t, yc_t, a_t, b_t = 0.0, 0.0, 0.15, 0.12
        X_in, Y_in = _make_ellipse_points(xc_t, yc_t, a_t, b_t, 0.0, n=80)
        noise = rng.normal(scale=0.002, size=(X_in.size, 2))  # 2 mm noise
        X_in = X_in + noise[:, 0]
        Y_in = Y_in + noise[:, 1]
        # 20 lateral outliers
        out_xy = rng.normal(loc=[0.4, 0.0], scale=0.02, size=(20, 2))
        X = np.concatenate([X_in, out_xy[:, 0]])
        Y = np.concatenate([Y_in, out_xy[:, 1]])

        ransac_result = _fit_ellipse_ransac(
            X, Y, n_iters=200, tau_sampson=0.006, rng=rng,
        )
        assert ransac_result is not None
        coefs_ransac, inlier_mask, _ = ransac_result

        coefs_refit = _refit_ellipse_geometric(
            X[inlier_mask], Y[inlier_mask], coefs_ransac,
        )
        assert coefs_refit is not None

        g_refit = _conic_to_geometric(coefs_refit)
        assert g_refit is not None
        xc, yc, _a, _b, _theta = g_refit
        # Sub-mm centre recovery on 2 mm noise after the full pipeline.
        assert abs(xc - xc_t) < 1e-3
        assert abs(yc - yc_t) < 1e-3


# ===========================================================================
# EL.5 — _fit_ellipse_check, production wrapper with quality checks
# ===========================================================================

def _make_stem_section_config(**overrides) -> EllipseFitConfig:
    """Default config tuned for a typical stem section (radius ~0.15 m,
    moderate noise). Overrides let individual tests tweak the bits they
    care about without spelling out every field."""
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


class TestFitEllipseCheck:

    def test_clean_stem_section_returns_status_0(self):
        # Synthetic stem section: ~circular cross-section, radius 0.15 m
        # with a slight ovality (a=0.16, b=0.14), no noise, no outliers.
        rng = np.random.default_rng(seed=42)
        xc_t, yc_t, a_t, b_t, theta_t = 0.5, 0.5, 0.16, 0.14, np.pi / 8.0
        X, Y = _make_ellipse_points(xc_t, yc_t, a_t, b_t, theta_t, n=120)

        cfg = _make_stem_section_config()
        xc, yc, a, b, theta, status, sector_pct = _fit_ellipse_check(
            X, Y, cfg, rng=rng,
        )

        assert status == 0
        assert abs(xc - xc_t) < 1e-4
        assert abs(yc - yc_t) < 1e-4
        assert abs(a - a_t) < 1e-4
        assert abs(b - b_t) < 1e-4
        assert _theta_diff_mod_pi(theta, theta_t) < 1e-4
        assert sector_pct > 80.0  # well-covered curve

    def test_too_few_points_returns_status_2(self):
        # 10 points, min_points_section=40 → status 2 immediately.
        X = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        Y = np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
        cfg = _make_stem_section_config()
        xc, yc, a, b, theta, status, sector_pct = _fit_ellipse_check(X, Y, cfg)
        assert status == 2
        assert (xc, yc, a, b, theta, sector_pct) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_outliers_handled_by_ransac(self):
        # 80 inliers on a clean stem section + 20 lateral outliers.
        # RANSAC inside _fit_ellipse_check must absorb them.
        rng = np.random.default_rng(seed=2026)
        xc_t, yc_t, a_t, b_t = 0.0, 0.0, 0.18, 0.14
        X_in, Y_in = _make_ellipse_points(xc_t, yc_t, a_t, b_t, 0.0, n=80)
        out_xy = rng.normal(loc=[0.4, 0.0], scale=0.02, size=(20, 2))
        X = np.concatenate([X_in, out_xy[:, 0]])
        Y = np.concatenate([Y_in, out_xy[:, 1]])

        cfg = _make_stem_section_config()
        xc, yc, a, b, theta, status, sector_pct = _fit_ellipse_check(
            X, Y, cfg, rng=rng,
        )

        # Status 0 OR 1 acceptable — both indicate a recovered fit.
        # What matters is the centre stays near the truth.
        assert status in (0, 1)
        assert abs(xc - xc_t) < 0.003
        assert abs(yc - yc_t) < 0.003
        assert abs(a - a_t) < 0.005
        assert abs(b - b_t) < 0.005

    def test_radius_out_of_range_rejected(self):
        # Generate a HUGE ellipse (a=2, b=1.5) but config r_max=0.40.
        # √(a·b) = √3 ≈ 1.73 > 0.40 → must fail the radius check.
        rng = np.random.default_rng(seed=42)
        X, Y = _make_ellipse_points(0.0, 0.0, 2.0, 1.5, 0.0, n=120)

        cfg = _make_stem_section_config()
        xc, yc, a, b, theta, status, _ = _fit_ellipse_check(X, Y, cfg, rng=rng)

        # Status must be 1 (rejected; retry didn't recover either).
        assert status == 1
        # All zeros surfaced.
        assert (xc, yc, a, b, theta) == (0.0, 0.0, 0.0, 0.0, 0.0)

    def test_extreme_aspect_ratio_rejected(self):
        # Highly elongated ellipse a=0.2, b=0.02 → aspect 0.1 << 0.5 min.
        # _fit_ellipse_check must reject (status 1).
        rng = np.random.default_rng(seed=42)
        X, Y = _make_ellipse_points(0.0, 0.0, 0.2, 0.02, 0.0, n=120)

        cfg = _make_stem_section_config()
        xc, yc, a, b, theta, status, _ = _fit_ellipse_check(X, Y, cfg, rng=rng)
        assert status == 1

    def test_pure_noise_returns_status_1(self):
        # 100 random points uniformly in a 0.3 m × 0.3 m square.
        # RANSAC may find some hypothesis, but the inlier_fraction or
        # sector check will fail it → status 1.
        rng = np.random.default_rng(seed=7)
        X = rng.uniform(-0.15, 0.15, size=100)
        Y = rng.uniform(-0.15, 0.15, size=100)

        cfg = _make_stem_section_config()
        _xc, _yc, _a, _b, _theta, status, _ = _fit_ellipse_check(
            X, Y, cfg, rng=rng,
        )
        assert status == 1

    def test_reproducible_with_seed(self):
        # Same seed → same output (the RNG flows through to RANSAC).
        rng_a = np.random.default_rng(seed=999)
        rng_b = np.random.default_rng(seed=999)
        X, Y = _make_ellipse_points(0.0, 0.0, 0.15, 0.12, np.pi / 9.0, n=80)
        # Add a few outliers so RANSAC has actual decisions to make.
        outliers_rng = np.random.default_rng(seed=1)
        out = outliers_rng.normal(loc=[0.3, 0.0], scale=0.01, size=(10, 2))
        X = np.concatenate([X, out[:, 0]])
        Y = np.concatenate([Y, out[:, 1]])

        cfg = _make_stem_section_config()
        res_a = _fit_ellipse_check(X, Y, cfg, rng=rng_a)
        res_b = _fit_ellipse_check(X, Y, cfg, rng=rng_b)

        assert res_a == res_b
