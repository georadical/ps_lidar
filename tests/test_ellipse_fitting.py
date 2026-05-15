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
