"""
Ellipse Fitting — algebraic, RANSAC, and geometric refit primitives.

This module provides ellipse-fitting building blocks shared by Phase 1B
(tracking-based ``tree_id`` assignment, coarse slices) and Phase 1C
(per-slice ellipse RANSAC, fine slices) of Mejora 1 — centerline curvo.

The public entry point — analogous to ``_fit_circle_check`` in
``src/core/trunk_validation.py`` — will be ``_fit_ellipse_check``, a
wrapper that orchestrates a RANSAC loop using the primitives in this
module and applies geometric quality checks (sector occupancy, inner
empty, aspect ratio, inlier fraction).

This file is built atomically across sub-fases EL.1 through EL.5; each
sub-fase adds one primitive and its synthetic-data validation gate. See
``project_mejora1_phase1b_decision.md`` (revised 2026-05-15) for the plan.

References
----------
- Fitzgibbon, A., Pilu, M., Fisher, R. B. (1999). "Direct least square
  fitting of ellipses." IEEE TPAMI 21(5):476-480. Original closed-form
  algebraic fit via generalised eigenvalue problem.
- Halíř, R., Flusser, J. (1998). "Numerically stable direct least squares
  fitting of ellipses." Proc. 6th International Conference in Central
  Europe on Computer Graphics and Visualization (WSCG'98). The
  reformulation used here, which avoids the singular-matrix failure
  mode of Fitzgibbon's original formulation by splitting the design
  matrix into quadratic and linear blocks and solving a reduced
  3×3 eigenvalue problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize


# ===========================================================================
# EL.1 — Algebraic ellipse fit (closed-form, Halíř-Flusser 1998)
# ===========================================================================

def _fit_ellipse_algebraic(
    X: np.ndarray, Y: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Fit an ellipse to 2D points using the direct algebraic method of
    Halíř-Flusser (numerically stable variant of Fitzgibbon 1999).

    Solves the constrained least-squares problem

        min ||D · a||²   subject to   4·A·C − B² = 1

    where ``a = [A, B, C, D, E, F]`` are the coefficients of the general
    conic ``F(x, y) = A·x² + B·x·y + C·y² + D·x + E·y + F = 0`` and the
    quadratic constraint forces the solution to be an ellipse (not a
    parabola or hyperbola).

    Parameters
    ----------
    X, Y : array_like
        1D arrays of x and y coordinates. Must have the same length and
        contain at least 5 points (an ellipse has 5 degrees of freedom).

    Returns
    -------
    coefs : ndarray of shape (6,) or None
        Conic coefficients ``[A, B, C, D, E, F]``. Returns ``None`` when:
          - fewer than 5 points are supplied;
          - the linear scatter sub-matrix is singular (degenerate
            configuration, e.g. collinear points);
          - no eigenvector satisfies the ellipse constraint
            ``4·A·C − B² > 0`` (data fits a parabola or hyperbola
            better than any ellipse).

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` have different lengths.

    Notes
    -----
    The Halíř-Flusser reformulation splits the design matrix
    ``D = [x², xy, y², x, y, 1]`` into a quadratic block ``D1`` and a
    linear block ``D2``, then solves a reduced 3×3 generalised eigenvalue
    problem on the quadratic part only. This avoids the rank-deficient
    6×6 problem of Fitzgibbon's original formulation when the points are
    nearly noise-free or clustered.
    """
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if X.size != Y.size:
        raise ValueError(
            f"X and Y must have the same length; got {X.size} and {Y.size}"
        )
    if X.size < 5:
        return None

    # --- Design matrix split into quadratic (D1) and linear (D2) blocks ---
    D1 = np.column_stack([X * X, X * Y, Y * Y])            # (N, 3)
    D2 = np.column_stack([X, Y, np.ones_like(X)])          # (N, 3)

    # --- Scatter matrices ---
    S1 = D1.T @ D1                                          # (3, 3)
    S2 = D1.T @ D2                                          # (3, 3)
    S3 = D2.T @ D2                                          # (3, 3)

    # Reduce the linear part: solve  S3 @ T = S2.T  for T = inv(S3) @ S2.T
    try:
        S3_inv_S2T = np.linalg.solve(S3, S2.T)              # (3, 3)
    except np.linalg.LinAlgError:
        return None

    # --- Inverse of the constraint matrix C1 enforcing 4·A·C − B² = 1 ---
    # C1 = [[0, 0, 2], [0, -1, 0], [2, 0, 0]]  =>  inv(C1) below.
    C1_inv = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, -1.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Reduced scatter matrix on the quadratic part (3×3).
    M = C1_inv @ (S1 - S2 @ S3_inv_S2T)

    # --- Eigendecomposition: pick the eigenvector with positive constraint ---
    # Halíř-Flusser theorem: exactly one eigenvector satisfies
    # 4·a₁·a₃ − a₂² > 0; that one is the ellipse solution.
    eigvals, eigvecs = np.linalg.eig(M)

    # eigvecs has shape (3, 3) with eigenvectors as columns:
    #   eigvecs[0, i] is the first component of the i-th eigenvector, etc.
    cond = np.real(4.0 * eigvecs[0] * eigvecs[2] - eigvecs[1] ** 2)
    idx = np.where(cond > 0)[0]
    if idx.size == 0:
        return None

    a1 = np.real(eigvecs[:, idx[0]]).astype(np.float64)     # quadratic part

    # Recover the linear part:  a2 = −inv(S3) @ S2.T @ a1
    a2 = -S3_inv_S2T @ a1                                    # (3,)

    return np.concatenate([a1, a2])


# ---------------------------------------------------------------------------

def _conic_to_geometric(
    coefs: np.ndarray,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Convert ellipse conic coefficients to geometric parameters.

    Given the general conic ``A·x² + B·x·y + C·y² + D·x + E·y + F = 0``,
    extract the centre ``(xc, yc)``, the semi-major and semi-minor axes
    ``(a, b)`` with ``a ≥ b``, and the rotation angle ``theta`` (radians,
    counter-clockwise from the +x axis to the semi-major axis).

    Parameters
    ----------
    coefs : ndarray of shape (6,)
        Conic coefficients ``[A, B, C, D, E, F]``, typically the output
        of :func:`_fit_ellipse_algebraic`.

    Returns
    -------
    (xc, yc, a, b, theta) : tuple of float or None
        Centre coordinates, semi-major axis, semi-minor axis, and rotation
        angle in radians. ``theta`` is normalised to ``[0, π)`` (an
        ellipse is symmetric under ``theta → theta + π``). Returns
        ``None`` if the coefficients do not represent a non-degenerate
        ellipse — namely when the discriminant ``B² − 4·A·C`` is
        non-negative (parabola or hyperbola), or when the centred-conic
        value at the centre has the wrong sign (numerical degeneracy).

    Raises
    ------
    ValueError
        If ``coefs`` does not have length 6.

    Notes
    -----
    The conversion uses the eigendecomposition of the 2×2 quadratic-form
    matrix

        Q = [[A,   B/2],
             [B/2, C  ]]

    The eigenvalues of ``Q`` give the semi-axes via
    ``a² = −F_c / λ_min`` and ``b² = −F_c / λ_max`` (where ``F_c`` is the
    conic value evaluated at the centre), and the eigenvectors give the
    axis orientation. The eigendecomposition approach is preferred to
    the direct closed-form formulas because it sidesteps the sign-bookkeeping
    that determines major-vs-minor axis assignment.
    """
    coefs = np.asarray(coefs, dtype=np.float64).ravel()
    if coefs.size != 6:
        raise ValueError(f"coefs must have length 6; got {coefs.size}")
    A, B, C, D, E, F = coefs

    discr = B * B - 4.0 * A * C
    if discr >= 0.0:
        return None  # parabola or hyperbola, not an ellipse

    # The conic equations  +coefs = 0  and  −coefs = 0  describe the same
    # curve, so the algebraic fit returns one of two equivalent sign
    # conventions. We normalise to  A + C > 0  (which makes the eigenvalues
    # of the quadratic-form matrix Q both positive, since for an ellipse
    # the two eigenvalues already share a sign — that sign equals the sign
    # of the trace ``A + C``).
    if A + C < 0.0:
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F
        # discr is invariant under conic-sign-flip (each term picks up
        # two sign flips), so no need to recompute.

    # --- Centre (solution of the gradient = 0 condition) ---
    xc = (2.0 * C * D - B * E) / discr
    yc = (2.0 * A * E - B * D) / discr

    # --- Conic value at the centre (after translation, the constant term) ---
    F_c = A * xc * xc + B * xc * yc + C * yc * yc + D * xc + E * yc + F

    # --- Eigendecomposition of the 2×2 quadratic-form matrix Q ---
    Q = np.array([[A, 0.5 * B], [0.5 * B, C]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(Q)  # eigh: symmetric, sorted ascending

    if eigvals[0] <= 0.0 or eigvals[1] <= 0.0:
        return None  # quadratic form not positive-definite ⇒ not an ellipse

    # In the eigenbasis, the centred conic is  λ_min·u² + λ_max·v² = −F_c.
    # For a real ellipse we need −F_c > 0.
    a_sq = -F_c / eigvals[0]   # corresponds to smaller eigenvalue ⇒ larger axis
    b_sq = -F_c / eigvals[1]   # corresponds to larger eigenvalue ⇒ smaller axis
    if a_sq <= 0.0 or b_sq <= 0.0:
        return None

    a_axis = float(np.sqrt(a_sq))
    b_axis = float(np.sqrt(b_sq))

    # Semi-major axis direction: eigenvector of the smaller eigenvalue.
    major_dir = eigvecs[:, 0]
    theta = float(np.arctan2(major_dir[1], major_dir[0]))

    # Normalise to [0, π) — an ellipse is symmetric under theta + π.
    theta = theta % np.pi

    return float(xc), float(yc), a_axis, b_axis, theta


# ===========================================================================
# EL.2 — Sampson distance (point-to-conic, scale-corrected approximation
#         to the true orthogonal distance)
# ===========================================================================

def _sampson_distance(
    X: np.ndarray, Y: np.ndarray,
    coefs: np.ndarray,
) -> np.ndarray:
    """
    Compute the Sampson distance from 2D points to a conic.

    The Sampson distance is the first-order approximation to the
    geometric (orthogonal) distance between a point and a conic. For
    the general conic ``F(x, y) = A·x² + B·x·y + C·y² + D·x + E·y + F``,

        d_Sampson(x, y) = |F(x, y)| / sqrt((∂F/∂x)² + (∂F/∂y)²)

    where the gradient components are

        ∂F/∂x = 2·A·x + B·y + D
        ∂F/∂y = B·x + 2·C·y + E.

    Parameters
    ----------
    X, Y : array_like
        1D arrays of point coordinates of equal length.
    coefs : ndarray of shape (6,)
        Conic coefficients ``[A, B, C, D, E, F]`` (typically the output
        of :func:`_fit_ellipse_algebraic`).

    Returns
    -------
    d : ndarray of shape (N,)
        Non-negative Sampson distance for each point.

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` have different lengths, or ``coefs`` does not
        have length 6.

    Notes
    -----
    Sampson distance is the metric of choice for RANSAC scoring against
    a conic hypothesis (Hartley & Zisserman, *Multiple View Geometry*,
    §11.4). It corrects the **scale bias** of the raw algebraic distance
    ``|F(x, y)|`` — for two geometrically similar ellipses differing only
    in scale, the algebraic distance to a point at the same physical
    offset scales with the conic's size, while Sampson does not.

    The approximation degrades as the point moves further from the curve:
    Sampson under-estimates the true orthogonal distance by an amount
    proportional to ``d²·κ`` where ``κ`` is the local curvature. For
    inlier scoring in RANSAC this is typically negligible (offsets of
    millimetres compared to centimetre-scale curvatures); for outliers
    the under-estimate makes the metric *more* discriminative, not less.

    Singular points where ``∇F = 0`` (e.g. the geometric centre of an
    ellipse) have ``d_Sampson`` undefined geometrically. To keep the
    function total, the gradient norm is floored at ``1e-300`` before
    division, returning a very large value at the singular set rather
    than ``NaN``.
    """
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if X.size != Y.size:
        raise ValueError(
            f"X and Y must have the same length; got {X.size} and {Y.size}"
        )
    coefs = np.asarray(coefs, dtype=np.float64).ravel()
    if coefs.size != 6:
        raise ValueError(f"coefs must have length 6; got {coefs.size}")
    A, B, C, D, E, F = coefs

    # Algebraic conic value F(x, y)
    f_val = A * X * X + B * X * Y + C * Y * Y + D * X + E * Y + F

    # Gradient components
    fx = 2.0 * A * X + B * Y + D
    fy = B * X + 2.0 * C * Y + E

    grad_norm = np.sqrt(fx * fx + fy * fy)
    # Floor at a sub-normal value to avoid div-by-zero at singular points
    # without contaminating well-defined results.
    grad_norm = np.maximum(grad_norm, 1e-300)

    return np.abs(f_val) / grad_norm


# ===========================================================================
# EL.3 — RANSAC ellipse fit (robust to outliers via consensus sampling)
# ===========================================================================

def _fit_ellipse_ransac(
    X: np.ndarray, Y: np.ndarray,
    n_iters: int = 200,
    tau_sampson: float = 0.005,
    min_inliers: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Fit an ellipse robustly via RANSAC + Sampson-distance inlier scoring.

    Repeatedly draws minimal samples of 5 points, fits an algebraic
    ellipse on each sample via :func:`_fit_ellipse_algebraic`, scores
    **all** points with :func:`_sampson_distance` against the hypothesis,
    and keeps the hypothesis with the most inliers (Sampson distance
    below ``tau_sampson``). After the loop, refits the algebraic ellipse
    on the consensus set to use the full inlier information.

    Parameters
    ----------
    X, Y : array_like
        1D arrays of point coordinates of equal length. Must contain at
        least 5 points (the minimum to define an ellipse).
    n_iters : int, default 200
        Number of RANSAC iterations. With inlier ratio ~0.5, 200 iters
        give >99.9% probability of drawing at least one all-inlier sample.
    tau_sampson : float, default 0.005
        Inlier threshold on Sampson distance, in the same units as
        ``X``, ``Y`` (e.g. metres for LiDAR slices). A point is an inlier
        if its Sampson distance to the hypothesis is strictly less than
        this value.
    min_inliers : int, default 5
        Minimum number of inliers for the result to be returned.
        Below this threshold the function returns ``None`` (the data
        does not support any non-trivial ellipse).
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If ``None``, a
        fresh default generator is used (non-deterministic across calls).

    Returns
    -------
    (coefs, inlier_mask, n_inliers) : tuple or None
        ``coefs`` : ndarray of shape (6,), the conic coefficients of the
        best (refitted) ellipse.
        ``inlier_mask`` : ndarray of shape (N,), boolean mask of inliers
        with respect to the refitted ellipse.
        ``n_inliers`` : int, ``inlier_mask.sum()``.

        Returns ``None`` if fewer than 5 points are supplied, if no
        sampled minimal set produces a valid ellipse over ``n_iters``
        iterations (extreme degeneracy), or if the best hypothesis has
        fewer than ``min_inliers`` inliers.

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` have different lengths, ``n_iters < 1``, or
        ``tau_sampson <= 0``.

    Notes
    -----
    A final refit step is performed on the consensus set to exploit the
    information from all inliers (the minimal-sample fit only used 5
    points). The refit is in algebraic form via the same Halíř-Flusser
    primitive; the geometric refit on orthogonal distance is sub-fase
    EL.4 and consumes the inlier mask returned here.

    The refit is only accepted if it does not lose inliers relative to
    the minimal-sample hypothesis — this guards against rare cases where
    the refit, while a better LS fit on the consensus, ends up slightly
    further from a few borderline inliers.
    """
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if X.size != Y.size:
        raise ValueError(
            f"X and Y must have the same length; got {X.size} and {Y.size}"
        )
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1; got {n_iters}")
    if tau_sampson <= 0.0:
        raise ValueError(f"tau_sampson must be positive; got {tau_sampson}")
    n = X.size
    if n < 5:
        return None

    if rng is None:
        rng = np.random.default_rng()

    best_n_inliers = -1
    best_inlier_mask: Optional[np.ndarray] = None
    best_coefs: Optional[np.ndarray] = None

    indices = np.arange(n)

    for _ in range(n_iters):
        sample_idx = rng.choice(indices, size=5, replace=False)
        coefs = _fit_ellipse_algebraic(X[sample_idx], Y[sample_idx])
        if coefs is None:
            continue

        d = _sampson_distance(X, Y, coefs)
        inlier_mask = d < tau_sampson
        n_inliers = int(inlier_mask.sum())

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inlier_mask = inlier_mask
            best_coefs = coefs

    if best_coefs is None or best_n_inliers < min_inliers:
        return None

    # --- Refit on the consensus set ---
    refit_coefs = _fit_ellipse_algebraic(X[best_inlier_mask], Y[best_inlier_mask])
    if refit_coefs is not None:
        d_refit = _sampson_distance(X, Y, refit_coefs)
        refit_mask = d_refit < tau_sampson
        n_refit = int(refit_mask.sum())
        # Accept refit only if it does not regress on inlier count.
        if n_refit >= best_n_inliers:
            return refit_coefs, refit_mask, n_refit

    return best_coefs, best_inlier_mask, best_n_inliers


# ===========================================================================
# EL.4 — Geometric LS refit on inliers (true orthogonal distance)
# ===========================================================================

def _geometric_to_conic(
    xc: float, yc: float, a: float, b: float, theta: float,
) -> np.ndarray:
    """Inverse of :func:`_conic_to_geometric`: build the 6 conic
    coefficients from geometric parameters of an ellipse.

    The output is the conic ``A·x² + B·x·y + C·y² + D·x + E·y + F`` such
    that the equation ``= 0`` holds on the ellipse defined by
    ``(xc, yc, a, b, theta)``. The overall scale of the coefficients is
    normalised so that ``F = A·xc² + B·xc·yc + C·yc² − 1`` — i.e. the
    canonical-frame equation is ``u²/a² + v²/b² = 1``.
    """
    c, s = np.cos(theta), np.sin(theta)
    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)

    A = c * c * inv_a2 + s * s * inv_b2
    B = 2.0 * c * s * (inv_a2 - inv_b2)
    C = s * s * inv_a2 + c * c * inv_b2
    D = -2.0 * A * xc - B * yc
    E = -2.0 * C * yc - B * xc
    F = A * xc * xc + B * xc * yc + C * yc * yc - 1.0

    return np.array([A, B, C, D, E, F], dtype=np.float64)


def _orthogonal_distance_to_ellipse(
    X: np.ndarray, Y: np.ndarray,
    xc: float, yc: float, a: float, b: float, theta: float,
    n_newton_iter: int = 20,
) -> np.ndarray:
    """Compute the true orthogonal (Euclidean) distance from each
    ``(X, Y)`` to the ellipse defined by geometric parameters.

    For each query point, transforms to the ellipse's canonical frame
    and Newton-iterates on the parametric angle ``t`` such that the
    foot of perpendicular sits at ``(a·cos t, b·sin t)``. The condition
    ``f(t) = 0`` is

        f(t) = −a·p·sin t + b·q·cos t + (a² − b²)·cos t · sin t

    with derivative

        f′(t) = −a·p·cos t − b·q·sin t + (a² − b²)·(cos²t − sin²t).

    Convergence is quadratic once ``t`` is within the basin of attraction
    of the foot of perpendicular; for inlier-class points (already close
    to the curve) the default 20 iterations are far more than needed.

    Returns an array of non-negative distances of shape ``(N,)``.
    """
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    dx_orig = X - xc
    dy_orig = Y - yc
    # Canonical-frame coordinates (rotation by −theta, then translation
    # has already been applied).
    p = dx_orig * cos_th + dy_orig * sin_th
    q = -dx_orig * sin_th + dy_orig * cos_th

    # Initial parametric guess: scaled angle of (p, q).
    t = np.arctan2(q * a, p * b)

    a2_minus_b2 = a * a - b * b
    for _ in range(n_newton_iter):
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        f = -a * p * sin_t + b * q * cos_t + a2_minus_b2 * cos_t * sin_t
        fp = (
            -a * p * cos_t
            - b * q * sin_t
            + a2_minus_b2 * (cos_t * cos_t - sin_t * sin_t)
        )
        # Safe Newton step: floor |f'| to avoid div-by-zero. The sign
        # is preserved via np.where rather than np.maximum so the
        # iteration direction is correct.
        safe_fp = np.where(np.abs(fp) > 1e-12, fp, 1e-12 * np.sign(fp + 1e-30))
        t = t - f / safe_fp

    x_foot = a * np.cos(t)
    y_foot = b * np.sin(t)

    # Rigid transformation preserves distance, so we can compute it in
    # canonical frame.
    dx = p - x_foot
    dy = q - y_foot
    return np.sqrt(dx * dx + dy * dy)


def _refit_ellipse_geometric(
    X: np.ndarray, Y: np.ndarray,
    initial_coefs: np.ndarray,
    n_newton_iter: int = 20,
    max_lm_iter: int = 50,
) -> Optional[np.ndarray]:
    """Geometric LS refit of an ellipse on a (typically inlier) point set.

    Refines an existing algebraic fit (e.g. the consensus refit returned
    by :func:`_fit_ellipse_ransac`) by minimising the **true orthogonal
    distance** of each point to the ellipse, parameterised geometrically
    as ``(xc, yc, a, b, theta)``. Levenberg-Marquardt via
    :func:`scipy.optimize.least_squares` is used, with the Jacobian
    estimated by finite differences (default).

    This is the precision pass of the RANSAC pipeline: the algebraic
    LS refit on inliers (inside RANSAC) is biased toward the conic's
    singular set; the geometric refit eliminates that bias and typically
    moves the centre and radii by sub-millimetre on noise-free data,
    and 1–3 mm on data with ~mm-scale noise.

    Parameters
    ----------
    X, Y : array_like
        Inlier 2D coordinates. Must contain at least 5 points.
    initial_coefs : ndarray of shape (6,)
        Initial conic coefficients (typically from
        :func:`_fit_ellipse_ransac`). Must represent a non-degenerate
        ellipse — :func:`_conic_to_geometric` is applied to extract
        the starting geometric parameters, and the function returns
        ``None`` if that conversion fails.
    n_newton_iter : int, default 20
        Newton iterations inside each orthogonal-distance evaluation.
        20 is generous; 5–10 suffice for inlier-class points.
    max_lm_iter : int, default 50
        Soft cap on Levenberg-Marquardt iterations
        (``max_nfev = 6 · max_lm_iter`` since each LM iteration costs
        roughly one Jacobian evaluation = 5 forward passes + 1 residual,
        for the 5 parameters).

    Returns
    -------
    coefs : ndarray of shape (6,) or None
        Refitted conic coefficients in the same convention as the
        algebraic primitive. ``None`` if the initial coefs do not
        represent a valid ellipse, fewer than 5 points are supplied,
        the LM optimisation fails, or the optimiser drifts to a
        non-positive semi-axis.
    """
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if X.size != Y.size:
        raise ValueError(
            f"X and Y must have the same length; got {X.size} and {Y.size}"
        )
    if X.size < 5:
        return None

    geom0 = _conic_to_geometric(initial_coefs)
    if geom0 is None:
        return None
    xc0, yc0, a0, b0, theta0 = geom0

    # Sentinel residual used when the LM optimiser briefly steps into
    # an invalid region (a or b ≤ 0). Returning a finite but very large
    # value lets LM bounce back rather than failing the whole run.
    big = 1e10

    def residual(params: np.ndarray) -> np.ndarray:
        xc, yc, a, b, theta = params
        if a <= 0.0 or b <= 0.0:
            return np.full(X.size, big, dtype=np.float64)
        return _orthogonal_distance_to_ellipse(
            X, Y, xc, yc, a, b, theta, n_newton_iter=n_newton_iter,
        )

    try:
        result = optimize.least_squares(
            residual,
            x0=np.array([xc0, yc0, a0, b0, theta0], dtype=np.float64),
            method="lm",
            max_nfev=6 * max_lm_iter,
        )
    except (ValueError, RuntimeError):
        return None

    xc, yc, a, b, theta = result.x
    if a <= 0.0 or b <= 0.0:
        return None

    # Convention: a (semi-major) ≥ b (semi-minor). If LM has settled
    # with them swapped, swap and rotate theta by π/2 to compensate.
    if a < b:
        a, b = b, a
        theta = theta + 0.5 * np.pi

    return _geometric_to_conic(float(xc), float(yc), float(a), float(b), float(theta))


# ===========================================================================
# EL.5 — _fit_ellipse_check: production wrapper analogous to
#         _fit_circle_check in src/core/trunk_validation.py
# ===========================================================================

@dataclass(frozen=True)
class EllipseFitConfig:
    """Configuration for :func:`_fit_ellipse_check`.

    Mirrors the relevant section of ``StemCleaningConfig`` used by
    ``_fit_circle_check`` in ``src/core/trunk_validation.py``, extended
    with RANSAC and aspect-ratio parameters specific to ellipses.

    All length-like parameters are in the same units as the input
    coordinates (typically metres for forestry LiDAR).
    """

    # --- Geometry constraints (same role as in StemCleaningConfig) ---
    min_points_section: int = 80
    r_min: float = 0.05               # min equivalent radius √(a·b)
    r_max: float = 0.40               # max equivalent radius √(a·b)
    inner_ratio: float = 0.5          # inner-empty check threshold
    max_inner_points: int = 5         # max points inside the inner ellipse

    # --- Sector occupancy (curve-arc coverage check) ---
    n_sectors: int = 16
    min_sectors: int = 9
    sector_width: float = 0.02        # band around the curve, orthogonal

    # --- RANSAC ---
    ransac_n_iters: int = 200
    ransac_tau_sampson: float = 0.005
    min_inlier_fraction: float = 0.6

    # --- Ellipse-specific ---
    min_aspect_ratio: float = 0.5     # b / a must be ≥ this (rejects
                                      # degenerate elongated fits, e.g.
                                      # branches + a few stem points)

    # --- Retry path (DBSCAN clustering) ---
    cluster_eps: float = 0.02


# --- Helpers used by _fit_ellipse_check -----------------------------------

def _inner_ellipse_count(
    X: np.ndarray, Y: np.ndarray,
    xc: float, yc: float, a: float, b: float, theta: float,
    ratio: float,
) -> int:
    """Count points strictly inside the **inner** ellipse — the same
    ellipse with semi-axes scaled by ``ratio``.

    Geometric analogue of ``_inner_circle_count``. A healthy stem
    section has very few points inside the inner ellipse: stems are
    hollow shapes seen from above (we see the perimeter, not the wood).
    Lots of inner points signal that the fit absorbed a noisy interior
    cluster instead of locking onto the outer perimeter.
    """
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    p = (X - xc) * cos_th + (Y - yc) * sin_th
    q = -(X - xc) * sin_th + (Y - yc) * cos_th
    a_inner = a * ratio
    b_inner = b * ratio
    inside = (p / a_inner) ** 2 + (q / b_inner) ** 2 < 1.0
    return int(inside.sum())


def _sector_occupancy_ellipse(
    X: np.ndarray, Y: np.ndarray,
    xc: float, yc: float, a: float, b: float, theta: float,
    n_sectors: int, min_sectors: int, width: float,
) -> Tuple[float, bool]:
    """Sector-occupancy check around the ellipse.

    Counts how many angular sectors around the centre contain at least
    one point lying within ``width`` of the ellipse curve (orthogonal
    distance). Returns ``(percent_occupied, is_ok)`` where ``is_ok`` is
    ``n_occupied >= min_sectors``.

    Mirrors ``_sector_occupancy`` for circles. The angular bin uses
    ``arctan2(Δx, Δy)`` (same swap convention as the circle helper) and
    the "near-curve" band uses true orthogonal distance (more robust
    on highly eccentric ellipses than the radial band used for circles).
    """
    d = _orthogonal_distance_to_ellipse(X, Y, xc, yc, a, b, theta)
    near = d < width
    if not near.any():
        return 0.0, False

    dx = X[near] - xc
    dy = Y[near] - yc
    # Same convention as _sector_occupancy in trunk_validation.py:
    # arctan2(dx, dy) — a 90° rotation of the standard angle, but for
    # sector binning this is irrelevant (any rotation preserves the
    # number of occupied sectors).
    angular = np.arctan2(dx, dy)
    sectors = np.floor(angular / (2.0 * np.pi / n_sectors))
    n_occupied = int(np.unique(sectors).size)
    pct = 100.0 * n_occupied / n_sectors
    return pct, n_occupied >= min_sectors


def _point_clustering_largest(
    X: np.ndarray, Y: np.ndarray, eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return points of the largest DBSCAN cluster on XY.

    Duplicated from ``trunk_validation._point_clustering_largest`` to
    keep ``ellipse_fitting.py`` standalone — avoids a future circular
    import once ``trunk_validation`` consumes ``_fit_ellipse_check``.
    """
    from sklearn.cluster import DBSCAN

    xy = np.column_stack([X, Y])
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(xy)
    if labels.max() < 0:
        return X, Y
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest = unique[np.argmax(counts)]
    mask = labels == largest
    return X[mask], Y[mask]


# --- Main wrapper ---------------------------------------------------------

def _fit_ellipse_check(
    X: np.ndarray, Y: np.ndarray,
    config: EllipseFitConfig,
    is_retry: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float, float, float, int, float]:
    """Fit an ellipse with quality checks; production wrapper around the
    EL.1–EL.4 primitives. Analogue of ``_fit_circle_check`` in
    ``src/core/trunk_validation.py``.

    Pipeline:
        1. Reject sections with too few points (status 2).
        2. :func:`_fit_ellipse_ransac` produces a candidate + inliers.
        3. :func:`_refit_ellipse_geometric` polishes on the consensus set.
        4. Quality checks: equivalent radius √(a·b) in [r_min, r_max],
           inner-ellipse-empty, sector occupancy, aspect ratio b/a,
           inlier fraction.
        5. If the fit fails any check (or RANSAC found nothing), retry
           once on the largest DBSCAN cluster.

    Returns
    -------
    (xc, yc, a, b, theta, check_status, sector_pct) : tuple
        Geometric ellipse parameters. ``a ≥ b``; ``theta`` is the
        rotation (radians) of the semi-major axis from the +x axis.

        ``check_status``:
            * 0 → valid first-attempt fit
            * 1 → fit returned after a retry attempt
                  (either the retry succeeded, or the retry failed but
                   we surface zeros — same convention as the circle helper)
            * 2 → not enough points in the input slice

        ``sector_pct`` is the percentage of sectors occupied around the
        fitted ellipse (used as a quality indicator downstream).
    """
    n = len(X)
    if n < config.min_points_section:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 2, 0.0

    # --- RANSAC + geometric refit + quality checks ---
    min_inliers = max(5, int(config.min_inlier_fraction * n))
    ransac_result = _fit_ellipse_ransac(
        X, Y,
        n_iters=config.ransac_n_iters,
        tau_sampson=config.ransac_tau_sampson,
        min_inliers=min_inliers,
        rng=rng,
    )

    fit_bad = ransac_result is None
    xc = yc = a = b = theta = 0.0
    sector_pct = 0.0

    if not fit_bad:
        coefs, inlier_mask, n_inliers = ransac_result
        refit_coefs = _refit_ellipse_geometric(
            X[inlier_mask], Y[inlier_mask], coefs,
        )
        if refit_coefs is not None:
            coefs = refit_coefs

        geom = _conic_to_geometric(coefs)
        if geom is None:
            fit_bad = True
        else:
            xc, yc, a, b, theta = geom
            r_equiv = float(np.sqrt(a * b))
            inlier_fraction = n_inliers / n

            n_inner = _inner_ellipse_count(
                X, Y, xc, yc, a, b, theta, config.inner_ratio,
            )
            sector_pct, sectors_ok = _sector_occupancy_ellipse(
                X, Y, xc, yc, a, b, theta,
                config.n_sectors, config.min_sectors, config.sector_width,
            )
            aspect_ratio = b / a

            fit_bad = (
                n_inner > config.max_inner_points
                or r_equiv < config.r_min
                or r_equiv > config.r_max
                or not sectors_ok
                or aspect_ratio < config.min_aspect_ratio
                or inlier_fraction < config.min_inlier_fraction
            )

    # --- Retry path (mirrors _fit_circle_check) ---
    if fit_bad and not is_retry:
        Xg, Yg = _point_clustering_largest(X, Y, config.cluster_eps)
        if len(Xg) >= config.min_points_section:
            return _fit_ellipse_check(Xg, Yg, config, is_retry=True, rng=rng)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0

    if fit_bad:
        # is_retry and still bad: surface zeros + status 1
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0

    return xc, yc, a, b, theta, (1 if is_retry else 0), sector_pct
