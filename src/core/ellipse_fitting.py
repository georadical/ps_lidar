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

from typing import Optional, Tuple

import numpy as np


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
