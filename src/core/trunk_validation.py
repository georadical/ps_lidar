"""
Trunk Validation — Stem Cleaning + Sectioning (inspired by 3DFin).

This module implements the post-extraction refinement of trunk points:

Step 1: clean_stems()
    Takes trunk_mask points and runs a SECOND verticality pass to remove
    non-vertical material (branches, attached understory, foliage) that
    survived the initial trunk extraction. This is 3DFin's Step 4.

Step 2: compute_stem_sections()
    Slices each cleaned stem into horizontal sections and fits circles
    using least-squares optimisation (scipy.optimize.leastsq). Returns
    per-section centres, radii, and quality metrics. This is 3DFin's Step 5.

The output of Step 2 can later be used for:
  - Per-section adaptive trunk mask refinement (Step 3, future)
  - DAP estimation
  - Taper curves
  - Feature extraction (sweep, knots, fluting, etc.)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import optimize

from src.core.features import voxelize_cloud, compute_verticality
from src.core.trunk_extraction import TrunkExtractionResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StemCleaningConfig:
    """
    Configuration for stem cleaning and sectioning.

    Verticality pass parameters:
        verticality_threshold: Minimum verticality to keep a point as stem.
        verticality_scale: Neighbourhood radius for PCA.
        voxel_resolution_xy: Horizontal voxel size.
        voxel_resolution_z: Vertical voxel size.

    Sectioning parameters (like 3DFin):
        section_len: Distance between section centres (metres).
        section_wid: Half-width of each section slice (metres).
            Points within [section_h - wid, section_h + wid] are used.
        min_points_section: Minimum points for circle fitting.
        r_min: Minimum valid fitted radius (metres).
        r_max: Maximum valid fitted radius (metres).
        n_sectors: Number of angular sectors for quality check.
        min_sectors: Minimum occupied sectors for valid fit.
        sector_width: Width around fitted circle to check sectors (metres).
        inner_circle_ratio: Ratio for inner circle quality test.
        max_inner_points: Maximum points in inner circle before flagging.
        minimum_height: Lowest height for sections (metres).
        maximum_height: Highest height for sections (metres).
    """
    # 2nd verticality pass
    verticality_threshold: float = 0.7
    verticality_scale: float = 0.1
    voxel_resolution_xy: float = 0.02
    voxel_resolution_z: float = 0.02

    # Sectioning
    section_len: float = 0.2       # distance between sections (m)
    section_wid: float = 0.05      # half-width of section slice (m)
    min_points_section: int = 80
    r_min: float = 0.03            # minimum radius (m)
    r_max: float = 0.50            # maximum radius (m)
    n_sectors: int = 16
    min_sectors: int = 9
    sector_width: float = 0.02     # width around circle for sector check (m)
    inner_circle_ratio: float = 0.5
    max_inner_points: int = 5
    minimum_height: float = 0.3    # lowest section (m)
    maximum_height: float = 25.0   # highest section (m)
    cluster_eps: float = 0.02      # DBSCAN eps for section clustering (m)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StemCleaningResult:
    """Result of stem cleaning (2nd verticality pass)."""
    stem_mask: np.ndarray         # (N,) bool — updated stem mask
    n_points_before: int
    n_points_after: int
    n_points_removed: int
    per_tree_stats: List[Dict[str, Any]]


@dataclass
class SectionResult:
    """Result of stem sectioning (circle fitting)."""
    X_c: np.ndarray               # (n_trees, n_sections) X centres
    Y_c: np.ndarray               # (n_trees, n_sections) Y centres
    R: np.ndarray                  # (n_trees, n_sections) radii
    check: np.ndarray             # (n_trees, n_sections) validity flag
    sector_pct: np.ndarray        # (n_trees, n_sections) sector occupancy %
    sections: np.ndarray          # (n_sections,) section heights
    tree_ids: List[int]           # tree IDs in order
    config: StemCleaningConfig


# ---------------------------------------------------------------------------
# Step 1: Stem cleaning — 2nd verticality pass
# ---------------------------------------------------------------------------

def clean_stems(
    xyz: np.ndarray,
    trunk_result: TrunkExtractionResult,
    config: StemCleaningConfig,
    verbose: bool = True,
) -> StemCleaningResult:
    """
    Clean trunk points with a second verticality pass.

    For each tree, takes trunk_mask points and recomputes verticality.
    Points below the threshold are removed from the trunk mask.
    This removes branches, attached understory, and foliage that
    survived the initial extraction.

    Args:
        xyz: (N, 3) full height-normalised point cloud.
        trunk_result: Output of extract_trunks().
        config: Cleaning parameters.
        verbose: Print progress.

    Returns:
        StemCleaningResult with updated stem_mask.
    """
    trunk_mask = trunk_result.trunk_mask.copy()
    tree_ids = trunk_result.tree_ids
    n_before = int(trunk_mask.sum())

    if verbose:
        print(f"\nStem cleaning (2nd verticality pass):")
        print(f"  Input: {n_before:,} trunk points")
        print(f"  Verticality threshold: {config.verticality_threshold}")
        print(f"  Scale: {config.verticality_scale}m")

    # Get all trunk points
    trunk_indices = np.where(trunk_mask)[0]
    trunk_pts = xyz[trunk_indices]

    if len(trunk_pts) == 0:
        return StemCleaningResult(
            stem_mask=trunk_mask,
            n_points_before=0, n_points_after=0, n_points_removed=0,
            per_tree_stats=[],
        )

    # Compute verticality on trunk points
    if verbose:
        print(f"  Computing verticality on {len(trunk_pts):,} trunk points...")

    vert = compute_verticality(
        trunk_pts,
        scale=config.verticality_scale,
        voxel_resolution_xy=config.voxel_resolution_xy,
        voxel_resolution_z=config.voxel_resolution_z,
    )

    # Filter by verticality threshold
    vert_mask = vert >= config.verticality_threshold

    # Remove non-vertical points from trunk mask
    remove_indices = trunk_indices[~vert_mask]
    trunk_mask[remove_indices] = False

    n_after = int(trunk_mask.sum())

    # Per-tree statistics
    unique_trees = sorted(set(tree_ids[trunk_result.trunk_mask]) - {-1})
    per_tree = []
    for tid in unique_trees:
        tree_before = trunk_result.trunk_mask & (tree_ids == tid)
        tree_after = trunk_mask & (tree_ids == tid)
        nb = int(tree_before.sum())
        na = int(tree_after.sum())
        pct = ((nb - na) / max(nb, 1)) * 100
        per_tree.append({
            "tree_id": tid,
            "before": nb,
            "after": na,
            "removed": nb - na,
            "pct_removed": pct,
        })
        if verbose:
            print(f"    Tree {tid:3d}: {nb:,} → {na:,} pts (-{pct:.0f}%)")

    if verbose:
        print(f"  Summary: {n_before:,} → {n_after:,} "
              f"({n_before - n_after:,} non-vertical points removed)")

    return StemCleaningResult(
        stem_mask=trunk_mask,
        n_points_before=n_before,
        n_points_after=n_after,
        n_points_removed=n_before - n_after,
        per_tree_stats=per_tree,
    )


# ---------------------------------------------------------------------------
# Circle fitting (least squares, like 3DFin/dendromatics)
# ---------------------------------------------------------------------------

def _fit_circle_ls(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a circle to 2D points using least-squares (scipy.optimize.leastsq).
    Returns (centre_xy, mean_radius).
    """
    def _calc_R(X, Y, xc, yc):
        return np.sqrt((X - xc)**2 + (Y - yc)**2)

    def _residuals(c, X, Y):
        Ri = _calc_R(X, Y, *c)
        return Ri - Ri.mean()

    x0 = np.array([X.mean(), Y.mean()])
    centre, _ = optimize.leastsq(_residuals, x0, args=(X, Y), maxfev=2000)
    radius = _calc_R(X, Y, *centre).mean()
    return centre, float(radius)


def _sector_occupancy(
    X: np.ndarray, Y: np.ndarray,
    xc: float, yc: float, R: float,
    n_sectors: int, min_sectors: int, width: float,
) -> Tuple[float, bool]:
    """
    Check sector occupancy around the fitted circle.
    Returns (pct_occupied, is_valid).
    """
    X_red = X - xc
    Y_red = Y - yc
    radial = np.sqrt(X_red**2 + Y_red**2)
    angular = np.arctan2(X_red, Y_red)

    # Points near the circle
    near = (radial > (R - width)) & (radial < (R + width))
    if near.sum() == 0:
        return 0.0, False

    sectors = np.floor(angular[near] / (2 * np.pi / n_sectors))
    n_occupied = len(np.unique(sectors))
    pct = n_occupied * 100 / n_sectors
    return pct, n_occupied >= min_sectors


def _inner_circle_count(
    X: np.ndarray, Y: np.ndarray,
    xc: float, yc: float, R: float, ratio: float,
) -> int:
    """Count points inside inner circle (radius * ratio)."""
    dist = np.sqrt((X - xc)**2 + (Y - yc)**2)
    return int(np.sum(dist < R * ratio))


def _point_clustering_largest(X: np.ndarray, Y: np.ndarray, eps: float):
    """DBSCAN on XY, return points from largest cluster."""
    from sklearn.cluster import DBSCAN
    xy = np.column_stack([X, Y])
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(xy)
    if labels.max() < 0:
        return X, Y
    # Find largest cluster
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest = unique[np.argmax(counts)]
    mask = labels == largest
    return X[mask], Y[mask]


def _fit_circle_check(
    X: np.ndarray, Y: np.ndarray,
    config: StemCleaningConfig,
    is_retry: bool = False,
) -> Tuple[float, float, float, int, float]:
    """
    Fit circle with quality checks (like 3DFin's fit_circle_check).
    Returns (xc, yc, R, check_status, sector_pct).

    check_status: 0 = valid, 1 = checked (retry), 2 = not enough points
    """
    if len(X) < config.min_points_section:
        return 0.0, 0.0, 0.0, 2, 0.0

    centre, R = _fit_circle_ls(X, Y)
    xc, yc = centre

    n_inner = _inner_circle_count(X, Y, xc, yc, R, config.inner_circle_ratio)
    sector_pct, sectors_ok = _sector_occupancy(
        X, Y, xc, yc, R,
        config.n_sectors, config.min_sectors, config.sector_width,
    )

    # Check if fit is good
    fit_bad = (
        n_inner > config.max_inner_points
        or R < config.r_min
        or R > config.r_max
        or not sectors_ok
    )

    if fit_bad and not is_retry:
        # Retry with largest cluster
        Xg, Yg = _point_clustering_largest(X, Y, config.cluster_eps)
        if len(Xg) >= config.min_points_section:
            return _fit_circle_check(Xg, Yg, config, is_retry=True)
        else:
            return 0.0, 0.0, 0.0, 1, 0.0

    return xc, yc, R, (1 if is_retry else 0), sector_pct


# ---------------------------------------------------------------------------
# Step 2: Compute sections (circle fitting per section, per tree)
# ---------------------------------------------------------------------------

def compute_stem_sections(
    xyz: np.ndarray,
    stem_mask: np.ndarray,
    tree_ids: np.ndarray,
    config: StemCleaningConfig,
    verbose: bool = True,
) -> SectionResult:
    """
    Compute stem sections with circle fitting for each tree.

    For each tree, slices the cleaned stem points into horizontal sections
    and fits circles using least-squares. Returns per-section centres,
    radii, and quality metrics.

    Args:
        xyz: (N, 3) full height-normalised point cloud.
        stem_mask: (N,) boolean mask — cleaned stem points.
        tree_ids: (N,) tree ID per point.
        config: Sectioning parameters.
        verbose: Print progress.

    Returns:
        SectionResult with per-tree, per-section circle fits.
    """
    sections = np.arange(
        config.minimum_height,
        config.maximum_height,
        config.section_len,
    )
    n_sections = len(sections)

    # Get unique tree IDs from stem mask
    unique_trees = sorted(set(tree_ids[stem_mask]) - {-1})
    n_trees = len(unique_trees)

    if verbose:
        print(f"\nStem sectioning:")
        print(f"  Trees: {n_trees}")
        print(f"  Sections: {n_sections} "
              f"({config.minimum_height}m → {config.maximum_height}m, "
              f"step={config.section_len}m, width=±{config.section_wid}m)")

    # Allocate output arrays
    X_c = np.zeros((n_trees, n_sections))
    Y_c = np.zeros((n_trees, n_sections))
    R = np.zeros((n_trees, n_sections))
    check = np.zeros((n_trees, n_sections))
    sector_pct = np.zeros((n_trees, n_sections))

    for i, tid in enumerate(unique_trees):
        # Get stem points for this tree
        tree_mask = stem_mask & (tree_ids == tid)
        tree_pts = xyz[tree_mask]

        n_valid = 0
        radii_valid = []

        for j, sh in enumerate(sections):
            # Select points in this section slice
            z_low = sh - config.section_wid
            z_high = sh + config.section_wid
            sec_mask = (tree_pts[:, 2] >= z_low) & (tree_pts[:, 2] < z_high)

            sec_X = tree_pts[sec_mask, 0]
            sec_Y = tree_pts[sec_mask, 1]

            # Fit circle with checks
            xc, yc, r, chk, spct = _fit_circle_check(sec_X, sec_Y, config)

            X_c[i, j] = xc
            Y_c[i, j] = yc
            R[i, j] = r
            check[i, j] = chk
            sector_pct[i, j] = spct

            if r > 0:
                n_valid += 1
                radii_valid.append(r)

        if verbose:
            mean_r = np.mean(radii_valid) if radii_valid else 0
            mean_d = mean_r * 2
            print(f"    Tree {tid:3d}: {n_valid}/{n_sections} valid sections, "
                  f"mean_diam={mean_d:.3f}m")

    return SectionResult(
        X_c=X_c,
        Y_c=Y_c,
        R=R,
        check=check,
        sector_pct=sector_pct,
        sections=sections,
        tree_ids=unique_trees,
        config=config,
    )


# ---------------------------------------------------------------------------
# Step 3: Tree-level filters (Height and Edge)
# ---------------------------------------------------------------------------

@dataclass
class TreeFilterConfig:
    """
    Configuration for final tree-level filtering (Stage B & D).
    
    Parameters:
        plot_center_x: X coordinate of plot centre.
        plot_center_y: Y coordinate of plot centre.
        max_distance_from_center: Maximum allowed distance from plot centre to 
                                  tree centroid (metres) before it's considered an edge tree.
    """
    plot_center_x: float = 0.0
    plot_center_y: float = 0.0
    max_distance_from_center: float = 15.5  # Slightly less than plot radius to catch edge intersections


@dataclass
class TreeFilterResult:
    """Result of final tree filtering."""
    stem_mask: np.ndarray         # (N,) bool — updated stem mask with remaining valid trees
    tree_ids: np.ndarray          # (N,) int — updated tree IDs (-1 for filtered)
    n_trees_before: int
    n_trees_after: int
    n_trees_removed: int
    trees_removed_edge: List[int]


def filter_trees(
    xyz: np.ndarray,
    stem_mask: np.ndarray,
    tree_ids: np.ndarray,
    config: TreeFilterConfig,
    verbose: bool = True,
) -> TreeFilterResult:
    """
    Filter trees by distance from plot centre.
    
    Trees that are too far from the
    centre (cut edge trees) are completely removed from the stem_mask and tree_ids.

    Args:
        xyz: (N, 3) full height-normalised point cloud.
        stem_mask: (N,) boolean mask of cleaned stem points.
        tree_ids: (N,) tree ID per point.
        config: Filter parameters.
        verbose: Print progress.

    Returns:
        TreeFilterResult with updated masks.
    """
    updated_stem_mask = stem_mask.copy()
    updated_tree_ids = tree_ids.copy()
    
    unique_trees = sorted(set(tree_ids[stem_mask]) - {-1})
    n_before = len(unique_trees)
    
    if verbose:
        print(f"\nTree-level filtering:")
        print(f"  Input: {n_before} trees")
        print(f"  Max distance from centre ({config.plot_center_x:.2f}, {config.plot_center_y:.2f}): {config.max_distance_from_center}m")

    removed_edge = []

    
    for tid in unique_trees:
        tree_pts_mask = stem_mask & (tree_ids == tid)
        x_vals = xyz[tree_pts_mask, 0]
        y_vals = xyz[tree_pts_mask, 1]
        
        # 1. Edge filter
        centroid_x = x_vals.mean()
        centroid_y = y_vals.mean()
        distance = np.sqrt((centroid_x - config.plot_center_x)**2 + (centroid_y - config.plot_center_y)**2)
        
        if distance > config.max_distance_from_center:
            removed_edge.append(tid)
            updated_stem_mask[tree_pts_mask] = False
            updated_tree_ids[tree_pts_mask] = -1
            if verbose:
                print(f"    Tree {tid:3d}: ✗ REJECTED (edge tree: dist={distance:.1f}m > {config.max_distance_from_center}m)")
            continue
            
        if verbose:
            print(f"    Tree {tid:3d}: ✓ valid (dist={distance:.1f}m)")

    n_after = len(unique_trees) - len(removed_edge)
    
    if verbose:
        print(f"  Summary: {n_before} → {n_after} trees ({len(removed_edge)} edge)")

    return TreeFilterResult(
        stem_mask=updated_stem_mask,
        tree_ids=updated_tree_ids,
        n_trees_before=n_before,
        n_trees_after=n_after,
        n_trees_removed=n_before - n_after,
        trees_removed_edge=removed_edge,
    )
