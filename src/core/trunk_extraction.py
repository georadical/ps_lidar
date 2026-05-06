"""
Trunk Extraction — Module 7

Extracts tree trunks from a height-normalized point cloud using
the proven dendromatics/3DFin pipeline:
  1. Extract horizontal stripe at breast height
  2. Voxelize
  3. Compute verticality (pgeof)
  4. Filter by verticality threshold
  5. DBSCAN clustering
  6. Iterative peeling (repeat 2-5)
  7. Compute tree axes via PCA
  8. Assign all points to nearest axis

Field parameters (measured before/after LiDAR scan):
  - stripe_lower_limit / stripe_upper_limit: range with minimal branches/understory
  - dbh_min / dbh_max: expected diameter range
  - height_max: tallest tree in plot
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal, Tuple

from .features import voxelize_cloud, compute_verticality, compute_sphericity


# ---------------------------------------------------------------------------
# Configuration dataclass — field parameters
# ---------------------------------------------------------------------------

@dataclass
class TrunkExtractionConfig:
    """
    Configuration for trunk extraction.

    Field parameters (measured in the field):
        stripe_lower_limit: Lower height of the detection stripe (above understory).
        stripe_upper_limit: Upper height of the detection stripe (below branches).
        dbh_min: Smallest expected stem diameter in the plot.
        dbh_max: Largest expected stem diameter in the plot.
        height_max: Height of the tallest tree in the plot.

    Algorithm parameters (defaults from 3DFin):
        peeling_iterations: Number of verticality-peeling passes (1-5).
        verticality_threshold: Minimum verticality to consider a point as stem.
        verticality_scale: Neighbourhood radius for PCA via pgeof.
        voxel_resolution_xy: Horizontal voxel size for subsampling.
        voxel_resolution_z: Vertical voxel size for subsampling.
        min_cluster_points: Minimum voxels per DBSCAN cluster.
        stem_search_radius: Search radius around axis for stem point assignment.
        max_axis_distance: Maximum distance from axis to assign a point.
        height_range: Proportion of the stripe height that a cluster must span.
        dbscan_eps: DBSCAN epsilon (auto-computed from voxel_resolution if None).

    Cluster validation parameters (applied in the stripe before axis computation):
        cluster_circularity_min: Minimum XY circularity (PCA eigenvalue ratio) for a
            stripe cluster to be considered a valid trunk cross-section.
        cluster_diameter_max_factor: Maximum cluster XY diameter as a multiple of dbh_max.
            Clusters wider than this are rejected as understory.
        cluster_min_height: Minimum height for clusters with small diameter.
            Rejects regeneration saplings.
        cluster_min_diameter: Minimum diameter for short clusters.
    """
    # Field parameters
    stripe_lower_limit: float = 0.7     # m
    stripe_upper_limit: float = 3.5     # m
    dbh_min: float = 0.09               # m
    dbh_max: float = 1.0                # m
    height_max: float = 25.0            # m

    # Algorithm parameters
    peeling_iterations: int = 2
    verticality_threshold: float = 0.7
    verticality_scale: float = 0.1      # m
    voxel_resolution_xy: float = 0.02   # m
    voxel_resolution_z: float = 0.02    # m
    min_cluster_points: int = 1000
    stem_search_radius: float = 1.0     # m
    max_axis_distance: float = 15.0     # m
    height_range: float = 0.7
    dbscan_eps: Optional[float] = None  # auto if None

    # Centerline construction (Improvement 1, Phase 1A — informational only).
    # After point-to-axis assignment, build a piecewise centerline (polyline) per
    # tree by tracking the median XY position of assigned points across vertical
    # slabs of `centerline_slab_step` metres. The result is attached to each
    # tree_axes entry under the key `centerline` (an (K, 3) ndarray sorted by Z,
    # or None if too few points). Downstream consumers (point assignment, stem
    # cleaning, sectioning) still operate on the straight-line axis — Phases 1B
    # and 1C will switch them to the polyline incrementally.
    centerline_slab_step: float = 0.5         # m
    centerline_min_points_per_slab: int = 5   # minimum points to record a control point

    # Cluster validation parameters
    cluster_circularity_min: float = 0.3
    cluster_diameter_max_factor: float = 1.5
    cluster_min_height: float = 2.0     # m — min height for small trees
    cluster_min_diameter: float = 0.05  # m — min diameter

    # Multi-scale geometric pre-filter (Improvement 4 — opt-in).
    # Optional voxel-level rejection applied AFTER the verticality threshold
    # and BEFORE DBSCAN, on the same voxel centroids of each peeling
    # iteration. Targets two failure modes that pure verticality lets
    # through:
    #   * chaotic understory / foliage scatter (high sphericity = isotropic
    #     PCA spread). Rejected when sphericity > sphericity_max.
    #   * sparse leaves / twigs (very low local point density). Rejected
    #     when points/m³ < density_min.
    # Defaults preserve existing behaviour (filter disabled). To enable,
    # set sphericity_max to e.g. 0.35 and/or density_min to a positive
    # value calibrated against your data.
    #
    # Empirical closure on Hovermap/TFS data (plot T460298A, 60 trees):
    # the per-tree sphericity diagnostic in notebooks/01_playground.ipynb
    # showed the false positives observed in this dataset (Tree 7 and 10:
    # merged-cluster artifacts with absurd diameters and inconsistent
    # sectioning) have LOWER sphericity (median ~0.18-0.20) than the
    # legitimate trees (median ~0.21). They are highly cylindrical, not
    # chaotic. No per-voxel sphericity threshold can separate the two
    # groups for this sensor and setup. The correct fix for those FPs is
    # section-based rejection (Improvement 6), not this pre-filter. The
    # implementation is left opt-in for sensors / scenes where FPs are
    # genuine chaotic-foliage clusters.
    sphericity_max: float = 1.0   # 1.0 = no rejection
    density_min: float = 0.0      # 0.0 = no rejection (points / voxel m³)


    # Axis refinement (optional)
    axis_refinement_mode: Literal["none", "basal_anchor"] = "none"
    basal_anchor_min_height: float = 0.15
    basal_anchor_max_height: float = 0.80
    basal_anchor_gap_to_stripe: float = 0.05
    basal_anchor_slice_step: float = 0.05
    basal_anchor_slice_half_width: float = 0.03
    basal_anchor_cluster_eps: float = 0.03
    basal_anchor_min_points: int = 60
    basal_anchor_min_circularity: float = 0.55
    basal_anchor_min_sector_pct: float = 0.60
    basal_anchor_max_inner_fraction: float = 0.20
    basal_anchor_min_support_slices: int = 3
    basal_anchor_search_radius_factor: float = 1.5
    basal_anchor_search_radius_min: float = 0.20
    basal_anchor_search_radius_max: float = 0.60
    basal_anchor_radius_min_factor: float = 0.80
    basal_anchor_radius_max_factor: float = 1.35
    basal_anchor_min_arc_coverage: float = 0.20
    basal_anchor_max_fit_residual_ratio: float = 0.12
    basal_anchor_max_center_drift: float = 0.08
    basal_anchor_max_radius_cv: float = 0.25
    basal_anchor_max_axis_ratio: float = 1.80
    basal_anchor_match_max_xy_distance: float = 0.75
    basal_anchor_match_max_tilt_deg: float = 30.0
    basal_anchor_match_max_xy_offset: float = 0.20
    basal_anchor_match_radius_ratio_min: float = 0.60
    basal_anchor_match_radius_ratio_max: float = 1.60

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrunkExtractionResult:
    """Result of trunk extraction."""
    trunk_mask: np.ndarray          # (N,) bool — is this point part of a trunk?
    tree_ids: np.ndarray            # (N,) int — tree ID per point (-1 = unassigned)
    n_trees: int                    # number of trees detected
    tree_axes: List[Dict[str, Any]] # list of axis info per tree
    cluster_points: np.ndarray      # (M, 3) points from the final stripe clusters
    config: TrunkExtractionConfig   # config used


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_stripe(
    xyz: np.ndarray,
    z_min: float,
    z_max: float,
) -> tuple:
    """Extract horizontal stripe between z_min and z_max."""
    z = xyz[:, 2]
    mask = (z >= z_min) & (z <= z_max)
    return xyz[mask], mask, np.where(mask)[0]


def _dbscan_cluster(
    xyz: np.ndarray,
    eps: float,
    min_samples: int = 5,
) -> np.ndarray:
    """
    DBSCAN clustering on XY coordinates (2D, as dendromatics does).
    Returns labels array (-1 for noise).
    """
    from sklearn.cluster import DBSCAN

    # Cluster in 2D (XY) — stems are vertical, so Z doesn't discriminate
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz[:, :2])
    return labels


def _filter_small_clusters(
    labels: np.ndarray,
    min_points: int,
) -> np.ndarray:
    """Remove clusters with fewer than min_points."""
    filtered = labels.copy()
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl == -1:
            continue
        count = np.sum(labels == lbl)
        if count < min_points:
            filtered[labels == lbl] = -1
    return filtered


def _filter_by_height_range(
    xyz: np.ndarray,
    labels: np.ndarray,
    stripe_height: float,
    min_range_ratio: float,
) -> np.ndarray:
    """Remove clusters that don't span enough of the stripe vertically."""
    filtered = labels.copy()
    z = xyz[:, 2]
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        z_lbl = z[mask]
        z_range = z_lbl.max() - z_lbl.min()
        if z_range < stripe_height * min_range_ratio:
            filtered[mask] = -1
    return filtered


def _relabel_clusters(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters to consecutive integers starting from 0."""
    relabeled = np.full_like(labels, -1)
    unique_labels = sorted(set(labels) - {-1})
    for new_id, old_id in enumerate(unique_labels):
        relabeled[labels == old_id] = new_id
    return relabeled


def _summarize_cluster_geometry(xyz: np.ndarray) -> Dict[str, float]:
    """Compute audit-friendly geometry metrics for a stripe cluster."""
    if len(xyz) == 0:
        return {
            "stripe_points": 0,
            "stripe_circularity": float("nan"),
            "stripe_diameter": float("nan"),
            "stripe_z_span": float("nan"),
        }

    xy = xyz[:, :2]
    centroid_xy = np.median(xy, axis=0)
    centred = xy - centroid_xy

    if len(xy) < 2:
        circularity = 0.0
    else:
        cov = np.cov(centred, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(np.atleast_2d(cov))
        eig_max = max(float(eigenvalues.max()), 1e-10)
        eig_min = max(float(eigenvalues.min()), 0.0)
        circularity = eig_min / eig_max

    radii = np.linalg.norm(centred, axis=1)

    return {
        "stripe_points": int(len(xyz)),
        "stripe_circularity": float(circularity),
        "stripe_diameter": float(2.0 * np.mean(radii)),
        "stripe_z_span": float(xyz[:, 2].max() - xyz[:, 2].min()),
    }


def _validate_stripe_clusters(
    xyz: np.ndarray,
    labels: np.ndarray,
    config: 'TrunkExtractionConfig',
    verbose: bool = False,
) -> np.ndarray:
    """
    Validate stripe clusters by cross-section geometry.

    For each cluster in the stripe, fits a circle to the XY cross-section
    and checks circularity + diameter. Invalid clusters (understory masses,
    regeneration saplings) are rejected by setting their labels to -1.

    This runs BEFORE axis computation, so rejected clusters never get axes
    and their points are never assigned to trees.

    Returns updated labels array with invalid clusters set to -1.
    """
    labels = labels.copy()
    unique_labels = sorted(set(labels) - {-1})
    max_diameter = config.dbh_max * config.cluster_diameter_max_factor
    rejected = []

    for lbl in unique_labels:
        mask = labels == lbl
        pts = xyz[mask]

        # --- Cross-section analysis (XY) ---
        xy = pts[:, :2]
        centroid_xy = np.median(xy, axis=0)  # robust centre
        centred = xy - centroid_xy

        # PCA on XY for circularity
        cov = np.cov(centred, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        eig_max = max(eigenvalues.max(), 1e-10)
        eig_min = max(eigenvalues.min(), 0.0)
        circularity = eig_min / eig_max

        # Diameter estimate (2 * mean radial distance)
        radii = np.linalg.norm(centred, axis=1)
        diameter = 2.0 * np.mean(radii)

        # Height span of this cluster
        z_span = pts[:, 2].max() - pts[:, 2].min()

        # --- Validation rules ---
        reason = ""

        # Rule 1: Too wide → understory mass
        if diameter > max_diameter:
            reason = f"too_wide: diam={diameter:.3f}m > {max_diameter:.3f}m"

        # Rule 2: Too low circularity → irregular shape (understory)
        elif circularity < config.cluster_circularity_min:
            reason = f"not_circular: circ={circularity:.3f} < {config.cluster_circularity_min}"

        # Rule 3: Small diameter + short height → regeneration
        elif (diameter < config.cluster_min_diameter * 2
              and z_span < config.cluster_min_height):
            reason = (f"regeneration: diam={diameter:.3f}m, "
                      f"h={z_span:.1f}m")

        if reason:
            labels[mask] = -1
            rejected.append((lbl, reason))
            if verbose:
                print(f"    Cluster {lbl}: ✗ REJECTED ({reason}) "
                      f"[{mask.sum()} pts]")
        else:
            if verbose:
                print(f"    Cluster {lbl}: ✓ valid "
                      f"(circ={circularity:.2f}, diam={diameter:.3f}m, "
                      f"h={z_span:.1f}m) [{mask.sum()} pts]")

    if verbose:
        n_remaining = len(unique_labels) - len(rejected)
        print(f"    Cluster validation: {len(unique_labels)} → {n_remaining} "
              f"({len(rejected)} rejected)")

    return labels


def _empty_basal_anchor(
    total_candidates: int = 0,
    validated_candidates: int = 0,
) -> Dict[str, Any]:
    """Default empty metadata for basal anchor refinement."""
    return {
        "basal_anchor_found": False,
        "basal_anchor_applied": False,
        "basal_anchor_candidates_total": int(total_candidates),
        "basal_anchor_candidates_validated": int(validated_candidates),
        "basal_anchor_x": float("nan"),
        "basal_anchor_y": float("nan"),
        "basal_anchor_z": float("nan"),
        "basal_anchor_radius": float("nan"),
        "basal_anchor_circularity": float("nan"),
        "basal_anchor_sector_pct": float("nan"),
        "basal_anchor_inner_fraction": float("nan"),
        "basal_anchor_support_slices": 0,
        "basal_anchor_center_offset": float("nan"),
        "basal_anchor_model": "none",
        "basal_anchor_arc_coverage": float("nan"),
        "basal_anchor_fit_residual": float("nan"),
        "basal_anchor_track_center_std": float("nan"),
        "basal_anchor_track_radius_cv": float("nan"),
        "basal_anchor_track_id": -1,
        "basal_anchor_track_score": float("nan"),
        "basal_anchor_match_passed": False,
        "basal_anchor_match_score": float("nan"),
        "basal_anchor_match_xy_distance": float("nan"),
        "basal_anchor_match_radius_ratio": float("nan"),
        "basal_anchor_match_tilt_deg": float("nan"),
        "axis_direction_delta_deg": 0.0,
    }


def _fit_circle_xy(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares circle fit in XY using a robust median-based initialization."""
    from scipy import optimize

    def _calc_r(x, y, xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def _residuals(center, x, y):
        radii = _calc_r(x, y, *center)
        return radii - radii.mean()

    x = xy[:, 0]
    y = xy[:, 1]
    init = np.array([np.median(x), np.median(y)], dtype=np.float64)
    centre, _ = optimize.leastsq(_residuals, init, args=(x, y), maxfev=2000)
    radius = float(_calc_r(x, y, *centre).mean())
    return np.asarray(centre, dtype=np.float64), radius


def _sector_pct_and_inner_fraction(
    xy: np.ndarray,
    centre: np.ndarray,
    radius: float,
    n_sectors: int = 16,
    ring_width: float = 0.02,
    inner_ratio: float = 0.5,
) -> Tuple[float, float]:
    """Compute sector occupancy and inner-circle occupancy fraction for a fitted circle."""
    centred = xy - centre
    radial = np.linalg.norm(centred, axis=1)
    if radius <= 1e-12 or len(xy) == 0:
        return 0.0, 1.0

    near_ring = np.abs(radial - radius) <= ring_width
    sector_pct = 0.0
    if np.any(near_ring):
        angles = np.arctan2(centred[near_ring, 1], centred[near_ring, 0])
        sector_ids = np.floor((angles + np.pi) / (2.0 * np.pi / n_sectors)).astype(int)
        sector_ids = np.clip(sector_ids, 0, n_sectors - 1)
        sector_pct = float(len(np.unique(sector_ids)) / n_sectors)

    inner_fraction = float(np.sum(radial < (radius * inner_ratio)) / max(len(xy), 1))
    return sector_pct, inner_fraction


def _rotate_points(xy: np.ndarray, centre: np.ndarray, theta: float) -> np.ndarray:
    """Rotate points into the local frame of a circle/ellipse."""
    shifted = xy - centre
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    return np.column_stack([
        shifted[:, 0] * ct + shifted[:, 1] * st,
        -shifted[:, 0] * st + shifted[:, 1] * ct,
    ])


def _sector_ids_from_angles(angles: np.ndarray, n_sectors: int = 32) -> np.ndarray:
    """Convert angles to discrete sector ids."""
    if angles.size == 0:
        return np.empty(0, dtype=np.int32)
    sector_ids = np.floor((angles + np.pi) / (2.0 * np.pi / n_sectors)).astype(np.int32)
    return np.clip(sector_ids, 0, n_sectors - 1)


def _fit_ellipse_xy(xy: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Least-squares ellipse fit with PCA initialization."""
    from scipy import optimize

    if len(xy) < 8:
        raise ValueError("Not enough points to fit ellipse")

    centre = np.median(xy, axis=0).astype(np.float64)
    centred = xy - centre
    cov = np.cov(centred, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(cov))
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    theta0 = float(np.arctan2(major[1], major[0]))
    rotated = _rotate_points(xy, centre, theta0)
    a0 = max(float(np.std(rotated[:, 0]) * np.sqrt(2.0)), 1e-3)
    b0 = max(float(np.std(rotated[:, 1]) * np.sqrt(2.0)), 1e-3)
    if b0 > a0:
        a0, b0 = b0, a0
        theta0 += np.pi / 2.0

    def _residuals(params: np.ndarray) -> np.ndarray:
        cx, cy, log_a, log_b, theta = params
        a = np.exp(log_a)
        b = np.exp(log_b)
        rot = _rotate_points(xy, np.array([cx, cy], dtype=np.float64), theta)
        norm = np.sqrt((rot[:, 0] / a) ** 2 + (rot[:, 1] / b) ** 2)
        return norm - 1.0

    init = np.array([centre[0], centre[1], np.log(a0), np.log(b0), theta0], dtype=np.float64)
    params, _ = optimize.leastsq(_residuals, init, maxfev=4000)
    cx, cy, log_a, log_b, theta = params
    a = float(np.exp(log_a))
    b = float(np.exp(log_b))
    if b > a:
        a, b = b, a
        theta += np.pi / 2.0
    return np.array([cx, cy], dtype=np.float64), a, b, float(theta)


def _fit_circle_robust(
    xy: np.ndarray,
    max_fit_residual_ratio: float,
) -> Optional[Dict[str, Any]]:
    """Robust circle fit with one trimming/refit pass."""
    if len(xy) < 6:
        return None

    centre, radius = _fit_circle_xy(xy)
    if not np.isfinite(radius) or radius <= 1e-12:
        return None

    radial = np.linalg.norm(xy - centre, axis=1)
    residuals = np.abs(radial - radius)
    threshold = max(radius * max_fit_residual_ratio * 1.5, 0.01)
    inliers = residuals <= threshold
    if int(inliers.sum()) >= max(6, len(xy) // 2):
        centre, radius = _fit_circle_xy(xy[inliers])
        radial = np.linalg.norm(xy - centre, axis=1)
        residuals = np.abs(radial - radius)
        inliers = residuals <= max(radius * max_fit_residual_ratio * 1.5, 0.01)

    if int(inliers.sum()) < 6:
        return None

    inlier_xy = xy[inliers]
    inlier_radial = np.linalg.norm(inlier_xy - centre, axis=1)
    fit_residual = float(np.median(np.abs(inlier_radial - radius)) / max(radius, 1e-12))
    angles = np.arctan2(inlier_xy[:, 1] - centre[1], inlier_xy[:, 0] - centre[0])
    sector_ids = _sector_ids_from_angles(angles)
    return {
        "model": "circle",
        "centre_xy": centre,
        "radius": float(radius),
        "equivalent_radius": float(radius),
        "axis_ratio": 1.0,
        "fit_residual": fit_residual,
        "arc_coverage": float(len(np.unique(sector_ids)) / 32.0),
        "sector_ids": sector_ids,
        "inlier_xy": inlier_xy,
    }


def _fit_ellipse_robust(
    xy: np.ndarray,
    max_fit_residual_ratio: float,
) -> Optional[Dict[str, Any]]:
    """Robust ellipse fit with one trimming/refit pass."""
    if len(xy) < 8:
        return None

    try:
        centre, major_axis, minor_axis, theta = _fit_ellipse_xy(xy)
    except Exception:
        return None

    if not np.isfinite(major_axis) or not np.isfinite(minor_axis):
        return None
    if major_axis <= 1e-12 or minor_axis <= 1e-12:
        return None

    rotated = _rotate_points(xy, centre, theta)
    residuals = np.abs(np.sqrt((rotated[:, 0] / major_axis) ** 2 + (rotated[:, 1] / minor_axis) ** 2) - 1.0)
    threshold = max(max_fit_residual_ratio * 1.5, 0.05)
    inliers = residuals <= threshold
    if int(inliers.sum()) >= max(8, len(xy) // 2):
        try:
            centre, major_axis, minor_axis, theta = _fit_ellipse_xy(xy[inliers])
        except Exception:
            return None
        rotated = _rotate_points(xy, centre, theta)
        residuals = np.abs(np.sqrt((rotated[:, 0] / major_axis) ** 2 + (rotated[:, 1] / minor_axis) ** 2) - 1.0)
        inliers = residuals <= max(max_fit_residual_ratio * 1.5, 0.05)

    if int(inliers.sum()) < 8:
        return None

    inlier_xy = xy[inliers]
    rotated_inliers = _rotate_points(inlier_xy, centre, theta)
    angles = np.arctan2(rotated_inliers[:, 1] / minor_axis, rotated_inliers[:, 0] / major_axis)
    sector_ids = _sector_ids_from_angles(angles)
    fit_residual = float(np.median(np.abs(np.sqrt((rotated_inliers[:, 0] / major_axis) ** 2 + (rotated_inliers[:, 1] / minor_axis) ** 2) - 1.0)))
    return {
        "model": "ellipse",
        "centre_xy": centre,
        "radius": float(np.sqrt(major_axis * minor_axis)),
        "equivalent_radius": float(np.sqrt(major_axis * minor_axis)),
        "axis_ratio": float(major_axis / max(minor_axis, 1e-12)),
        "fit_residual": fit_residual,
        "arc_coverage": float(len(np.unique(sector_ids)) / 32.0),
        "sector_ids": sector_ids,
        "inlier_xy": inlier_xy,
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "theta": float(theta),
    }


def _score_arc_model(
    model_fit: Dict[str, Any],
    total_points: int,
    config: "TrunkExtractionConfig",
) -> float:
    """Score a fitted arc model; higher is better."""
    inlier_fraction = float(len(model_fit["inlier_xy"]) / max(total_points, 1))
    residual_term = 1.0 - min(
        model_fit["fit_residual"] / max(config.basal_anchor_max_fit_residual_ratio, 1e-12),
        1.0,
    )
    axis_term = 1.0 - min(
        max(model_fit["axis_ratio"] - 1.0, 0.0) / max(config.basal_anchor_max_axis_ratio - 1.0, 1e-12),
        1.0,
    )
    return float(
        0.45 * model_fit["arc_coverage"]
        + 0.30 * residual_term
        + 0.15 * inlier_fraction
        + 0.10 * axis_term
    )


def _best_arc_model_for_component(
    xy: np.ndarray,
    config: "TrunkExtractionConfig",
    min_radius: float,
    max_radius: float,
) -> Optional[Dict[str, Any]]:
    """Evaluate robust circle/ellipse models for a basal component and keep the best valid one."""
    candidates = []
    for fit in (
        _fit_circle_robust(xy, config.basal_anchor_max_fit_residual_ratio),
        _fit_ellipse_robust(xy, config.basal_anchor_max_fit_residual_ratio),
    ):
        if fit is None:
            continue
        radius = fit["equivalent_radius"]
        if not (min_radius <= radius <= max_radius):
            continue
        if fit["arc_coverage"] < config.basal_anchor_min_arc_coverage:
            continue
        if fit["fit_residual"] > config.basal_anchor_max_fit_residual_ratio:
            continue
        if fit["axis_ratio"] > config.basal_anchor_max_axis_ratio:
            continue
        fit["score"] = _score_arc_model(fit, len(xy), config)
        fit["inlier_fraction"] = float(len(fit["inlier_xy"]) / max(len(xy), 1))
        candidates.append(fit)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item["score"], item["arc_coverage"], -item["fit_residual"]), reverse=True)
    best = candidates[0]
    if best["model"] == "ellipse":
        circle = next((item for item in candidates if item["model"] == "circle"), None)
        if circle is not None and (
            best["axis_ratio"] <= 1.10
            or circle["score"] >= best["score"] * 0.97
        ):
            best = circle
    return best


def _collect_global_basal_slice_candidates(
    xyz: np.ndarray,
    config: "TrunkExtractionConfig",
) -> List[Dict[str, Any]]:
    """Collect per-slice basal arc candidates without PCA guidance."""
    z_upper = min(config.basal_anchor_max_height, config.stripe_lower_limit - config.basal_anchor_gap_to_stripe)
    z_lower = config.basal_anchor_min_height
    if z_upper <= z_lower:
        return []

    slice_centres = np.arange(z_lower, z_upper + 1e-9, config.basal_anchor_slice_step, dtype=np.float64)
    slice_candidates: List[Dict[str, Any]] = []
    min_radius = (config.dbh_min / 2.0) * config.basal_anchor_radius_min_factor
    max_radius = (config.dbh_max / 2.0) * config.basal_anchor_radius_max_factor
    min_samples = max(5, min(config.basal_anchor_min_points // 4, config.basal_anchor_min_points))

    for slice_idx, z_center in enumerate(slice_centres):
        slice_mask = np.abs(xyz[:, 2] - z_center) <= config.basal_anchor_slice_half_width
        if not np.any(slice_mask):
            continue

        slice_pts = xyz[slice_mask]
        if len(slice_pts) < config.basal_anchor_min_points:
            continue

        labels = _dbscan_cluster(slice_pts, eps=config.basal_anchor_cluster_eps, min_samples=min_samples)
        for lbl in sorted(set(labels) - {-1}):
            component = slice_pts[labels == lbl]
            if len(component) < config.basal_anchor_min_points:
                continue
            best_model = _best_arc_model_for_component(
                component[:, :2],
                config,
                min_radius,
                max_radius,
            )
            if best_model is None:
                continue
            best_model["z_center"] = float(z_center)
            best_model["slice_id"] = int(slice_idx)
            best_model["n_points"] = int(len(component))
            slice_candidates.append(best_model)

    return slice_candidates


def _build_basal_tracks(
    slice_candidates: List[Dict[str, Any]],
    config: "TrunkExtractionConfig",
) -> List[List[Dict[str, Any]]]:
    """Link slice-level basal candidates into multi-slice global tracks."""
    if not slice_candidates:
        return []

    tracks: List[List[Dict[str, Any]]] = []
    max_gap_z = config.basal_anchor_slice_step * 2.5
    max_center_step = max(config.basal_anchor_max_center_drift * 1.25, config.basal_anchor_cluster_eps * 2.0)
    radius_step_tol = max(config.basal_anchor_max_radius_cv * 2.0, 0.35)

    for candidate in sorted(slice_candidates, key=lambda item: item["z_center"]):
        best_track_idx = None
        best_cost = np.inf
        for idx, track in enumerate(tracks):
            prev = track[-1]
            if candidate["z_center"] <= prev["z_center"]:
                continue
            if candidate["z_center"] - prev["z_center"] > max_gap_z:
                continue
            center_dist = float(np.linalg.norm(candidate["centre_xy"] - prev["centre_xy"]))
            if center_dist > max_center_step:
                continue
            radius_ratio = candidate["equivalent_radius"] / max(prev["equivalent_radius"], 1e-12)
            radius_delta = abs(radius_ratio - 1.0)
            if radius_delta > radius_step_tol:
                continue
            cost = (
                center_dist / max(max_center_step, 1e-12)
                + radius_delta
                - 0.10 * candidate["score"]
            )
            if cost < best_cost:
                best_cost = cost
                best_track_idx = idx

        if best_track_idx is None:
            tracks.append([candidate])
        else:
            tracks[best_track_idx].append(candidate)

    return tracks


def _summarize_basal_track(track: List[Dict[str, Any]], track_id: int) -> Dict[str, Any]:
    """Summarize a basal track for ranking and matching."""
    centres = np.vstack([item["centre_xy"] for item in track]).astype(np.float64)
    radii = np.array([item["equivalent_radius"] for item in track], dtype=np.float64)
    centre_xy = np.median(centres, axis=0)
    center_std = float(np.mean(np.linalg.norm(centres - centre_xy, axis=1)))
    radius_cv = float(np.std(radii) / max(np.mean(radii), 1e-12))
    sector_ids = np.concatenate([item["sector_ids"] for item in track if len(item["sector_ids"]) > 0])
    arc_coverage = float(len(np.unique(sector_ids)) / 32.0) if sector_ids.size > 0 else 0.0
    fit_residual = float(np.mean([item["fit_residual"] for item in track]))
    axis_ratio = float(np.mean([item["axis_ratio"] for item in track]))
    model = "ellipse" if sum(item["model"] == "ellipse" for item in track) > len(track) / 2 else "circle"
    return {
        "track_id": int(track_id),
        "centre_xy": centre_xy,
        "z_anchor": float(min(item["z_center"] for item in track)),
        "equivalent_radius": float(np.median(radii)),
        "axis_ratio": axis_ratio,
        "arc_coverage": arc_coverage,
        "fit_residual": fit_residual,
        "center_std": center_std,
        "radius_cv": radius_cv,
        "support_slices": int(len(track)),
        "model": model,
        "raw_score": float(np.mean([item["score"] for item in track])),
    }


def _normalize_metric(values: np.ndarray, invert: bool = False) -> np.ndarray:
    """Min-max normalize a metric; returns ones for a constant finite metric."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    finite_values = values[finite]
    lo = float(finite_values.min())
    hi = float(finite_values.max())
    if hi - lo <= 1e-12:
        out[finite] = 1.0
    else:
        out[finite] = (finite_values - lo) / (hi - lo)
    if invert:
        out[finite] = 1.0 - out[finite]
    return out


def _finalize_track_scores(tracks: List[Dict[str, Any]]) -> None:
    """Attach a normalized track score to validated basal tracks."""
    if not tracks:
        return

    support_norm = _normalize_metric(np.array([track["support_slices"] for track in tracks], dtype=np.float64))
    coverage_norm = _normalize_metric(np.array([track["arc_coverage"] for track in tracks], dtype=np.float64))
    residual_norm = _normalize_metric(np.array([track["fit_residual"] for track in tracks], dtype=np.float64), invert=True)
    center_norm = _normalize_metric(np.array([track["center_std"] for track in tracks], dtype=np.float64), invert=True)
    radius_norm = _normalize_metric(np.array([track["radius_cv"] for track in tracks], dtype=np.float64), invert=True)

    for idx, track in enumerate(tracks):
        track["track_score"] = float(
            0.30 * support_norm[idx]
            + 0.25 * coverage_norm[idx]
            + 0.20 * residual_norm[idx]
            + 0.15 * center_norm[idx]
            + 0.10 * radius_norm[idx]
        )


def _detect_basal_tracks(
    xyz: np.ndarray,
    config: "TrunkExtractionConfig",
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Detect global basal tracks in the low band without PCA guidance."""
    slice_candidates = _collect_global_basal_slice_candidates(xyz, config)
    tracks = _build_basal_tracks(slice_candidates, config)
    track_summaries = [_summarize_basal_track(track, track_id=idx) for idx, track in enumerate(tracks)]
    total_candidates = int(len(track_summaries))
    validated_tracks = [
        summary
        for summary in track_summaries
        if summary["support_slices"] >= config.basal_anchor_min_support_slices
        and summary["center_std"] <= config.basal_anchor_max_center_drift
        and summary["radius_cv"] <= config.basal_anchor_max_radius_cv
        and summary["arc_coverage"] >= config.basal_anchor_min_arc_coverage
        and summary["fit_residual"] <= config.basal_anchor_max_fit_residual_ratio
        and summary["axis_ratio"] <= config.basal_anchor_max_axis_ratio
    ]
    _finalize_track_scores(validated_tracks)
    return validated_tracks, total_candidates, int(len(validated_tracks))


def _pair_track_to_axis(
    track: Dict[str, Any],
    axis: Dict[str, Any],
    config: "TrunkExtractionConfig",
) -> Optional[Dict[str, Any]]:
    """Evaluate whether a basal track is compatible with a stripe tree candidate."""
    stripe_centroid = np.asarray(axis["centroid"], dtype=np.float64)
    line_point = np.array([track["centre_xy"][0], track["centre_xy"][1], track["z_anchor"]], dtype=np.float64)
    vector = stripe_centroid - line_point
    norm = np.linalg.norm(vector)
    if norm <= 1e-12 or vector[2] <= 0.0:
        return None

    direction = vector / norm
    tilt_deg = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))))
    if tilt_deg > config.basal_anchor_match_max_tilt_deg:
        return None

    xy_distance = float(np.linalg.norm(track["centre_xy"] - stripe_centroid[:2]))
    if xy_distance > config.basal_anchor_match_max_xy_distance:
        return None

    stripe_diameter = float(axis.get("stripe_diameter", np.nan))
    stripe_radius = stripe_diameter / 2.0 if np.isfinite(stripe_diameter) else np.nan
    if not np.isfinite(stripe_radius) or stripe_radius <= 0.0:
        return None

    radius_ratio = float(track["equivalent_radius"] / stripe_radius)
    if not (config.basal_anchor_match_radius_ratio_min <= radius_ratio <= config.basal_anchor_match_radius_ratio_max):
        return None

    xy_distance_norm = min(xy_distance / max(config.basal_anchor_match_max_xy_distance, 1e-12), 1.0)
    radius_error_scale = max(
        abs(config.basal_anchor_match_radius_ratio_max - 1.0),
        abs(1.0 - config.basal_anchor_match_radius_ratio_min),
        1e-12,
    )
    radius_ratio_error_norm = min(abs(radius_ratio - 1.0) / radius_error_scale, 1.0)
    match_score = float(
        0.45 * track["track_score"]
        + 0.35 * (1.0 - xy_distance_norm)
        + 0.20 * (1.0 - radius_ratio_error_norm)
    )
    return {
        "track_id": int(track["track_id"]),
        "tree_id": int(axis["tree_id"]),
        "track": track,
        "tree_axis": axis,
        "line_point": line_point,
        "direction": direction,
        "xy_distance": xy_distance,
        "radius_ratio": radius_ratio,
        "tilt_deg": tilt_deg,
        "match_score": match_score,
    }


def _match_basal_tracks_to_axes(
    tracks: List[Dict[str, Any]],
    axes: List[Dict[str, Any]],
    config: "TrunkExtractionConfig",
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Match global basal tracks to stripe trees with a 1-to-1 exclusive assignment."""
    if not tracks or not axes:
        return {}, {}

    from scipy.optimize import linear_sum_assignment

    huge = 1e6
    pair_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
    best_by_tree: Dict[int, Dict[str, Any]] = {}
    cost_matrix = np.full((len(tracks), len(axes)), huge, dtype=np.float64)

    for track_idx, track in enumerate(tracks):
        for axis_idx, axis in enumerate(axes):
            pair = _pair_track_to_axis(track, axis, config)
            if pair is None:
                continue
            pair_lookup[(track_idx, axis_idx)] = pair
            cost_matrix[track_idx, axis_idx] = 1.0 - pair["match_score"]
            tree_id = pair["tree_id"]
            if tree_id not in best_by_tree or pair["match_score"] > best_by_tree[tree_id]["match_score"]:
                best_by_tree[tree_id] = pair

    if not pair_lookup:
        return best_by_tree, {}

    rows, cols = linear_sum_assignment(cost_matrix)
    assigned_by_tree: Dict[int, Dict[str, Any]] = {}
    for row, col in zip(rows, cols):
        if cost_matrix[row, col] >= huge:
            continue
        pair = pair_lookup[(row, col)]
        assigned_by_tree[pair["tree_id"]] = pair

    return best_by_tree, assigned_by_tree


def _anchor_metadata_from_pair(
    pair: Dict[str, Any],
    total_candidates: int,
    validated_candidates: int,
    applied: bool,
) -> Dict[str, Any]:
    """Convert a matched or best candidate pair into per-tree audit metadata."""
    track = pair["track"]
    circularity = 1.0 / max(track["axis_ratio"], 1e-12)
    return {
        "basal_anchor_found": True,
        "basal_anchor_applied": bool(applied),
        "basal_anchor_candidates_total": int(total_candidates),
        "basal_anchor_candidates_validated": int(validated_candidates),
        "basal_anchor_x": float(track["centre_xy"][0]),
        "basal_anchor_y": float(track["centre_xy"][1]),
        "basal_anchor_z": float(track["z_anchor"]),
        "basal_anchor_radius": float(track["equivalent_radius"]),
        "basal_anchor_circularity": float(circularity),
        "basal_anchor_sector_pct": float(track["arc_coverage"]),
        "basal_anchor_inner_fraction": float("nan"),
        "basal_anchor_support_slices": int(track["support_slices"]),
        "basal_anchor_center_offset": float(pair["xy_distance"]),
        "basal_anchor_model": track["model"],
        "basal_anchor_arc_coverage": float(track["arc_coverage"]),
        "basal_anchor_fit_residual": float(track["fit_residual"]),
        "basal_anchor_track_center_std": float(track["center_std"]),
        "basal_anchor_track_radius_cv": float(track["radius_cv"]),
        "basal_anchor_track_id": int(track["track_id"]),
        "basal_anchor_track_score": float(track["track_score"]),
        "basal_anchor_match_passed": bool(applied),
        "basal_anchor_match_score": float(pair["match_score"]),
        "basal_anchor_match_xy_distance": float(pair["xy_distance"]),
        "basal_anchor_match_radius_ratio": float(pair["radius_ratio"]),
        "basal_anchor_match_tilt_deg": float(pair["tilt_deg"]),
        "axis_direction_delta_deg": 0.0,
    }


def _refine_axes_with_basal_anchor(
    xyz: np.ndarray,
    axes: List[Dict[str, Any]],
    config: "TrunkExtractionConfig",
) -> List[Dict[str, Any]]:
    """Refine preliminary PCA axes using global basal tracks and exclusive matching."""
    tracks, total_candidates, validated_candidates = _detect_basal_tracks(xyz, config)
    best_by_tree, assigned_by_tree = _match_basal_tracks_to_axes(tracks, axes, config)

    refined_axes: List[Dict[str, Any]] = []
    for axis in axes:
        tree_id = int(axis["tree_id"])
        refined = dict(axis)
        refined["axis_source"] = "pca"
        refined["line_point"] = np.asarray(axis["centroid"], dtype=np.float64)
        refined.update(_empty_basal_anchor(total_candidates, validated_candidates))

        best_pair = best_by_tree.get(tree_id)
        if best_pair is not None:
            refined.update(
                _anchor_metadata_from_pair(
                    best_pair,
                    total_candidates=total_candidates,
                    validated_candidates=validated_candidates,
                    applied=False,
                )
            )

        assigned_pair = assigned_by_tree.get(tree_id)
        if assigned_pair is None:
            refined_axes.append(refined)
            continue

        line_point = np.asarray(assigned_pair["line_point"], dtype=np.float64)
        direction = np.asarray(assigned_pair["direction"], dtype=np.float64)
        if direction[2] <= 0.0:
            refined_axes.append(refined)
            continue

        old_dir = np.asarray(axis["direction"], dtype=np.float64)
        old_dir = old_dir / (np.linalg.norm(old_dir) + 1e-12)
        cos_delta = np.clip(np.dot(direction, old_dir), -1.0, 1.0)

        refined.update(
            _anchor_metadata_from_pair(
                assigned_pair,
                total_candidates=total_candidates,
                validated_candidates=validated_candidates,
                applied=True,
            )
        )
        refined["line_point"] = line_point
        refined["direction"] = direction
        refined["axis_source"] = "basal_anchor"
        refined["axis_direction_delta_deg"] = float(np.degrees(np.arccos(cos_delta)))
        refined_axes.append(refined)

    return refined_axes


def _compute_axes_pca(
    xyz: np.ndarray,
    labels: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Compute tree axis for each cluster using PCA.
    Returns list of axis dicts with centroid, direction, id.
    """
    axes = []
    unique_labels = sorted(set(labels) - {-1})
    for lbl in unique_labels:
        mask = labels == lbl
        pts = xyz[mask]
        centroid = pts.mean(axis=0)
        geometry = _summarize_cluster_geometry(pts)

        # PCA to get principal direction
        centered = pts - centroid
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Largest eigenvalue = direction of most variance = stem axis
        axis_dir = eigenvectors[:, np.argmax(eigenvalues)]

        # Force axis to point upward
        if axis_dir[2] < 0:
            axis_dir = -axis_dir

        axes.append({
            "tree_id": int(lbl),
            "centroid": centroid,
            "direction": axis_dir,
            "n_points": int(mask.sum()),
            "z_min": float(pts[:, 2].min()),
            "z_max": float(pts[:, 2].max()),
            **geometry,
        })
    return axes


def _build_tree_centerlines(
    xyz: np.ndarray,
    tree_ids: np.ndarray,
    tree_axes: List[Dict[str, Any]],
    slab_step: float,
    min_points_per_slab: int,
) -> List[Dict[str, Any]]:
    """
    Build a piecewise centerline (polyline) per tree by tracking the median XY
    position of assigned points across vertical slabs.

    Improvement 1, Phase 1A: this enriches each ``tree_axes`` entry with a
    ``centerline`` field (a (K, 3) ndarray of control points sorted by Z, or
    ``None`` if the tree has too few points to build a meaningful polyline).
    Downstream consumers (point assignment, stem cleaning, sectioning) are not
    yet aware of this field — they still use the straight-line axis. Future
    phases (1B, 1C) will switch them to the polyline incrementally.

    Parameters
    ----------
    xyz : (N, 3) ndarray
        Full input point cloud.
    tree_ids : (N,) int ndarray
        Per-point tree assignment from ``_assign_points_to_axes`` (-1 for
        unassigned points).
    tree_axes : list of dict
        Tree axis records produced by ``_compute_axes_pca`` /
        ``_refine_axes_with_basal_anchor``. Mutated in place.
    slab_step : float
        Vertical spacing between control points, in metres.
    min_points_per_slab : int
        Minimum number of assigned points a slab must contain to produce a
        control point. Slabs below this threshold are skipped (the polyline
        has gaps).

    Returns
    -------
    The same ``tree_axes`` list, with each entry's ``centerline`` field set.
    """
    for ax in tree_axes:
        tid = ax["tree_id"]
        mask = tree_ids == tid
        if not mask.any():
            ax["centerline"] = None
            continue
        pts = xyz[mask]
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())
        if z_max - z_min < slab_step:
            ax["centerline"] = None
            continue
        n_slabs = max(1, int(np.ceil((z_max - z_min) / slab_step)))
        slab_indices = np.minimum(
            ((pts[:, 2] - z_min) / slab_step).astype(np.int64),
            n_slabs - 1,
        )
        control_points: List[List[float]] = []
        for s in range(n_slabs):
            slab_mask = slab_indices == s
            if int(slab_mask.sum()) < min_points_per_slab:
                continue
            slab_pts = pts[slab_mask]
            cx = float(np.median(slab_pts[:, 0]))
            cy = float(np.median(slab_pts[:, 1]))
            cz = float(np.median(slab_pts[:, 2]))
            control_points.append([cx, cy, cz])
        if len(control_points) < 2:
            ax["centerline"] = None
        else:
            ax["centerline"] = np.asarray(control_points, dtype=np.float64)
    return tree_axes


def _assign_points_to_axes(
    xyz: np.ndarray,
    axes: List[Dict[str, Any]],
    max_distance: float,
) -> np.ndarray:
    """
    Assign each point to the nearest tree axis.

    Distance is computed as point-to-line distance in 3D.
    Points beyond max_distance remain unassigned (-1).
    """
    n_points = len(xyz)
    tree_ids = np.full(n_points, -1, dtype=np.int32)

    if not axes:
        return tree_ids

    # Compute distance from each point to each axis
    best_dist = np.full(n_points, np.inf)

    for ax in axes:
        c = np.asarray(ax.get("line_point", ax["centroid"]), dtype=np.float64)
        d = ax["direction"]
        d_norm = d / (np.linalg.norm(d) + 1e-12)

        # Vector from centroid to each point
        v = xyz - c

        # Project onto axis direction
        proj_len = v @ d_norm

        # Closest point on line
        closest = c + np.outer(proj_len, d_norm)

        # Perpendicular distance
        dist = np.linalg.norm(xyz - closest, axis=1)

        # Update assignments for closer axes
        closer = dist < best_dist
        tree_ids[closer] = ax["tree_id"]
        best_dist[closer] = dist[closer]

    # Unassign points beyond max_distance
    tree_ids[best_dist > max_distance] = -1

    return tree_ids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_trunks(
    xyz: np.ndarray,
    config: Optional[TrunkExtractionConfig] = None,
    verbose: bool = False,
) -> TrunkExtractionResult:
    """
    Extract tree trunks from a height-normalized point cloud.

    Follows the dendromatics/3DFin pipeline:
      1. Extract horizontal stripe [lower_limit, upper_limit]
      2. Voxelize stripe
      3. Compute verticality (pgeof C++ backend)
      4. Filter by verticality threshold
      5. DBSCAN clustering in 2D (XY)
      6. Filter small clusters and those with insufficient height range
      7. Iterate steps 2-6 (peeling)
      8. Compute tree axes via PCA
      9. Assign ALL points to nearest axis

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) height-normalized point cloud (ground removed, Z = height above ground).
    config : TrunkExtractionConfig, optional
        Configuration with field and algorithm parameters. Uses defaults if None.
    verbose : bool
        Print progress information.

    Returns
    -------
    TrunkExtractionResult
        Contains trunk_mask, tree_ids, tree axes, etc.
    """
    if config is None:
        config = TrunkExtractionConfig()

    n_total = len(xyz)
    if verbose:
        print(f"Trunk extraction: {n_total:,} points")
        print(f"  Stripe: {config.stripe_lower_limit}m – {config.stripe_upper_limit}m")
        print(f"  Verticality threshold: {config.verticality_threshold}")
        print(f"  Peeling iterations: {config.peeling_iterations}")
        print(f"  Min cluster points: {config.min_cluster_points}")

    # Auto-compute DBSCAN eps from voxel resolution (as dendromatics does)
    eps = config.dbscan_eps
    if eps is None:
        eps = config.voxel_resolution_xy * np.sqrt(3)

    stripe_height = config.stripe_upper_limit - config.stripe_lower_limit

    # ======================================================================
    # Step 1: Extract stripe
    # ======================================================================
    stripe_xyz, stripe_mask, stripe_indices = _extract_stripe(
        xyz, config.stripe_lower_limit, config.stripe_upper_limit,
    )
    if verbose:
        print(f"  Stripe points: {len(stripe_xyz):,}")

    if len(stripe_xyz) < config.min_cluster_points:
        if verbose:
            print("  WARNING: Not enough points in stripe for clustering")
        return TrunkExtractionResult(
            trunk_mask=np.zeros(n_total, dtype=bool),
            tree_ids=np.full(n_total, -1, dtype=np.int32),
            n_trees=0,
            tree_axes=[],
            cluster_points=np.empty((0, 3)),
            config=config,
        )

    # ======================================================================
    # Steps 2-6: Iterative verticality peeling
    # ======================================================================
    current_points = stripe_xyz.copy()

    for iteration in range(config.peeling_iterations):
        if verbose:
            print(f"  Peeling iteration {iteration + 1}/{config.peeling_iterations}")

        # Step 2: Voxelize
        centroids, pt_to_vox, n_vox = voxelize_cloud(
            current_points,
            resolution_xy=config.voxel_resolution_xy,
            resolution_z=config.voxel_resolution_z,
        )
        if verbose:
            print(f"    Voxels: {n_vox:,}")

        # Step 3: Compute verticality on voxel centroids
        vert_voxels = compute_verticality(
            centroids,
            scale=config.verticality_scale,
            voxel_resolution_xy=config.voxel_resolution_xy,
            voxel_resolution_z=config.voxel_resolution_z,
        )

        # Step 4: Filter voxels by verticality (operate on centroids, not raw points)
        vox_vert_mask = vert_voxels >= config.verticality_threshold
        n_vox_before = n_vox

        # Step 4b: Optional multi-scale geometric pre-filter (Improvement 4).
        # Narrows vox_vert_mask further by rejecting voxels with chaotic
        # PCA spread (sphericity > sphericity_max) and/or low local point
        # density (points/m³ < density_min). Both filters share the voxel
        # centroids already computed in Step 2, so the only added cost is
        # one extra pgeof call (sphericity).
        if config.sphericity_max < 1.0 or config.density_min > 0.0:
            n_after_vert = int(vox_vert_mask.sum())
            if config.sphericity_max < 1.0:
                sph_voxels = compute_sphericity(
                    centroids,
                    scale=config.verticality_scale,
                    voxel_resolution_xy=config.voxel_resolution_xy,
                    voxel_resolution_z=config.voxel_resolution_z,
                )
                vox_vert_mask = vox_vert_mask & (sph_voxels <= config.sphericity_max)
            if config.density_min > 0.0:
                voxel_volume = (
                    config.voxel_resolution_xy
                    * config.voxel_resolution_xy
                    * config.voxel_resolution_z
                )
                voxel_counts = np.bincount(pt_to_vox, minlength=n_vox)
                voxel_density = voxel_counts / voxel_volume
                vox_vert_mask = vox_vert_mask & (voxel_density >= config.density_min)
            if verbose:
                n_after_prefilter = int(vox_vert_mask.sum())
                print(
                    f"    Pre-filter (sphericity≤{config.sphericity_max}, "
                    f"density≥{config.density_min}): "
                    f"{n_after_vert:,} → {n_after_prefilter:,} voxels "
                    f"(rejected {n_after_vert - n_after_prefilter:,})"
                )

        # Map voxel filter back to raw points
        pt_vert_mask = vox_vert_mask[pt_to_vox]
        n_before = len(current_points)
        current_points = current_points[pt_vert_mask]

        # Also filter centroids and rebuild mapping for DBSCAN
        kept_centroids = centroids[vox_vert_mask]

        if verbose:
            print(f"    After verticality filter: {len(current_points):,} pts "
                  f"(removed {n_before - len(current_points):,}), "
                  f"{len(kept_centroids):,} voxels")

        if len(kept_centroids) < 10:
            if verbose:
                print(f"    Not enough voxels after filtering, stopping peeling")
            break

        # Step 5: DBSCAN clustering on VOXEL CENTROIDS (2D) — memory-safe
        vox_labels = _dbscan_cluster(kept_centroids, eps=eps)

        # Step 6: Filter small voxel clusters
        vox_labels = _filter_small_clusters(
            vox_labels, max(config.min_cluster_points // 50, 10),
        )

        # Filter voxel clusters by height range
        vox_labels = _filter_by_height_range(
            kept_centroids, vox_labels,
            stripe_height, config.height_range,
        )

        # Map voxel labels back to raw points
        # Rebuild pt_to_vox for the filtered subset
        old_to_new_vox = np.full(n_vox, -1, dtype=np.int64)
        old_indices = np.where(vox_vert_mask)[0]
        for new_idx, old_idx in enumerate(old_indices):
            old_to_new_vox[old_idx] = new_idx
        
        # Map each raw point to its new voxel index
        pt_to_new_vox = old_to_new_vox[pt_to_vox[pt_vert_mask]]
        
        # Get cluster label for each raw point from its voxel
        cluster_labels = vox_labels[pt_to_new_vox]

        # Keep only clustered points for next iteration
        valid = cluster_labels >= 0
        current_points = current_points[valid]
        cluster_labels = cluster_labels[valid]

        n_clusters = len(set(cluster_labels) - {-1})
        if verbose:
            print(f"    Clusters kept: {n_clusters}, points: {len(current_points):,}")

        if len(current_points) < config.min_cluster_points:
            break

    # ======================================================================
    # Step 7: Relabel and compute axes
    # ======================================================================
    if len(current_points) == 0:
        if verbose:
            print("  No trunk clusters found")
        return TrunkExtractionResult(
            trunk_mask=np.zeros(n_total, dtype=bool),
            tree_ids=np.full(n_total, -1, dtype=np.int32),
            n_trees=0,
            tree_axes=[],
            cluster_points=np.empty((0, 3)),
            config=config,
        )

    # Final DBSCAN on voxel centroids to ensure clean labels
    final_centroids, final_pt_to_vox, final_n_vox = voxelize_cloud(
        current_points,
        resolution_xy=config.voxel_resolution_xy,
        resolution_z=config.voxel_resolution_z,
    )
    final_vox_labels = _dbscan_cluster(final_centroids, eps=eps)
    final_vox_labels = _filter_small_clusters(
        final_vox_labels, max(config.min_cluster_points // 50, 10),
    )
    # Map voxel labels to raw points
    final_labels = final_vox_labels[final_pt_to_vox]
    valid = final_labels >= 0
    current_points = current_points[valid]
    final_labels = final_labels[valid]
    final_labels = _relabel_clusters(final_labels)

    # ======================================================================
    # Step 7a: Validate stripe clusters (cross-section geometry)
    # ======================================================================
    if verbose:
        print(f"  Cluster validation (cross-section geometry):")
    final_labels = _validate_stripe_clusters(
        current_points, final_labels, config, verbose=verbose,
    )
    # Remove rejected points and relabel
    valid_after_validation = final_labels >= 0
    current_points = current_points[valid_after_validation]
    final_labels = final_labels[valid_after_validation]
    final_labels = _relabel_clusters(final_labels)

    if len(current_points) == 0:
        if verbose:
            print("  No valid clusters after validation")
        return TrunkExtractionResult(
            trunk_mask=np.zeros(n_total, dtype=bool),
            tree_ids=np.full(n_total, -1, dtype=np.int32),
            n_trees=0,
            tree_axes=[],
            cluster_points=np.empty((0, 3)),
            config=config,
        )

    # Compute preliminary PCA axes
    tree_axes = _compute_axes_pca(current_points, final_labels)
    if config.axis_refinement_mode == "basal_anchor":
        tree_axes = _refine_axes_with_basal_anchor(xyz, tree_axes, config)
    n_trees = len(tree_axes)

    if verbose:
        print(f"  Trees detected: {n_trees}")
        for ax in tree_axes:
            print(f"    Tree {ax['tree_id']}: {ax['n_points']} pts, "
                  f"z=[{ax['z_min']:.1f}, {ax['z_max']:.1f}]m")

    # ======================================================================
    # Step 8: Assign ALL points to nearest axis
    # ======================================================================
    tree_ids = _assign_points_to_axes(xyz, tree_axes, config.max_axis_distance)

    # Step 8b: Build piecewise centerlines (Improvement 1, Phase 1A).
    # Adds a 'centerline' field to each tree_axes entry. Currently informational
    # only — point assignment, stem cleaning and sectioning still use the
    # straight-line axis. The polyline is exposed for visualisation in Module 9
    # and for the downstream phases that will switch to it.
    tree_axes = _build_tree_centerlines(
        xyz,
        tree_ids,
        tree_axes,
        slab_step=config.centerline_slab_step,
        min_points_per_slab=config.centerline_min_points_per_slab,
    )
    if verbose:
        n_with_centerline = sum(1 for ax in tree_axes if ax.get("centerline") is not None)
        print(
            f"  Centerlines built: {n_with_centerline}/{len(tree_axes)} trees "
            f"(slab={config.centerline_slab_step}m, "
            f"min_pts={config.centerline_min_points_per_slab})"
        )

    # Build trunk mask: points within stem_search_radius of their axis
    trunk_mask = np.zeros(n_total, dtype=bool)
    for ax in tree_axes:
        c = np.asarray(ax.get("line_point", ax["centroid"]), dtype=np.float64)
        d = ax["direction"]
        d_norm = d / (np.linalg.norm(d) + 1e-12)

        assigned = tree_ids == ax["tree_id"]
        pts = xyz[assigned]

        v = pts - c
        proj_len = v @ d_norm
        closest = c + np.outer(proj_len, d_norm)
        perp_dist = np.linalg.norm(pts - closest, axis=1)

        # Within stem search radius = trunk
        is_trunk = perp_dist <= config.stem_search_radius
        assigned_indices = np.where(assigned)[0]
        trunk_mask[assigned_indices[is_trunk]] = True

    n_trunk = trunk_mask.sum()
    if verbose:
        print(f"  Trunk points: {n_trunk:,} ({100 * n_trunk / n_total:.1f}%)")
        print(f"  Assigned points: {(tree_ids >= 0).sum():,} ({100 * (tree_ids >= 0).sum() / n_total:.1f}%)")

    return TrunkExtractionResult(
        trunk_mask=trunk_mask,
        tree_ids=tree_ids,
        n_trees=n_trees,
        tree_axes=tree_axes,
        cluster_points=current_points,
        config=config,
    )
