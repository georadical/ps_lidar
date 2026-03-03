"""
Trunk Extraction Module — Brick 7

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
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .features import voxelize_cloud, compute_verticality


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
        })
    return axes


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
        c = ax["centroid"]
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

    # Compute PCA axes
    tree_axes = _compute_axes_pca(current_points, final_labels)
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

    # Build trunk mask: points within stem_search_radius of their axis
    trunk_mask = np.zeros(n_total, dtype=bool)
    for ax in tree_axes:
        c = ax["centroid"]
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
