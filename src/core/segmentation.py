"""
Tree Segmentation Module

Provides functions for individualizing trees from point clouds,
assigning unique tree_id to each point including stems, branches, and crowns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

import pgeof
from pgeof import EFeatureID
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


@dataclass
class TreeInfo:
    """Metadata for a single detected tree."""
    tree_id: int
    centroid: np.ndarray  # (3,) XYZ
    n_points: int
    height_max: float
    height_min: float
    axis_direction: np.ndarray  # (3,) PCA1 direction
    axis_deviation_deg: float  # Deviation from vertical


@dataclass
class TreeSegmentationResult:
    """Result of tree segmentation operation."""
    tree_ids: np.ndarray  # (N,) int array with tree_id per point
    n_trees: int
    unassigned_count: int
    tree_info: List[TreeInfo]


def voxelize_cloud(
    xyz: np.ndarray,
    resolution: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelize point cloud using grid-based binning.
    
    Args:
        xyz: (N, 3) array of points.
        resolution: Voxel size in meters.
    
    Returns:
        voxel_xyz: (M, 3) voxel centroids.
        voxel_indices: (M,) indices of first point in each voxel.
        point_to_voxel: (N,) mapping from original points to voxel index.
    """
    # Compute voxel keys
    voxel_keys = np.floor(xyz / resolution).astype(np.int32)
    
    # Find unique voxels
    _, voxel_indices, point_to_voxel = np.unique(
        voxel_keys, axis=0, return_index=True, return_inverse=True
    )
    
    # Compute voxel centroids by averaging points in each voxel
    n_voxels = len(voxel_indices)
    voxel_xyz = np.zeros((n_voxels, 3), dtype=np.float64)
    voxel_counts = np.zeros(n_voxels, dtype=np.int32)
    
    np.add.at(voxel_xyz, point_to_voxel, xyz)
    np.add.at(voxel_counts, point_to_voxel, 1)
    
    voxel_xyz /= voxel_counts[:, np.newaxis]
    
    return voxel_xyz, voxel_indices, point_to_voxel


def compute_verticality(
    xyz: np.ndarray,
    scale: float = 0.1,
    max_knn: int = 50000
) -> np.ndarray:
    """
    Compute verticality feature for each point using pgeof.
    
    Verticality measures how vertical the local surface normal is.
    Values close to 1 indicate vertical structures (stems).
    
    Args:
        xyz: (N, 3) array of points.
        scale: Neighborhood radius for feature computation.
        max_knn: Maximum k for nearest neighbor search.
    
    Returns:
        verticality: (N,) array with verticality values [0, 1].
    """
    verticality = pgeof.compute_features_selected(
        xyz.astype(np.float32),
        scale,
        max_knn,
        [EFeatureID.Verticality]
    )
    return verticality.ravel()


def extract_stem_stripe(
    xyz: np.ndarray,
    z_min: float = 0.5,
    z_max: float = 3.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract horizontal stripe from normalized point cloud for stem detection.
    
    Args:
        xyz: (N, 3) array with normalized Z coordinates.
        z_min: Lower height limit in meters.
        z_max: Upper height limit in meters.
    
    Returns:
        stripe_xyz: (M, 3) points within the stripe.
        stripe_indices: (M,) indices into original array.
    """
    mask = (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    stripe_indices = np.where(mask)[0]
    stripe_xyz = xyz[mask]
    return stripe_xyz, stripe_indices


def detect_stem_clusters(
    xyz: np.ndarray,
    verticality: np.ndarray,
    vert_threshold: float = 0.7,
    eps: float = 0.1,
    min_samples: int = 10,
    min_cluster_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect stem clusters using verticality filtering and DBSCAN.
    
    Args:
        xyz: (N, 3) array of points.
        verticality: (N,) verticality values.
        vert_threshold: Minimum verticality to consider as stem.
        eps: DBSCAN epsilon (neighborhood radius).
        min_samples: DBSCAN min_samples.
        min_cluster_points: Minimum points per valid cluster.
    
    Returns:
        cluster_labels: (N,) cluster labels (-1 for non-stem points).
        valid_clusters: Array of valid cluster IDs.
    """
    # Filter by verticality
    vert_mask = verticality > vert_threshold
    
    if np.sum(vert_mask) < min_samples:
        warnings.warn("Not enough high-verticality points for stem detection")
        return np.full(len(xyz), -1), np.array([])
    
    # Initialize labels
    cluster_labels = np.full(len(xyz), -1, dtype=np.int32)
    
    # DBSCAN on high-verticality points
    high_vert_xyz = xyz[vert_mask]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(high_vert_xyz)
    
    # Map labels back to original indices
    vert_indices = np.where(vert_mask)[0]
    cluster_labels[vert_indices] = clustering.labels_
    
    # Find valid clusters (enough points)
    unique_labels, counts = np.unique(clustering.labels_, return_counts=True)
    valid_mask = (unique_labels >= 0) & (counts >= min_cluster_points)
    valid_clusters = unique_labels[valid_mask]
    
    return cluster_labels, valid_clusters


def compute_tree_axes(
    xyz: np.ndarray,
    cluster_labels: np.ndarray,
    valid_clusters: np.ndarray,
    min_height_range: float = 1.0
) -> List[dict]:
    """
    Compute tree axis for each valid stem cluster using PCA.
    
    Args:
        xyz: (N, 3) array of points.
        cluster_labels: (N,) cluster labels from detect_stem_clusters.
        valid_clusters: Array of valid cluster IDs.
        min_height_range: Minimum vertical extent to be considered a valid stem.
    
    Returns:
        List of axis dictionaries with keys:
        - tree_id, centroid, axis_direction, axis_samples, height_range
    """
    axes = []
    tree_id = 0
    
    for cluster_id in valid_clusters:
        mask = cluster_labels == cluster_id
        stem_points = xyz[mask]
        
        # Check height range
        height_range = np.ptp(stem_points[:, 2])
        if height_range < min_height_range:
            continue
        
        # PCA for axis direction
        pca = PCA(n_components=3)
        pca.fit(stem_points)
        
        centroid = np.mean(stem_points, axis=0)
        axis_direction = pca.components_[0]
        
        # Ensure axis points upward
        if axis_direction[2] < 0:
            axis_direction = -axis_direction
        
        # Calculate deviation from vertical
        vertical = np.array([0, 0, 1])
        cos_angle = np.abs(np.dot(axis_direction, vertical))
        deviation_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Sample points along axis for KDTree
        # Extend axis from ground to top of tree
        z_min = np.min(xyz[:, 2])
        z_max = np.max(xyz[:, 2])
        
        # Calculate t parameter for axis line: P = centroid + t * axis_direction
        # Find t for z_min and z_max
        if np.abs(axis_direction[2]) > 0.01:
            t_min = (z_min - centroid[2]) / axis_direction[2]
            t_max = (z_max - centroid[2]) / axis_direction[2]
            
            t_values = np.linspace(min(t_min, t_max), max(t_min, t_max), 
                                   int((z_max - z_min) / 0.05))
            axis_samples = centroid + np.outer(t_values, axis_direction)
        else:
            # Axis is nearly horizontal, use vertical samples through centroid
            z_values = np.arange(z_min, z_max, 0.05)
            axis_samples = np.column_stack([
                np.full_like(z_values, centroid[0]),
                np.full_like(z_values, centroid[1]),
                z_values
            ])
        
        axes.append({
            'tree_id': tree_id,
            'cluster_id': cluster_id,
            'centroid': centroid,
            'axis_direction': axis_direction,
            'axis_samples': axis_samples,
            'height_range': height_range,
            'deviation_deg': deviation_deg,
            'n_stem_points': np.sum(mask)
        })
        tree_id += 1
    
    return axes


def assign_tree_ids(
    xyz: np.ndarray,
    axes: List[dict],
    max_distance: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign tree_id to each point based on distance to nearest axis.
    
    Args:
        xyz: (N, 3) array of points.
        axes: List of axis dictionaries from compute_tree_axes.
        max_distance: Maximum distance to axis to be assigned.
    
    Returns:
        tree_ids: (N,) array with tree_id for each point (-1 if unassigned).
        distances: (N,) distance to nearest axis.
    """
    if not axes:
        return np.full(len(xyz), -1), np.full(len(xyz), np.inf)
    
    # Combine all axis samples into single array with tree_id labels
    all_samples = []
    sample_tree_ids = []
    
    for axis in axes:
        samples = axis['axis_samples']
        all_samples.append(samples)
        sample_tree_ids.extend([axis['tree_id']] * len(samples))
    
    all_samples = np.vstack(all_samples)
    sample_tree_ids = np.array(sample_tree_ids)
    
    # Build KDTree and query
    tree = cKDTree(all_samples)
    distances, indices = tree.query(xyz, k=1, workers=-1)
    
    # Assign tree_id based on nearest axis sample
    tree_ids = sample_tree_ids[indices]
    
    # Mark points too far from any axis as unassigned
    tree_ids[distances > max_distance] = -1
    
    return tree_ids, distances


def segment_trees(
    xyz: np.ndarray,
    voxel_resolution: float = 0.05,
    stripe_z_min: float = 0.5,
    stripe_z_max: float = 3.5,
    verticality_scale: float = 0.15,
    verticality_threshold: float = 0.7,
    dbscan_eps: float = 0.15,
    dbscan_min_samples: int = 10,
    min_stem_points: int = 100,
    min_stem_height: float = 1.0,
    max_axis_distance: float = 2.0,
    verbose: bool = False
) -> TreeSegmentationResult:
    """
    Segment point cloud into individual trees.
    
    This is the main entry point for tree segmentation. It combines:
    1. Voxelization for speed
    2. Verticality filtering to isolate stems
    3. DBSCAN clustering of stem points
    4. PCA-based axis detection per stem
    5. Distance-based assignment of all points to nearest axis
    
    Args:
        xyz: (N, 3) array of height-normalized vegetation points.
        voxel_resolution: Voxel size for initial downsampling.
        stripe_z_min: Lower height limit for stem detection stripe.
        stripe_z_max: Upper height limit for stem detection stripe.
        verticality_scale: Neighborhood radius for verticality computation.
        verticality_threshold: Minimum verticality to be considered stem.
        dbscan_eps: DBSCAN neighborhood radius.
        dbscan_min_samples: DBSCAN minimum samples per cluster.
        min_stem_points: Minimum points for a valid stem cluster.
        min_stem_height: Minimum vertical extent for a valid stem.
        max_axis_distance: Maximum distance from axis to be assigned.
        verbose: Print progress information.
    
    Returns:
        TreeSegmentationResult with tree_ids and metadata.
    """
    n_original = len(xyz)
    
    if verbose:
        print(f"Input: {n_original:,} points")
    
    # Step 1: Voxelize for speed
    if verbose:
        print("Step 1: Voxelizing...")
    voxel_xyz, voxel_indices, point_to_voxel = voxelize_cloud(xyz, voxel_resolution)
    
    if verbose:
        print(f"  Voxelized to {len(voxel_xyz):,} voxels")
    
    # Step 2: Extract stem stripe
    if verbose:
        print(f"Step 2: Extracting stripe ({stripe_z_min}-{stripe_z_max}m)...")
    stripe_xyz, stripe_voxel_indices = extract_stem_stripe(voxel_xyz, stripe_z_min, stripe_z_max)
    
    if len(stripe_xyz) < min_stem_points:
        warnings.warn(f"Only {len(stripe_xyz)} points in stripe, not enough for segmentation")
        return TreeSegmentationResult(
            tree_ids=np.full(n_original, -1),
            n_trees=0,
            unassigned_count=n_original,
            tree_info=[]
        )
    
    if verbose:
        print(f"  Stripe: {len(stripe_xyz):,} voxels")
    
    # Step 3: Compute verticality
    if verbose:
        print("Step 3: Computing verticality...")
    verticality = compute_verticality(stripe_xyz, verticality_scale)
    
    if verbose:
        high_vert_count = np.sum(verticality > verticality_threshold)
        print(f"  High verticality (>{verticality_threshold}): {high_vert_count:,} voxels")
    
    # Step 4: Detect stem clusters
    if verbose:
        print("Step 4: Clustering stems...")
    cluster_labels, valid_clusters = detect_stem_clusters(
        stripe_xyz, verticality, verticality_threshold,
        dbscan_eps, dbscan_min_samples, min_stem_points
    )
    
    if len(valid_clusters) == 0:
        warnings.warn("No valid stem clusters detected")
        return TreeSegmentationResult(
            tree_ids=np.full(n_original, -1),
            n_trees=0,
            unassigned_count=n_original,
            tree_info=[]
        )
    
    if verbose:
        print(f"  Found {len(valid_clusters)} stem clusters")
    
    # Step 5: Compute tree axes
    if verbose:
        print("Step 5: Computing tree axes...")
    axes = compute_tree_axes(stripe_xyz, cluster_labels, valid_clusters, min_stem_height)
    
    if not axes:
        warnings.warn("No valid tree axes detected")
        return TreeSegmentationResult(
            tree_ids=np.full(n_original, -1),
            n_trees=0,
            unassigned_count=n_original,
            tree_info=[]
        )
    
    if verbose:
        print(f"  Detected {len(axes)} tree axes")
    
    # Step 6: Assign tree_ids to voxels
    if verbose:
        print("Step 6: Assigning points to trees...")
    voxel_tree_ids, voxel_distances = assign_tree_ids(voxel_xyz, axes, max_axis_distance)
    
    # Step 7: Propagate to original points
    tree_ids = voxel_tree_ids[point_to_voxel]
    
    unassigned_count = np.sum(tree_ids == -1)
    
    if verbose:
        assigned_count = n_original - unassigned_count
        print(f"  Assigned: {assigned_count:,} ({100*assigned_count/n_original:.1f}%)")
        print(f"  Unassigned: {unassigned_count:,} ({100*unassigned_count/n_original:.1f}%)")
    
    # Step 8: Build tree info
    tree_info = []
    for axis in axes:
        tid = axis['tree_id']
        mask = tree_ids == tid
        if np.sum(mask) == 0:
            continue
        
        tree_points = xyz[mask]
        tree_info.append(TreeInfo(
            tree_id=tid,
            centroid=axis['centroid'],
            n_points=np.sum(mask),
            height_max=np.max(tree_points[:, 2]),
            height_min=np.min(tree_points[:, 2]),
            axis_direction=axis['axis_direction'],
            axis_deviation_deg=axis['deviation_deg']
        ))
    
    if verbose:
        print(f"\n=== Summary ===")
        print(f"Trees detected: {len(tree_info)}")
        for info in tree_info:
            print(f"  Tree {info.tree_id}: {info.n_points:,} pts, "
                  f"H={info.height_max:.1f}m, dev={info.axis_deviation_deg:.1f}°")
    
    return TreeSegmentationResult(
        tree_ids=tree_ids,
        n_trees=len(tree_info),
        unassigned_count=unassigned_count,
        tree_info=tree_info
    )


def export_tree_locations(
    seg_result: TreeSegmentationResult,
    output_path: str,
    dbh_values: np.ndarray = None,
) -> np.ndarray:
    """
    Export tree locations as ASCII file for CloudCompare visualization.
    
    Inspired by 3DFin's tree_locator, this exports the base location of each tree.
    CloudCompare can open this directly: File > Open, select as "ASCII cloud".
    
    Output format (space-separated):
        X Y Z tree_id height n_points [dbh]
    
    Args:
        seg_result: TreeSegmentationResult from segment_trees().
        output_path: Path to output file (.txt or .asc).
        dbh_values: Optional array of DBH values per tree (in meters).
    
    Returns:
        tree_locations: (n_trees, 5+) array with location data.
    """
    from pathlib import Path
    
    if not seg_result.tree_info:
        raise ValueError("No trees in segmentation result")
    
    n_trees = len(seg_result.tree_info)
    has_dbh = dbh_values is not None and len(dbh_values) == n_trees
    
    # Build data array
    n_cols = 7 if has_dbh else 6
    tree_locations = np.zeros((n_trees, n_cols), dtype=np.float64)
    
    for i, info in enumerate(seg_result.tree_info):
        tree_locations[i, 0] = info.centroid[0]  # X
        tree_locations[i, 1] = info.centroid[1]  # Y
        tree_locations[i, 2] = info.height_min   # Z base
        tree_locations[i, 3] = info.tree_id      # tree_id
        tree_locations[i, 4] = info.height_max   # height
        tree_locations[i, 5] = info.n_points     # n_points
        if has_dbh:
            tree_locations[i, 6] = dbh_values[i]
    
    # Write ASCII file
    output_path = Path(output_path)
    
    # Header for CloudCompare (optional comment line)
    header = "//X Y Z tree_id height n_points"
    if has_dbh:
        header += " dbh"
    
    with open(output_path, 'w') as f:
        f.write(header + "\n")
        for row in tree_locations:
            if has_dbh:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {int(row[3])} {row[4]:.2f} {int(row[5])} {row[6]:.3f}\n")
            else:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {int(row[3])} {row[4]:.2f} {int(row[5])}\n")
    
    return tree_locations
