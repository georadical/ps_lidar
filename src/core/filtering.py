"""
Filtering Module

Provides functions for point cloud filtering including noise removal.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


@dataclass
class NoiseFilterResult:
    """Result of noise filtering operation."""
    clean_xyz: np.ndarray        # (M, 3) cleaned points
    clean_indices: np.ndarray    # (M,) indices into original array
    noise_mask: np.ndarray       # (N,) boolean mask (True = noise)
    n_removed: int               # Number of removed points
    removal_percentage: float    # Percentage removed


def filter_noise_sor(
    xyz: np.ndarray,
    k_neighbors: int = 10,
    std_ratio: float = 2.0,
    verbose: bool = False
) -> NoiseFilterResult:
    """
    Statistical Outlier Removal (SOR) for point cloud noise filtering.
    
    Removes points that are farther than `std_ratio` standard deviations
    from the mean distance to their k nearest neighbors.
    
    Args:
        xyz: (N, 3) array of points.
        k_neighbors: Number of neighbors to analyze for each point.
        std_ratio: Standard deviation multiplier threshold.
        verbose: Print progress information.
    
    Returns:
        NoiseFilterResult with cleaned points and statistics.
    
    Example:
        >>> result = filter_noise_sor(xyz, k_neighbors=10, std_ratio=2.0)
        >>> clean_xyz = result.clean_xyz
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required for SOR filtering. Install with: pip install open3d")
    
    n_original = len(xyz)
    
    if n_original < k_neighbors + 1:
        warnings.warn(f"Not enough points ({n_original}) for SOR with k={k_neighbors}")
        return NoiseFilterResult(
            clean_xyz=xyz.copy(),
            clean_indices=np.arange(n_original),
            noise_mask=np.zeros(n_original, dtype=bool),
            n_removed=0,
            removal_percentage=0.0
        )
    
    if verbose:
        print(f"SOR filtering: {n_original:,} points, k={k_neighbors}, std={std_ratio}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Apply Statistical Outlier Removal
    pcd_clean, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=k_neighbors,
        std_ratio=std_ratio
    )
    
    inlier_indices = np.array(inlier_indices)
    
    # Build results
    noise_mask = np.ones(n_original, dtype=bool)
    noise_mask[inlier_indices] = False
    
    n_removed = n_original - len(inlier_indices)
    removal_percentage = 100 * n_removed / n_original
    
    if verbose:
        print(f"  Removed: {n_removed:,} points ({removal_percentage:.2f}%)")
        print(f"  Remaining: {len(inlier_indices):,} points")
    
    return NoiseFilterResult(
        clean_xyz=xyz[inlier_indices],
        clean_indices=inlier_indices,
        noise_mask=noise_mask,
        n_removed=n_removed,
        removal_percentage=removal_percentage
    )


def filter_noise_radius(
    xyz: np.ndarray,
    radius: float = 0.1,
    min_neighbors: int = 5,
    verbose: bool = False
) -> NoiseFilterResult:
    """
    Radius Outlier Removal for point cloud noise filtering.
    
    Removes points that have fewer than `min_neighbors` within `radius`.
    
    Args:
        xyz: (N, 3) array of points.
        radius: Search radius in meters.
        min_neighbors: Minimum neighbors required to keep point.
        verbose: Print progress information.
    
    Returns:
        NoiseFilterResult with cleaned points and statistics.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required for radius filtering. Install with: pip install open3d")
    
    n_original = len(xyz)
    
    if verbose:
        print(f"Radius filtering: {n_original:,} points, r={radius}m, min_n={min_neighbors}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Apply Radius Outlier Removal
    pcd_clean, inlier_indices = pcd.remove_radius_outlier(
        nb_points=min_neighbors,
        radius=radius
    )
    
    inlier_indices = np.array(inlier_indices)
    
    # Build results
    noise_mask = np.ones(n_original, dtype=bool)
    noise_mask[inlier_indices] = False
    
    n_removed = n_original - len(inlier_indices)
    removal_percentage = 100 * n_removed / n_original
    
    if verbose:
        print(f"  Removed: {n_removed:,} points ({removal_percentage:.2f}%)")
        print(f"  Remaining: {len(inlier_indices):,} points")
    
    return NoiseFilterResult(
        clean_xyz=xyz[inlier_indices],
        clean_indices=inlier_indices,
        noise_mask=noise_mask,
        n_removed=n_removed,
        removal_percentage=removal_percentage
    )


@dataclass
class UnderstorySeparationResult:
    """Result of understory separation operation."""
    tree_xyz: np.ndarray          # (M, 3) tree points
    tree_indices: np.ndarray      # (M,) indices into original array
    understory_xyz: np.ndarray    # (K, 3) understory points
    understory_indices: np.ndarray # (K,) indices into original array
    n_trees: int                  # Points classified as trees
    n_understory: int             # Points classified as understory


def separate_understory(
    xyz: np.ndarray,
    tree_ids: np.ndarray,
    min_tree_height: float = 5.0,
    max_ground_start: float = 1.5,
    min_vertical_extent: float = 2.0,
    connectivity_radius: float = 0.5,
    connectivity_z_step: float = 0.5,
    verbose: bool = False
) -> Tuple[UnderstorySeparationResult, np.ndarray]:
    """
    Separate tree points from understory vegetation.
    
    Validates each detected tree cluster using criteria inspired by SimpleForest:
    1. Height check: Tree must reach above min_tree_height
    2. Ground connectivity: Cluster must have points near ground (< max_ground_start)
    3. Vertical extent: Cluster must span min_vertical_extent vertically
    4. Vertical connectivity: Must have continuous presence through height bands
    
    Args:
        xyz: (N, 3) array of height-normalized vegetation points.
        tree_ids: (N,) array of tree_id per point (-1 for unassigned).
        min_tree_height: Minimum height to be considered a tree (meters).
        max_ground_start: Maximum height where tree base must be present.
        min_vertical_extent: Minimum vertical extent of cluster.
        connectivity_radius: Maximum gap allowed in vertical connectivity.
        connectivity_z_step: Height step for connectivity analysis.
        verbose: Print progress information.
    
    Returns:
        Tuple of (UnderstorySeparationResult, valid_tree_ids)
        valid_tree_ids: Array of tree_ids that passed validation.
    """
    from scipy.spatial import cKDTree
    
    n_original = len(xyz)
    unique_ids = np.unique(tree_ids[tree_ids >= 0])
    
    if verbose:
        print(f"Understory separation: {len(unique_ids)} clusters to validate")
        print(f"  Criteria: height>{min_tree_height}m, base<{max_ground_start}m, extent>{min_vertical_extent}m")
    
    valid_tree_ids = []
    rejection_reasons = {'height': 0, 'ground': 0, 'extent': 0, 'connectivity': 0}
    
    for tid in unique_ids:
        mask = tree_ids == tid
        cluster_xyz = xyz[mask]
        
        if len(cluster_xyz) < 10:
            continue
        
        z_vals = cluster_xyz[:, 2]
        z_min = np.min(z_vals)
        z_max = np.max(z_vals)
        z_extent = z_max - z_min
        
        # Criterion 1: Height check
        if z_max < min_tree_height:
            rejection_reasons['height'] += 1
            continue
        
        # Criterion 2: Ground connectivity - must start near ground
        if z_min > max_ground_start:
            rejection_reasons['ground'] += 1
            continue
        
        # Criterion 3: Vertical extent
        if z_extent < min_vertical_extent:
            rejection_reasons['extent'] += 1
            continue
        
        # Criterion 4: Vertical connectivity (simplified Region Growing check)
        # Check that there are points in regular height bands from bottom to top
        z_bands = np.arange(z_min, z_max, connectivity_z_step)
        connected = True
        
        for z_band in z_bands:
            band_mask = (z_vals >= z_band) & (z_vals < z_band + connectivity_z_step * 2)
            if np.sum(band_mask) < 3:  # Require at least 3 points per band
                connected = False
                break
        
        if not connected:
            rejection_reasons['connectivity'] += 1
            continue
        
        # Passed all criteria
        valid_tree_ids.append(tid)
    
    valid_tree_ids = np.array(valid_tree_ids)
    
    # Create masks for tree vs understory
    tree_mask = np.isin(tree_ids, valid_tree_ids)
    understory_mask = (tree_ids >= 0) & ~tree_mask
    unassigned_mask = tree_ids == -1
    
    # Include unassigned points as understory (conservative approach)
    understory_mask = understory_mask | unassigned_mask
    
    if verbose:
        print(f"\n  Valid trees: {len(valid_tree_ids)}")
        print(f"  Rejected: height={rejection_reasons['height']}, "
              f"ground={rejection_reasons['ground']}, "
              f"extent={rejection_reasons['extent']}, "
              f"connectivity={rejection_reasons['connectivity']}")
        print(f"  Tree points: {np.sum(tree_mask):,}")
        print(f"  Understory points: {np.sum(understory_mask):,}")
    
    result = UnderstorySeparationResult(
        tree_xyz=xyz[tree_mask],
        tree_indices=np.where(tree_mask)[0],
        understory_xyz=xyz[understory_mask],
        understory_indices=np.where(understory_mask)[0],
        n_trees=np.sum(tree_mask),
        n_understory=np.sum(understory_mask)
    )
    
    return result, valid_tree_ids


@dataclass
class UnderstoryClassificationResult:
    """Result of understory classification using geometric features."""
    is_tree: np.ndarray           # (N,) boolean mask (True = tree)
    is_understory: np.ndarray     # (N,) boolean mask (True = understory)
    is_stem: np.ndarray           # (N,) boolean mask (True = likely stem)
    n_tree: int                   # Points classified as trees
    n_understory: int             # Points classified as understory


def classify_understory(
    xyz: np.ndarray,
    verticality: np.ndarray,
    linearity: np.ndarray,
    sphericity: np.ndarray,
    dist_to_ground: np.ndarray,
    dist_to_top: np.ndarray,
    verticality_threshold: float = 0.7,
    linearity_threshold: float = 0.4,
    sphericity_threshold: float = 0.5,
    max_understory_height: float = 3.0,
    verbose: bool = False
) -> UnderstoryClassificationResult:
    """
    Classify points as tree or understory using geometric features.
    
    This function uses PCA-based geometric features rather than height alone
    to distinguish tree structures from understory vegetation.
    
    Classification logic:
    - STEM: High verticality AND high linearity (vertical linear structures)
    - CANOPY: Connected to stems (handled separately in connectivity validation)
    - UNDERSTORY: High sphericity + low position + disconnected from stems
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    verticality : np.ndarray
        Verticality feature (0-1) from PCA.
    linearity : np.ndarray
        Linearity feature (0-1) from PCA.
    sphericity : np.ndarray
        Sphericity feature (0-1) from PCA.
    dist_to_ground : np.ndarray
        Distance from point to ground in vertical cylinder.
    dist_to_top : np.ndarray
        Distance from point to top in vertical cylinder.
    verticality_threshold : float
        Threshold for stem detection (default: 0.7).
    linearity_threshold : float
        Threshold for stem detection (default: 0.4).
    sphericity_threshold : float
        Threshold for understory detection (default: 0.5).
    max_understory_height : float
        Maximum height for understory classification (meters).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    UnderstoryClassificationResult
        Classification result with masks for tree, understory, and stem points.
    
    Notes
    -----
    This is the first pass of classification. Points classified as "tree" here
    should be further validated using connectivity analysis (Phase 3) to ensure
    they form continuous vertical structures.
    """
    n_points = len(xyz)
    z = xyz[:, 2]
    
    if verbose:
        print(f"Classifying understory: {n_points:,} points")
        print(f"  Thresholds: verticality>{verticality_threshold}, "
              f"linearity>{linearity_threshold}, sphericity>{sphericity_threshold}")
    
    # Initialize masks
    is_stem = np.zeros(n_points, dtype=bool)
    is_understory = np.zeros(n_points, dtype=bool)
    is_tree = np.zeros(n_points, dtype=bool)
    
    # ========================================================================
    # Step 1: Identify likely STEM points
    # Stems have high verticality AND high linearity
    # ========================================================================
    is_stem = (verticality >= verticality_threshold) & (linearity >= linearity_threshold)
    
    if verbose:
        print(f"  Potential stems: {np.sum(is_stem):,} points ({100*np.mean(is_stem):.1f}%)")
    
    # ========================================================================
    # Step 2: Identify likely UNDERSTORY points
    # Understory: high sphericity + low proximity to canopy + low height
    # ========================================================================
    
    # Criteria for understory:
    # - High sphericity (scattered, shrub-like)
    # - Low position relative to local maximum (far from canopy)
    # - Not identified as stem
    relative_height = dist_to_ground / (dist_to_ground + dist_to_top + 1e-6)
    
    is_understory = (
        (sphericity >= sphericity_threshold) &  # Scattered/shrub-like geometry
        (relative_height < 0.4) &               # In lower 40% of local height range
        (z < max_understory_height) &           # Below understory height threshold
        (~is_stem)                              # Not already classified as stem
    )
    
    if verbose:
        print(f"  Understory: {np.sum(is_understory):,} points ({100*np.mean(is_understory):.1f}%)")
    
    # ========================================================================
    # Step 3: Classify remaining points
    # Points not understory = tree (stem or canopy)
    # ========================================================================
    is_tree = ~is_understory
    
    if verbose:
        print(f"  Tree (including canopy): {np.sum(is_tree):,} points ({100*np.mean(is_tree):.1f}%)")
    
    return UnderstoryClassificationResult(
        is_tree=is_tree,
        is_understory=is_understory,
        is_stem=is_stem,
        n_tree=np.sum(is_tree),
        n_understory=np.sum(is_understory)
    )


def validate_tree_connectivity(
    xyz: np.ndarray,
    is_stem: np.ndarray,
    horizontal_radius: float = 0.3,
    vertical_radius: float = 0.8,
    min_component_size: int = 50,
    verbose: bool = False
) -> np.ndarray:
    """
    Validate tree points using anisotropic graph connectivity.
    
    Points are considered part of a tree if they are connected to verified
    stem points through a graph where edges have different horizontal and
    vertical tolerance.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    is_stem : np.ndarray
        Boolean mask indicating verified stem points.
    horizontal_radius : float
        Maximum horizontal distance for edge connection (default: 0.3m).
        Smaller values prevent merging nearby trees.
    vertical_radius : float
        Maximum vertical distance for edge connection (default: 0.8m).
        Larger values allow gaps in foliage while maintaining connectivity.
    min_component_size : int
        Minimum points for a connected component to be valid (default: 50).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    is_valid_tree : np.ndarray
        Boolean mask indicating points that are connected to stems.
    
    Notes
    -----
    The algorithm:
    1. Build a KD-Tree with anisotropic scaling (Z compressed)
    2. Find connected components starting from stem points
    3. Points not reachable from stems are classified as disconnected (understory)
    
    This is inspired by the Graph Theory approach where edges connect
    spatially proximate points, and connected components represent objects.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components
    
    n_points = len(xyz)
    n_stems = np.sum(is_stem)
    
    if verbose:
        print(f"Validating tree connectivity: {n_points:,} points, {n_stems:,} stems")
        print(f"  Anisotropic radii: horizontal={horizontal_radius}m, vertical={vertical_radius}m")
    
    if n_stems == 0:
        if verbose:
            print("  Warning: No stem points found, returning all as invalid")
        return np.zeros(n_points, dtype=bool)
    
    # ========================================================================
    # Step 1: Scale coordinates for anisotropic search
    # Scale Z so that vertical_radius becomes equivalent to horizontal_radius
    # ========================================================================
    z_scale = horizontal_radius / vertical_radius
    xyz_scaled = xyz.copy()
    xyz_scaled[:, 2] *= z_scale
    
    if verbose:
        print(f"  Building KD-Tree with Z-scaling factor: {z_scale:.2f}")
    
    # Build tree on scaled coordinates
    tree = cKDTree(xyz_scaled)
    
    # ========================================================================
    # Step 2: Build sparse adjacency graph
    # ========================================================================
    if verbose:
        print(f"  Finding neighbors within radius {horizontal_radius}m (scaled)...")
    
    # Query all neighbors within the (now isotropic) radius
    neighbor_lists = tree.query_ball_point(xyz_scaled, r=horizontal_radius, workers=-1)
    
    # Build sparse adjacency matrix
    adjacency = lil_matrix((n_points, n_points), dtype=bool)
    
    for i, neighbors in enumerate(neighbor_lists):
        for j in neighbors:
            if i != j:
                adjacency[i, j] = True
                adjacency[j, i] = True
    
    adjacency = adjacency.tocsr()
    
    if verbose:
        n_edges = adjacency.nnz // 2
        print(f"  Graph: {n_points:,} nodes, {n_edges:,} edges")
    
    # ========================================================================
    # Step 3: Find connected components
    # ========================================================================
    n_components, labels = connected_components(
        adjacency, 
        directed=False, 
        return_labels=True
    )
    
    if verbose:
        print(f"  Found {n_components:,} connected components")
    
    # ========================================================================
    # Step 4: Identify components that contain stem points
    # ========================================================================
    stem_indices = np.where(is_stem)[0]
    stem_labels = labels[stem_indices]
    valid_labels = np.unique(stem_labels)
    
    # Count points per valid component
    is_valid_tree = np.isin(labels, valid_labels)
    
    # Filter out small components
    for label in valid_labels:
        component_mask = labels == label
        if np.sum(component_mask) < min_component_size:
            is_valid_tree[component_mask] = False
    
    if verbose:
        n_valid = np.sum(is_valid_tree)
        n_invalid = n_points - n_valid
        print(f"  Valid tree points: {n_valid:,} ({100*n_valid/n_points:.1f}%)")
        print(f"  Disconnected (understory): {n_invalid:,} ({100*n_invalid/n_points:.1f}%)")
    
    return is_valid_tree


