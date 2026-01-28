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
    local_radius: Optional[np.ndarray] = None,
    verticality_threshold: float = 0.7,
    linearity_threshold: float = 0.4,
    sphericity_threshold: float = 0.3,
    max_understory_height: float = 2.0,
    min_canopy_clearance: float = 3.0,
    min_stem_radius: float = 0.05,
    use_height_adaptive: bool = True,
    verbose: bool = False
) -> UnderstoryClassificationResult:
    """
    Classify points as tree or understory using geometric features.
    
    This function uses PCA-based geometric features rather than height alone
    to distinguish tree structures from understory vegetation.
    
    **NEW in v2:** Height-dependent thresholds and diameter-based filtering
    
    Classification logic:
    - STEM: High verticality AND high linearity (vertical linear structures)
    - CANOPY: Connected to stems (handled separately in connectivity validation)
    - UNDERSTORY: High sphericity + low position + low linearity + NOT under canopy
    
    Protection mechanisms:
    - Low branches protected by dist_to_top check (if canopy above, not understory)
    - Stems protected by high linearity requirement
    - Height-adaptive thresholds: stricter at low heights, relaxed higher up
    - Diameter filtering: thin stems (understory) vs thick stems (trees)
    
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
    local_radius : np.ndarray, optional
        Estimated local stem radius for each point. If provided, enables diameter-based filtering.
    verticality_threshold : float
        Base threshold for stem detection (default: 0.7). Adjusted by height if use_height_adaptive=True.
    linearity_threshold : float
        Threshold for stem detection (default: 0.4).
    sphericity_threshold : float
        Base threshold for understory detection (default: 0.3). Adjusted by height if use_height_adaptive=True.
    max_understory_height : float
        Maximum height for understory classification (default: 2.0m).
    min_canopy_clearance : float
        If dist_to_top > this value, point is under canopy = NOT understory (default: 3.0m).
    min_stem_radius : float
        Minimum radius for tree stems (default: 0.05m). Only used if local_radius provided.
    use_height_adaptive : bool
        Use height-dependent thresholds (default: True).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    UnderstoryClassificationResult
        Classification result with masks for tree, understory, and stem points.
    """
    n_points = len(xyz)
    z = xyz[:, 2]
    
    if verbose:
        print(f"Classifying understory: {n_points:,} points")
        print(f"  Base thresholds: verticality>{verticality_threshold}, "
              f"linearity>{linearity_threshold}, sphericity>{sphericity_threshold}")
        print(f"  Protection: max_height<{max_understory_height}m, canopy_clearance>{min_canopy_clearance}m")
        if use_height_adaptive:
            print(f"  Height-adaptive thresholds: ENABLED")
        if local_radius is not None:
            print(f"  Diameter filtering: ENABLED (min_radius={min_stem_radius}m)")
    
    # ========================================================================
    # Height-dependent threshold adjustment
    # ========================================================================
    if use_height_adaptive:
        # Compute adaptive thresholds based on height
        # Low heights (0-2m): strict (more likely understory)
        # Mid heights (2-4m): moderate
        # High heights (4m+): relaxed (crown region)
        
        verticality_thresh_adaptive = np.where(
            z < 2.0,
            verticality_threshold + 0.15,  # Stricter: 0.85 if base=0.7
            np.where(
                z < 4.0,
                verticality_threshold + 0.05,  # Moderate: 0.75
                verticality_threshold - 0.05   # Relaxed: 0.65
            )
        )
        
        sphericity_thresh_adaptive = np.where(
            z < 2.0,
            sphericity_threshold - 0.15,  # Stricter: 0.15 if base=0.3
            np.where(
                z < 4.0,
                sphericity_threshold - 0.05,  # Moderate: 0.25
                sphericity_threshold + 0.05   # Relaxed: 0.35
            )
        )
        
        if verbose:
            print(f"  Adaptive verticality: {verticality_thresh_adaptive.min():.2f} - {verticality_thresh_adaptive.max():.2f}")
            print(f"  Adaptive sphericity: {sphericity_thresh_adaptive.min():.2f} - {sphericity_thresh_adaptive.max():.2f}")
    else:
        verticality_thresh_adaptive = np.full(n_points, verticality_threshold)
        sphericity_thresh_adaptive = np.full(n_points, sphericity_threshold)
    
    # Initialize masks
    is_stem = np.zeros(n_points, dtype=bool)
    is_understory = np.zeros(n_points, dtype=bool)
    is_tree = np.zeros(n_points, dtype=bool)
    
    # ========================================================================
    # Step 1: Identify likely STEM points
    # Stems have high verticality AND high linearity
    # ========================================================================
    is_stem = (verticality >= verticality_thresh_adaptive) & (linearity >= linearity_threshold)
    
    # Diameter-based refinement if radius data available
    if local_radius is not None:
        # Thin stems (radius < threshold) are likely understory
        is_thin_stem = local_radius < min_stem_radius
        is_stem = is_stem & ~is_thin_stem
        
        if verbose:
            n_thin = np.sum(is_thin_stem & (verticality >= verticality_thresh_adaptive) & (linearity >= linearity_threshold))
            print(f"  Thin stems filtered: {n_thin:,} points")
    
    if verbose:
        print(f"  Potential stems: {np.sum(is_stem):,} points ({100*np.mean(is_stem):.1f}%)")
    
    # ========================================================================
    # Step 2: Identify likely UNDERSTORY points
    # Understory must satisfy ALL conditions:
    #   - High sphericity (scattered, shrub-like geometry)
    #   - Low linearity (excludes stems and branches)
    #   - Near ground (low dist_to_ground)
    #   - NOT under significant canopy (low dist_to_top = no tree above)
    #   - Not already classified as stem
    # ========================================================================
    
    # Relative position in local column
    relative_height = dist_to_ground / (dist_to_ground + dist_to_top + 1e-6)
    
    # Protection for low branches: if there's significant canopy above, it's NOT understory
    is_under_canopy = dist_to_top > min_canopy_clearance
    
    is_understory = (
        (sphericity >= sphericity_thresh_adaptive) &  # Scattered/shrub-like geometry (adaptive)
        (linearity < linearity_threshold) &           # NOT linear (protects stems/branches)
        (relative_height < 0.3) &                     # In lower 30% of local height range
        (z < max_understory_height) &                 # Below understory height threshold
        (~is_under_canopy) &                          # NOT under significant canopy (protects low branches)
        (~is_stem)                                    # Not already classified as stem
    )
    
    if verbose:
        n_under_canopy = np.sum(is_under_canopy & (z < max_understory_height))
        print(f"  Protected (under canopy): {n_under_canopy:,} low points")
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


def validate_tree_connectivity_fast(
    xyz: np.ndarray,
    is_stem: np.ndarray,
    voxel_size: float = 0.1,
    min_component_size: int = 50,
    min_tree_height: float = 5.0,
    verbose: bool = False
) -> np.ndarray:
    """
    Optimized tree connectivity validation using voxel-based graph.
    
    Instead of building a graph on all 6M points, this:
    1. Voxelizes points to ~300K voxels
    2. Builds connectivity graph on voxels (much smaller)
    3. Finds connected components containing stem voxels
    4. Maps validity back to original points
    
    This achieves 10-20x speedup with acceptable accuracy.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    is_stem : np.ndarray
        Boolean mask indicating verified stem points.
    voxel_size : float
        Voxel size for graph construction (default: 0.1m).
    min_component_size : int
        Minimum voxels for a valid component (default: 50).
    min_tree_height : float
        Minimum vertical extent (Z range) for a valid tree component.
        Components shorter than this are classified as understory.
        Can be calibrated with field measurements (default: 5.0m).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    is_valid_tree : np.ndarray
        Boolean mask indicating points connected to stems.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    
    n_points = len(xyz)
    n_stems = np.sum(is_stem)
    
    if verbose:
        print(f"Validating tree connectivity (optimized): {n_points:,} points, {n_stems:,} stems")
        print(f"  Voxel size: {voxel_size}m")
    
    if n_stems == 0:
        if verbose:
            print("  Warning: No stem points found, returning all as invalid")
        return np.zeros(n_points, dtype=bool)
    
    # ========================================================================
    # Step 1: Voxelize point cloud
    # ========================================================================
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    n_voxels = len(unique_voxels)
    
    if verbose:
        print(f"  Voxelized: {n_points:,} points → {n_voxels:,} voxels")
    
    # Determine which voxels contain stem points
    voxel_has_stem = np.zeros(n_voxels, dtype=bool)
    stem_voxel_indices = inverse_indices[is_stem]
    voxel_has_stem[stem_voxel_indices] = True
    n_stem_voxels = np.sum(voxel_has_stem)
    
    if verbose:
        print(f"  Stem voxels: {n_stem_voxels:,}")
    
    # ========================================================================
    # Step 2: Build voxel adjacency graph using 26-connectivity
    # ========================================================================
    if verbose:
        print("  Building voxel adjacency graph...")
    
    # Create a dictionary for fast voxel lookup
    voxel_to_idx = {tuple(v): i for i, v in enumerate(unique_voxels)}
    
    # 26-connectivity offsets (all neighbors in 3D grid)
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx != 0 or dy != 0 or dz != 0:
                    offsets.append((dx, dy, dz))
    
    # Build edges list
    rows = []
    cols = []
    
    for i, voxel in enumerate(unique_voxels):
        for dx, dy, dz in offsets:
            neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
            if neighbor in voxel_to_idx:
                j = voxel_to_idx[neighbor]
                rows.append(i)
                cols.append(j)
    
    # Create sparse adjacency matrix
    data = np.ones(len(rows), dtype=bool)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_voxels, n_voxels))
    
    if verbose:
        n_edges = len(rows) // 2
        print(f"  Graph: {n_voxels:,} voxels, {n_edges:,} edges")
    
    # ========================================================================
    # Step 3: Find connected components
    # ========================================================================
    n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
    
    if verbose:
        print(f"  Found {n_components:,} connected components")
    
    # ========================================================================
    # Step 4: Identify valid components (those with stem voxels)
    # ========================================================================
    stem_voxel_indices = np.where(voxel_has_stem)[0]
    stem_labels = labels[stem_voxel_indices]
    valid_labels = np.unique(stem_labels)
    
    # Mark voxels in valid components
    voxel_is_valid = np.isin(labels, valid_labels)
    
    # Filter small components and short components
    n_filtered_size = 0
    n_filtered_height = 0
    
    for label in valid_labels:
        component_mask = labels == label
        component_voxels = unique_voxels[component_mask]
        
        # Check component size
        if np.sum(component_mask) < min_component_size:
            voxel_is_valid[component_mask] = False
            n_filtered_size += 1
            continue
        
        # Check vertical extent (Z range)
        z_min = component_voxels[:, 2].min() * voxel_size
        z_max = component_voxels[:, 2].max() * voxel_size
        z_range = z_max - z_min
        
        if z_range < min_tree_height:
            voxel_is_valid[component_mask] = False
            n_filtered_height += 1
    
    if verbose:
        print(f"  Filtered: {n_filtered_size} small, {n_filtered_height} short (<{min_tree_height}m)")
    
    # ========================================================================
    # Step 5: Map validity back to original points
    # ========================================================================
    is_valid_tree = voxel_is_valid[inverse_indices]
    
    if verbose:
        n_valid = np.sum(is_valid_tree)
        n_invalid = n_points - n_valid
        print(f"  Valid tree points: {n_valid:,} ({100*n_valid/n_points:.1f}%)")
        print(f"  Disconnected (understory): {n_invalid:,} ({100*n_invalid/n_points:.1f}%)")
    
    return is_valid_tree


def filter_understory_stripe(
    xyz: np.ndarray,
    is_stem: np.ndarray,
    is_valid_tree: np.ndarray,
    stripe_max_height: float = 2.0,
    verbose: bool = False
) -> np.ndarray:
    """
    Filter understory within a height band (stripe) based on stem classification.
    
    Within the stripe zone (ground to stripe_max_height), points that are NOT
    stems are classified as understory and removed from the valid tree mask.
    Points above the stripe are protected (assumed to be canopy).
    
    This approach is inspired by 3DFin's stripe concept but uses our existing
    stem classification rather than re-computing verticality.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3). Z should be normalized height.
    is_stem : np.ndarray
        Boolean mask from classify_understory indicating stem points.
    is_valid_tree : np.ndarray
        Boolean mask from validate_tree_connectivity.
    stripe_max_height : float
        Upper limit of the stripe zone in meters. Points below this height
        that are not stems will be classified as understory.
        Can be calibrated with field measurements (default: 2.0m).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    is_tree_final : np.ndarray
        Boolean mask with understory filtered out. True = tree, False = understory.
    
    Notes
    -----
    Logic:
    - Points with Z >= stripe_max_height: Keep if is_valid_tree (canopy protected)
    - Points with Z < stripe_max_height: Keep ONLY if is_stem (understory removed)
    
    This ensures:
    1. Canopy is never removed (above stripe)
    2. Stems are always kept (within stripe)
    3. Non-stem vegetation within stripe is removed as understory
    """
    n_points = len(xyz)
    z = xyz[:, 2]
    
    if verbose:
        print(f"Filtering understory in stripe (0 to {stripe_max_height}m)...")
    
    # Start with the connectivity-based mask
    is_tree_final = is_valid_tree.copy()
    
    # Within the stripe zone: keep only stems
    in_stripe = z < stripe_max_height
    not_stem = ~is_stem
    
    # Points to remove: in stripe AND not stem AND currently marked as tree
    understory_in_stripe = in_stripe & not_stem & is_valid_tree
    
    # Remove understory from final mask
    is_tree_final[understory_in_stripe] = False
    
    if verbose:
        n_removed = np.sum(understory_in_stripe)
        n_final_tree = np.sum(is_tree_final)
        n_final_understory = n_points - n_final_tree
        
        print(f"  Points in stripe (<{stripe_max_height}m): {np.sum(in_stripe):,}")
        print(f"  Stems in stripe (protected): {np.sum(in_stripe & is_stem):,}")
        print(f"  Understory removed from stripe: {n_removed:,}")
        print(f"  Final: {n_final_tree:,} tree pts ({100*n_final_tree/n_points:.1f}%)")
        print(f"         {n_final_understory:,} understory pts ({100*n_final_understory/n_points:.1f}%)")
    
    return is_tree_final


# ============================================================================
# ITERATIVE PEELING FOR UNDERSTORY SEPARATION
# ============================================================================

@dataclass
class IterativePeelingResult:
    """Result of iterative peeling understory separation."""
    is_tree: np.ndarray          # (N,) boolean mask (True = tree, False = understory)
    n_seeds: int                 # Number of initial seed points
    n_iterations: int            # Number of expansion iterations performed
    n_tree: int                  # Final number of tree points
    n_understory: int            # Final number of understory points
    expansion_percentage: float  # Percentage of points added via expansion


def iterative_peeling_understory(
    xyz: np.ndarray,
    verticality: np.ndarray,
    linearity: np.ndarray,
    sphericity: np.ndarray,
    dist_to_ground: np.ndarray,
    seed_verticality: float = 0.9,
    seed_linearity: float = 0.6,
    seed_height_min: float = 1.0,
    seed_height_max: float = 2.5,
    expansion_verticality: float = 0.5,
    expansion_radius: float = 0.3,
    max_iterations: int = 50,
    verbose: bool = False
) -> IterativePeelingResult:
    """
    Separate trees from understory using iterative peeling (region growing).
    
    This algorithm addresses two key failures of threshold-based classification:
    1. False negatives: Trunk edge points with unreliable features
    2. False positives: Isolated understory clusters that pass thresholds
    
    Strategy:
    - Start with ultra-reliable trunk seeds (high verticality + linearity, 1-2.5m height)
    - Iteratively expand to neighbors meeting relaxed criteria
    - Natural protection of entire trunk cylinders
    - Automatic exclusion of isolated understory
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    verticality : np.ndarray
        Verticality feature (0-1) for each point.
    linearity : np.ndarray
        Linearity feature (0-1) for each point.
    sphericity : np.ndarray
        Sphericity feature (0-1) for each point.
    dist_to_ground : np.ndarray
        Distance to ground for each point (meters).
    seed_verticality : float
        Minimum verticality for seed selection (default: 0.9).
    seed_linearity : float
        Minimum linearity for seed selection (default: 0.6).
    seed_height_min : float
        Minimum height for seed selection (default: 1.0m).
    seed_height_max : float
        Maximum height for seed selection (default: 2.5m).
    expansion_verticality : float
        Minimum verticality for expansion (default: 0.5).
    expansion_radius : float
        Neighbor search radius for expansion (default: 0.3m).
    max_iterations : int
        Maximum number of expansion iterations (default: 50).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    IterativePeelingResult
        Result object containing is_tree mask and statistics.
    
    Example
    -------
    >>> result = iterative_peeling_understory(
    ...     xyz, verticality, linearity, sphericity, dist_to_ground,
    ...     seed_verticality=0.9,
    ...     expansion_verticality=0.5,
    ...     verbose=True
    ... )
    >>> trees_xyz = xyz[result.is_tree]
    >>> understory_xyz = xyz[~result.is_tree]
    """
    from scipy.spatial import cKDTree
    
    n_points = len(xyz)
    
    if verbose:
        print(f"Iterative peeling for {n_points:,} points...")
        print(f"  Seed criteria: verticality>{seed_verticality}, linearity>{seed_linearity}, "
              f"height {seed_height_min}-{seed_height_max}m")
        print(f"  Expansion criteria: verticality>{expansion_verticality}, radius={expansion_radius}m")
    
    # ========================================================================
    # Step 1: Initialize seeds (ultra-reliable trunk points)
    # ========================================================================
    seed_mask = (
        (verticality > seed_verticality) &
        (linearity > seed_linearity) &
        (dist_to_ground >= seed_height_min) &
        (dist_to_ground <= seed_height_max)
    )
    
    n_seeds = np.sum(seed_mask)
    
    if n_seeds == 0:
        if verbose:
            print("  ⚠ WARNING: No seeds found! Adjusting criteria...")
        # Fallback: relax criteria slightly
        seed_mask = (
            (verticality > seed_verticality - 0.1) &
            (linearity > seed_linearity - 0.1) &
            (dist_to_ground >= seed_height_min) &
            (dist_to_ground <= seed_height_max)
        )
        n_seeds = np.sum(seed_mask)
        
        if n_seeds == 0:
            warnings.warn("No seeds found even with relaxed criteria. Returning all points as understory.")
            return IterativePeelingResult(
                is_tree=np.zeros(n_points, dtype=bool),
                n_seeds=0,
                n_iterations=0,
                n_tree=0,
                n_understory=n_points,
                expansion_percentage=0.0
            )
    
    if verbose:
        print(f"  Seeds: {n_seeds:,} points ({100*n_seeds/n_points:.2f}%)")
    
    # Initialize tree mask with seeds
    is_tree = seed_mask.copy()
    
    # ========================================================================
    # Step 2: Build KDTree for neighbor queries
    # ========================================================================
    if verbose:
        print("  Building KDTree...")
    
    tree = cKDTree(xyz)
    
    # ========================================================================
    # Step 3: Iterative expansion
    # ========================================================================
    if verbose:
        print(f"  Expanding from seeds (max {max_iterations} iterations)...")
    
    previous_tree = seed_mask.copy()
    
    for iteration in range(max_iterations):
        # Find frontier (points added in last iteration)
        new_tree_points = is_tree & ~previous_tree
        n_frontier = np.sum(new_tree_points)
        
        if n_frontier == 0:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
        
        if verbose and (iteration % 5 == 0 or iteration < 3):
            n_current = np.sum(is_tree)
            print(f"    Iteration {iteration}: {n_current:,} tree points (+{n_frontier:,} frontier)")
        
        # Update previous for next iteration
        previous_tree = is_tree.copy()
        
        # Query neighbors of frontier points
        frontier_indices = np.where(new_tree_points)[0]
        
        for point_idx in frontier_indices:
            # Find neighbors within expansion radius
            neighbor_indices = tree.query_ball_point(xyz[point_idx], expansion_radius)
            
            # Add neighbors if they meet expansion criteria
            for n_idx in neighbor_indices:
                if not is_tree[n_idx]:
                    # Relaxed criteria: only verticality check
                    if verticality[n_idx] > expansion_verticality:
                        is_tree[n_idx] = True
    
    # ========================================================================
    # Step 4: Compute statistics
    # ========================================================================
    n_tree = np.sum(is_tree)
    n_understory = n_points - n_tree
    n_expanded = n_tree - n_seeds
    expansion_percentage = 100 * n_expanded / n_seeds if n_seeds > 0 else 0.0
    
    if verbose:
        print(f"  ✓ Peeling complete")
        print(f"    Seeds: {n_seeds:,}")
        print(f"    Expanded: +{n_expanded:,} points ({expansion_percentage:.1f}% of seeds)")
        print(f"    Final: {n_tree:,} tree ({100*n_tree/n_points:.1f}%), "
              f"{n_understory:,} understory ({100*n_understory/n_points:.1f}%)")
    
    return IterativePeelingResult(
        is_tree=is_tree,
        n_seeds=n_seeds,
        n_iterations=iteration + 1 if n_frontier > 0 else iteration,
        n_tree=n_tree,
        n_understory=n_understory,
        expansion_percentage=expansion_percentage
    )

