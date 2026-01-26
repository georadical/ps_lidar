"""
Feature extraction module for point cloud geometric analysis.

This module provides functions to compute geometric features from point clouds
using PCA (Principal Component Analysis) and spatial search methods.

Features computed:
- Verticality: How vertical a local structure is (0-1)
- Linearity: How linear a structure is (poles, wires, stems)
- Planarity: How planar a structure is (ground, walls)
- Sphericity: How scattered/spherical points are (foliage, shrubs)

Reference: Demantké et al. (2011) - "Dimensionality based scale selection in 3D lidar point clouds"
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GeometricFeatures:
    """Container for PCA-based geometric features."""
    
    verticality: np.ndarray
    """Verticality feature (0-1). High values indicate vertical structures."""
    
    linearity: np.ndarray
    """Linearity feature (0-1). High values indicate linear structures (stems, poles)."""
    
    planarity: np.ndarray
    """Planarity feature (0-1). High values indicate planar structures (ground)."""
    
    sphericity: np.ndarray
    """Sphericity/scattering feature (0-1). High values indicate scattered points (foliage)."""
    
    omnivariance: np.ndarray
    """Omnivariance (cube root of product of eigenvalues). Measures local point scatter."""
    
    eigenvalues: np.ndarray
    """Raw eigenvalues (n, 3) sorted descending: λ1 >= λ2 >= λ3."""
    
    normals: np.ndarray
    """Surface normals (n, 3) - eigenvector of smallest eigenvalue."""


def compute_geometric_features(
    xyz: np.ndarray,
    k_neighbors: int = 20,
    radius: Optional[float] = None,
    verbose: bool = False
) -> GeometricFeatures:
    """
    Compute PCA-based geometric features for each point.
    
    For each point, the local neighborhood is analyzed using PCA to extract
    eigenvalues and eigenvectors. From these, geometric descriptors are derived
    that characterize the local 3D structure.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    k_neighbors : int
        Number of nearest neighbors for KNN search (default: 20).
    radius : float, optional
        If provided, use radius search instead of KNN.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    GeometricFeatures
        Dataclass containing all computed features.
    
    Notes
    -----
    Eigenvalue-based features:
    - linearity = (λ1 - λ2) / λ1
    - planarity = (λ2 - λ3) / λ1
    - sphericity = λ3 / λ1
    - omnivariance = (λ1 * λ2 * λ3)^(1/3)
    
    Verticality is computed from the angle between the surface normal
    (eigenvector of λ3) and the vertical axis [0, 0, 1].
    """
    n_points = len(xyz)
    
    if verbose:
        print(f"Computing geometric features for {n_points:,} points...")
        print(f"  Method: {'Radius=' + str(radius) if radius else 'KNN=' + str(k_neighbors)}")
    
    # Build KD-Tree for efficient neighbor search
    tree = cKDTree(xyz)
    
    # Initialize output arrays
    eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    verticality = np.zeros(n_points, dtype=np.float32)
    linearity = np.zeros(n_points, dtype=np.float32)
    planarity = np.zeros(n_points, dtype=np.float32)
    sphericity = np.zeros(n_points, dtype=np.float32)
    omnivariance = np.zeros(n_points, dtype=np.float32)
    
    # Query neighbors (batch for efficiency)
    if radius is not None:
        # Radius search - variable number of neighbors
        neighbor_lists = tree.query_ball_point(xyz, r=radius, workers=-1)
    else:
        # KNN search - fixed number of neighbors
        _, neighbor_indices = tree.query(xyz, k=k_neighbors, workers=-1)
        neighbor_lists = neighbor_indices
    
    if verbose:
        print("  Computing PCA for each point neighborhood...")
    
    # Compute PCA for each point
    for i in range(n_points):
        if radius is not None:
            neighbors = neighbor_lists[i]
            if len(neighbors) < 3:
                # Not enough neighbors - set default values
                normals[i] = [0, 0, 1]
                continue
            neighbor_xyz = xyz[neighbors]
        else:
            neighbor_xyz = xyz[neighbor_lists[i]]
        
        # Compute covariance matrix
        centered = neighbor_xyz - np.mean(neighbor_xyz, axis=0)
        cov = np.cov(centered.T)
        
        # Compute eigenvalues and eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            normals[i] = [0, 0, 1]
            continue
        
        # Sort eigenvalues descending (eigh returns ascending)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Store eigenvalues
        eigenvalues[i] = eigvals
        
        # Normal = eigenvector of smallest eigenvalue
        normal = eigvecs[:, 2]
        # Orient normal upward
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal
        
        # Compute features (with numerical stability)
        λ1, λ2, λ3 = eigvals
        if λ1 > 1e-10:
            linearity[i] = (λ1 - λ2) / λ1
            planarity[i] = (λ2 - λ3) / λ1
            sphericity[i] = λ3 / λ1
        
        if λ1 > 0 and λ2 > 0 and λ3 > 0:
            omnivariance[i] = np.cbrt(λ1 * λ2 * λ3)
        
        # Verticality = 1 - abs(normal · [0,0,1])
        # High verticality = normal is horizontal = structure is vertical
        verticality[i] = 1.0 - abs(normal[2])
        
        # Progress
        if verbose and (i + 1) % 500000 == 0:
            print(f"  Processed {i + 1:,} / {n_points:,} points ({100*(i+1)/n_points:.1f}%)")
    
    if verbose:
        print(f"  ✓ Feature computation complete")
        print(f"    Verticality: min={verticality.min():.2f}, max={verticality.max():.2f}, mean={verticality.mean():.2f}")
        print(f"    Linearity: min={linearity.min():.2f}, max={linearity.max():.2f}, mean={linearity.mean():.2f}")
        print(f"    Sphericity: min={sphericity.min():.2f}, max={sphericity.max():.2f}, mean={sphericity.mean():.2f}")
    
    return GeometricFeatures(
        verticality=verticality,
        linearity=linearity,
        planarity=planarity,
        sphericity=sphericity,
        omnivariance=omnivariance,
        eigenvalues=eigenvalues,
        normals=normals
    )


def compute_relative_features(
    xyz: np.ndarray,
    cylinder_radius: float = 0.5,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative height features using cylindrical neighborhood search.
    
    For each point, finds all points within a vertical cylinder and computes:
    - dist_to_ground: distance from point to lowest point in cylinder
    - dist_to_top: distance from point to highest point in cylinder
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    cylinder_radius : float
        Horizontal radius of the cylinder for neighbor search.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    dist_to_ground : np.ndarray
        Distance from each point to the lowest point in its cylinder.
    dist_to_top : np.ndarray
        Distance from each point to the highest point in its cylinder.
    
    Notes
    -----
    This uses a 2D KD-Tree on XY coordinates for efficient cylindrical search,
    then computes Z statistics for each cylinder.
    """
    n_points = len(xyz)
    
    if verbose:
        print(f"Computing relative features for {n_points:,} points...")
        print(f"  Cylinder radius: {cylinder_radius}m")
    
    # Build 2D KD-Tree on XY coordinates for cylindrical search
    xy = xyz[:, :2]
    tree_2d = cKDTree(xy)
    
    # Query all neighbors within cylinder radius (2D)
    neighbor_lists = tree_2d.query_ball_point(xy, r=cylinder_radius, workers=-1)
    
    # Initialize output arrays
    dist_to_ground = np.zeros(n_points, dtype=np.float32)
    dist_to_top = np.zeros(n_points, dtype=np.float32)
    
    if verbose:
        print("  Computing Z statistics for each cylinder...")
    
    z = xyz[:, 2]
    
    for i in range(n_points):
        neighbors = neighbor_lists[i]
        if len(neighbors) == 0:
            continue
        
        neighbor_z = z[neighbors]
        z_min = np.min(neighbor_z)
        z_max = np.max(neighbor_z)
        
        dist_to_ground[i] = z[i] - z_min
        dist_to_top[i] = z_max - z[i]
        
        # Progress
        if verbose and (i + 1) % 500000 == 0:
            print(f"  Processed {i + 1:,} / {n_points:,} points ({100*(i+1)/n_points:.1f}%)")
    
    if verbose:
        print(f"  ✓ Relative feature computation complete")
        print(f"    dist_to_ground: min={dist_to_ground.min():.2f}, max={dist_to_ground.max():.2f}")
        print(f"    dist_to_top: min={dist_to_top.min():.2f}, max={dist_to_top.max():.2f}")
    
    return dist_to_ground, dist_to_top


def compute_all_features(
    xyz: np.ndarray,
    k_neighbors: int = 20,
    cylinder_radius: float = 0.5,
    verbose: bool = False
) -> Tuple[GeometricFeatures, np.ndarray, np.ndarray]:
    """
    Compute all geometric and relative features.
    
    Convenience function that calls both compute_geometric_features and
    compute_relative_features.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    k_neighbors : int
        Number of neighbors for PCA computation.
    cylinder_radius : float
        Radius for cylindrical relative feature computation.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    features : GeometricFeatures
        PCA-based geometric features.
    dist_to_ground : np.ndarray
        Distance from each point to ground in cylinder.
    dist_to_top : np.ndarray
        Distance from each point to top in cylinder.
    """
    features = compute_geometric_features(xyz, k_neighbors=k_neighbors, verbose=verbose)
    dist_to_ground, dist_to_top = compute_relative_features(xyz, cylinder_radius=cylinder_radius, verbose=verbose)
    
    return features, dist_to_ground, dist_to_top


def compute_geometric_features_fast(
    xyz: np.ndarray,
    voxel_size: float = 0.1,
    k_neighbors: int = 20,
    verbose: bool = False
) -> GeometricFeatures:
    """
    Optimized PCA features using voxelization + re-projection.
    
    Instead of computing PCA for each of the ~6M points, this function:
    1. Voxelizes the cloud to ~300K voxel centroids
    2. Computes PCA features on the voxels
    3. Re-projects features to original points via nearest-neighbor lookup
    
    This achieves 10-20x speedup with minimal loss of accuracy.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    voxel_size : float
        Size of voxels in meters (default: 0.1m = 10cm).
    k_neighbors : int
        Number of neighbors for PCA computation on voxels.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    GeometricFeatures
        Dataclass containing all computed features (same length as input xyz).
    """
    n_points = len(xyz)
    
    if verbose:
        print(f"Computing geometric features (optimized) for {n_points:,} points...")
        print(f"  Voxel size: {voxel_size}m, K-neighbors: {k_neighbors}")
    
    # ========================================================================
    # Step 1: Voxelize point cloud
    # ========================================================================
    # Compute voxel indices for each point
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    
    # Get unique voxels and mapping
    unique_voxels, inverse_indices = np.unique(
        voxel_indices, axis=0, return_inverse=True
    )
    n_voxels = len(unique_voxels)
    
    if verbose:
        print(f"  Voxelized: {n_points:,} points → {n_voxels:,} voxels ({100*n_voxels/n_points:.1f}%)")
    
    # Compute voxel centroids (average of points in each voxel)
    voxel_centroids = np.zeros((n_voxels, 3), dtype=np.float64)
    voxel_counts = np.zeros(n_voxels, dtype=np.int32)
    
    for i in range(n_points):
        voxel_idx = inverse_indices[i]
        voxel_centroids[voxel_idx] += xyz[i]
        voxel_counts[voxel_idx] += 1
    
    voxel_centroids /= voxel_counts[:, np.newaxis]
    
    # ========================================================================
    # Step 2: Compute PCA features on voxel centroids
    # ========================================================================
    if verbose:
        print(f"  Computing PCA on {n_voxels:,} voxel centroids...")
    
    # Build KD-Tree on voxel centroids
    tree = cKDTree(voxel_centroids)
    
    # Initialize voxel-level feature arrays
    voxel_verticality = np.zeros(n_voxels, dtype=np.float32)
    voxel_linearity = np.zeros(n_voxels, dtype=np.float32)
    voxel_planarity = np.zeros(n_voxels, dtype=np.float32)
    voxel_sphericity = np.zeros(n_voxels, dtype=np.float32)
    voxel_omnivariance = np.zeros(n_voxels, dtype=np.float32)
    voxel_eigenvalues = np.zeros((n_voxels, 3), dtype=np.float32)
    voxel_normals = np.zeros((n_voxels, 3), dtype=np.float32)
    
    # KNN query for all voxels at once
    _, neighbor_indices = tree.query(voxel_centroids, k=min(k_neighbors, n_voxels), workers=-1)
    
    # Compute PCA for each voxel
    for i in range(n_voxels):
        neighbor_xyz = voxel_centroids[neighbor_indices[i]]
        
        if len(neighbor_xyz) < 3:
            voxel_normals[i] = [0, 0, 1]
            continue
        
        # Compute covariance matrix
        centered = neighbor_xyz - np.mean(neighbor_xyz, axis=0)
        cov = np.cov(centered.T)
        
        # Compute eigenvalues and eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            voxel_normals[i] = [0, 0, 1]
            continue
        
        # Sort eigenvalues descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Store eigenvalues
        voxel_eigenvalues[i] = eigvals
        
        # Normal = eigenvector of smallest eigenvalue
        normal = eigvecs[:, 2]
        if normal[2] < 0:
            normal = -normal
        voxel_normals[i] = normal
        
        # Compute features
        λ1, λ2, λ3 = eigvals
        if λ1 > 1e-10:
            voxel_linearity[i] = (λ1 - λ2) / λ1
            voxel_planarity[i] = (λ2 - λ3) / λ1
            voxel_sphericity[i] = λ3 / λ1
        
        if λ1 > 0 and λ2 > 0 and λ3 > 0:
            voxel_omnivariance[i] = np.cbrt(λ1 * λ2 * λ3)
        
        voxel_verticality[i] = 1.0 - abs(normal[2])
        
        # Progress
        if verbose and (i + 1) % 100000 == 0:
            print(f"    Processed {i + 1:,} / {n_voxels:,} voxels ({100*(i+1)/n_voxels:.1f}%)")
    
    # ========================================================================
    # Step 3: Re-project features to original points
    # ========================================================================
    if verbose:
        print(f"  Re-projecting features to {n_points:,} original points...")
    
    # Use inverse_indices to map voxel features to original points
    verticality = voxel_verticality[inverse_indices]
    linearity = voxel_linearity[inverse_indices]
    planarity = voxel_planarity[inverse_indices]
    sphericity = voxel_sphericity[inverse_indices]
    omnivariance = voxel_omnivariance[inverse_indices]
    eigenvalues = voxel_eigenvalues[inverse_indices]
    normals = voxel_normals[inverse_indices]
    
    if verbose:
        print(f"  ✓ Feature computation complete (optimized)")
        print(f"    Verticality: min={verticality.min():.2f}, max={verticality.max():.2f}, mean={verticality.mean():.2f}")
        print(f"    Linearity: min={linearity.min():.2f}, max={linearity.max():.2f}, mean={linearity.mean():.2f}")
        print(f"    Sphericity: min={sphericity.min():.2f}, max={sphericity.max():.2f}, mean={sphericity.mean():.2f}")
    
    return GeometricFeatures(
        verticality=verticality,
        linearity=linearity,
        planarity=planarity,
        sphericity=sphericity,
        omnivariance=omnivariance,
        eigenvalues=eigenvalues,
        normals=normals
    )


def compute_all_features_fast(
    xyz: np.ndarray,
    voxel_size: float = 0.1,
    k_neighbors: int = 20,
    cylinder_radius: float = 0.5,
    verbose: bool = False
) -> Tuple[GeometricFeatures, np.ndarray, np.ndarray]:
    """
    Compute all features using optimized voxel-based method.
    
    This is the recommended function for large point clouds (>1M points).
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    voxel_size : float
        Size of voxels for PCA optimization (default: 0.1m).
    k_neighbors : int
        Number of neighbors for PCA computation.
    cylinder_radius : float
        Radius for cylindrical relative feature computation.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    features : GeometricFeatures
        PCA-based geometric features.
    dist_to_ground : np.ndarray
        Distance from each point to ground in cylinder.
    dist_to_top : np.ndarray
        Distance from each point to top in cylinder.
    """
    features = compute_geometric_features_fast(
        xyz, voxel_size=voxel_size, k_neighbors=k_neighbors, verbose=verbose
    )
    dist_to_ground, dist_to_top = compute_relative_features_fast(
        xyz, voxel_size=voxel_size, verbose=verbose
    )
    
    return features, dist_to_ground, dist_to_top


def compute_relative_features_fast(
    xyz: np.ndarray,
    voxel_size: float = 0.1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized relative height features using 2D voxel grid.
    
    Instead of querying neighbors for each point individually, this:
    1. Creates a 2D grid (XY) with voxel_size resolution
    2. For each grid cell, computes z_min and z_max of all points in that column
    3. Maps results back to original points via grid cell lookup
    
    This achieves 10-20x speedup with acceptable accuracy for forestry applications.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    voxel_size : float
        Size of 2D grid cells in meters (default: 0.1m).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    dist_to_ground : np.ndarray
        Distance from each point to the lowest point in its grid column.
    dist_to_top : np.ndarray
        Distance from each point to the highest point in its grid column.
    """
    n_points = len(xyz)
    
    if verbose:
        print(f"Computing relative features (optimized) for {n_points:,} points...")
        print(f"  Grid cell size: {voxel_size}m")
    
    # ========================================================================
    # Step 1: Create 2D grid indices
    # ========================================================================
    xy = xyz[:, :2]
    z = xyz[:, 2]
    
    # Compute grid cell indices for each point (2D only)
    grid_indices = np.floor(xy / voxel_size).astype(np.int32)
    
    # Create unique key for each grid cell
    # Offset to handle negative coordinates
    offset = np.min(grid_indices, axis=0)
    grid_indices_offset = grid_indices - offset
    
    # Create unique integer key for each cell
    max_y = grid_indices_offset[:, 1].max() + 1
    cell_keys = grid_indices_offset[:, 0] * max_y + grid_indices_offset[:, 1]
    
    # Get unique cells and mapping
    unique_cells, inverse_indices = np.unique(cell_keys, return_inverse=True)
    n_cells = len(unique_cells)
    
    if verbose:
        print(f"  Grid: {n_points:,} points → {n_cells:,} cells ({100*n_cells/n_points:.2f}%)")
    
    # ========================================================================
    # Step 2: Compute z_min and z_max for each grid cell
    # ========================================================================
    if verbose:
        print("  Computing Z statistics per grid cell...")
    
    # Initialize with extreme values
    cell_z_min = np.full(n_cells, np.inf, dtype=np.float32)
    cell_z_max = np.full(n_cells, -np.inf, dtype=np.float32)
    
    # Compute min/max per cell using np.minimum.at and np.maximum.at
    np.minimum.at(cell_z_min, inverse_indices, z)
    np.maximum.at(cell_z_max, inverse_indices, z)
    
    # ========================================================================
    # Step 3: Map results back to original points
    # ========================================================================
    if verbose:
        print(f"  Mapping results to {n_points:,} original points...")
    
    # Each point gets the z_min/z_max of its grid cell
    point_z_min = cell_z_min[inverse_indices]
    point_z_max = cell_z_max[inverse_indices]
    
    dist_to_ground = (z - point_z_min).astype(np.float32)
    dist_to_top = (point_z_max - z).astype(np.float32)
    
    if verbose:
        print(f"  ✓ Relative feature computation complete (optimized)")
        print(f"    dist_to_ground: min={dist_to_ground.min():.2f}, max={dist_to_ground.max():.2f}")
        print(f"    dist_to_top: min={dist_to_top.min():.2f}, max={dist_to_top.max():.2f}")
    
    return dist_to_ground, dist_to_top


