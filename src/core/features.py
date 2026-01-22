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
