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
from typing import Tuple, Optional, Literal, Sequence, Dict, Any
import numpy as np
from scipy.spatial import cKDTree
import time


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


def _voxelize_cloud(
    xyz: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Voxelize points and return voxel centroids plus point-to-voxel mapping.

    Returns
    -------
    voxel_centroids : np.ndarray
        (M, 3) voxel centroids.
    inverse_indices : np.ndarray
        (N,) voxel index per input point.
    n_voxels : int
        Number of unique voxels.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    n_voxels = int(inverse_indices.max()) + 1 if len(inverse_indices) > 0 else 0

    voxel_centroids = np.zeros((n_voxels, 3), dtype=np.float64)
    if n_voxels == 0:
        return voxel_centroids, inverse_indices, n_voxels

    voxel_counts = np.bincount(inverse_indices, minlength=n_voxels).astype(np.int32)
    np.add.at(voxel_centroids, inverse_indices, xyz)
    voxel_centroids /= np.maximum(voxel_counts, 1)[:, np.newaxis]

    return voxel_centroids, inverse_indices, n_voxels


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
    if n_points == 0:
        empty = np.array([], dtype=np.float32)
        return GeometricFeatures(
            verticality=empty,
            linearity=empty,
            planarity=empty,
            sphericity=empty,
            omnivariance=empty,
            eigenvalues=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
        )
    
    if verbose:
        print(f"Computing geometric features (optimized) for {n_points:,} points...")
        print(f"  Voxel size: {voxel_size}m, K-neighbors: {k_neighbors}")
    
    # ========================================================================
    # Step 1: Voxelize point cloud
    # ========================================================================
    voxel_centroids, inverse_indices, n_voxels = _voxelize_cloud(xyz, voxel_size)
    
    if verbose:
        print(f"  Voxelized: {n_points:,} points → {n_voxels:,} voxels ({100*n_voxels/n_points:.1f}%)")
    
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


def compute_geometric_features_pgeof(
    xyz: np.ndarray,
    voxel_size: float = 0.1,
    scale: float = 0.15,
    max_knn: int = 50000,
    verbose: bool = False
) -> GeometricFeatures:
    """
    Compute geometric features with pgeof (C++ backend), then re-project.

    Notes
    -----
    - `pgeof` provides `Scattering`; this is mapped to `sphericity`.
    - `pgeof` does not expose eigenvalues/omnivariance in this API, so those are
      returned as zeros for compatibility with `GeometricFeatures`.
    """
    try:
        import pgeof
        from pgeof import EFeatureID
    except ImportError as exc:
        raise ImportError(
            "pgeof backend requested but package is not installed. "
            "Install with: pip install pgeof"
        ) from exc

    n_points = len(xyz)
    if n_points == 0:
        empty = np.array([], dtype=np.float32)
        return GeometricFeatures(
            verticality=empty,
            linearity=empty,
            planarity=empty,
            sphericity=empty,
            omnivariance=empty,
            eigenvalues=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
        )

    if verbose:
        print(f"Computing geometric features (pgeof) for {n_points:,} points...")
        print(f"  Voxel size: {voxel_size}m, scale={scale}, max_knn={max_knn}")

    voxel_centroids, inverse_indices, n_voxels = _voxelize_cloud(xyz, voxel_size)
    if verbose:
        print(f"  Voxelized: {n_points:,} points -> {n_voxels:,} voxels ({100*n_voxels/n_points:.1f}%)")

    selected_features = [
        EFeatureID.Verticality,
        EFeatureID.Linearity,
        EFeatureID.Planarity,
        EFeatureID.Scattering,
        EFeatureID.Normal_x,
        EFeatureID.Normal_y,
        EFeatureID.Normal_z,
    ]

    voxel_features = pgeof.compute_features_selected(
        voxel_centroids.astype(np.float32),
        scale,
        max_knn,
        selected_features,
    )
    voxel_features = np.asarray(voxel_features, dtype=np.float32)
    if voxel_features.ndim != 2 or voxel_features.shape[1] != len(selected_features):
        raise RuntimeError(
            f"Unexpected pgeof output shape {voxel_features.shape}, "
            f"expected (n_voxels, {len(selected_features)})"
        )

    voxel_verticality = voxel_features[:, 0]
    voxel_linearity = voxel_features[:, 1]
    voxel_planarity = voxel_features[:, 2]
    voxel_sphericity = voxel_features[:, 3]  # scattering
    voxel_normals = voxel_features[:, 4:7]

    verticality = voxel_verticality[inverse_indices]
    linearity = voxel_linearity[inverse_indices]
    planarity = voxel_planarity[inverse_indices]
    sphericity = voxel_sphericity[inverse_indices]
    normals = voxel_normals[inverse_indices]

    if verbose:
        print("  Feature computation complete (pgeof)")
        print(f"    Verticality: min={verticality.min():.2f}, max={verticality.max():.2f}, mean={verticality.mean():.2f}")
        print(f"    Linearity: min={linearity.min():.2f}, max={linearity.max():.2f}, mean={linearity.mean():.2f}")
        print(f"    Sphericity: min={sphericity.min():.2f}, max={sphericity.max():.2f}, mean={sphericity.mean():.2f}")

    return GeometricFeatures(
        verticality=verticality,
        linearity=linearity,
        planarity=planarity,
        sphericity=sphericity,
        omnivariance=np.zeros(n_points, dtype=np.float32),
        eigenvalues=np.zeros((n_points, 3), dtype=np.float32),
        normals=normals,
    )


def compute_all_features_fast(
    xyz: np.ndarray,
    voxel_size: float = 0.1,
    k_neighbors: int = 20,
    cylinder_radius: float = 0.5,
    verbose: bool = False,
    backend: Optional[Literal["voxel", "pgeof"]] = None,
    pgeof_scale: float = 0.15,
    pgeof_max_knn: int = 50000,
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
    backend : {"voxel", "pgeof"} | None
        Geometric feature backend. If None, uses env var
        `PS_LIDAR_FEATURE_BACKEND` (default: "voxel").
    pgeof_scale : float
        Neighborhood scale for pgeof backend.
    pgeof_max_knn : int
        Max neighbors for pgeof backend.
    
    Returns
    -------
    features : GeometricFeatures
        PCA-based geometric features.
    dist_to_ground : np.ndarray
        Distance from each point to ground in cylinder.
    dist_to_top : np.ndarray
        Distance from each point to top in cylinder.
    """
    import os

    selected_backend = (backend or os.getenv("PS_LIDAR_FEATURE_BACKEND", "voxel")).lower()
    if selected_backend not in {"voxel", "pgeof"}:
        raise ValueError(
            f"Unsupported backend '{selected_backend}'. Use 'voxel' or 'pgeof'."
        )

    if selected_backend == "voxel":
        features = compute_geometric_features_fast(
            xyz, voxel_size=voxel_size, k_neighbors=k_neighbors, verbose=verbose
        )
    else:
        features = compute_geometric_features_pgeof(
            xyz,
            voxel_size=voxel_size,
            scale=pgeof_scale,
            max_knn=pgeof_max_knn,
            verbose=verbose,
        )

    dist_to_ground, dist_to_top = compute_relative_features_fast(
        xyz, voxel_size=voxel_size, verbose=verbose
    )
    
    return features, dist_to_ground, dist_to_top


def benchmark_feature_backends(
    xyz: np.ndarray,
    backends: Sequence[str] = ("voxel", "pgeof"),
    repeats: int = 1,
    sample_size: Optional[int] = None,
    seed: int = 42,
    voxel_size: float = 0.1,
    k_neighbors: int = 20,
    pgeof_scale: float = 0.15,
    pgeof_max_knn: int = 50000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Reproducible benchmark for geometric-feature backends.

    Returns a report dictionary with timing stats and feature deltas.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    rng = np.random.default_rng(seed)
    xyz_eval = xyz
    if sample_size is not None and sample_size < len(xyz):
        idx = rng.choice(len(xyz), size=sample_size, replace=False)
        xyz_eval = xyz[idx]

    if verbose:
        print(f"Benchmarking backends on {len(xyz_eval):,} points (repeats={repeats})")

    report: Dict[str, Any] = {
        "metadata": {
            "n_points_original": int(len(xyz)),
            "n_points_evaluated": int(len(xyz_eval)),
            "sample_size": None if sample_size is None else int(sample_size),
            "seed": int(seed),
            "repeats": int(repeats),
            "voxel_size": float(voxel_size),
            "k_neighbors": int(k_neighbors),
            "pgeof_scale": float(pgeof_scale),
            "pgeof_max_knn": int(pgeof_max_knn),
        },
        "backends": {},
        "comparisons": {},
    }

    first_run_features: Dict[str, GeometricFeatures] = {}

    for backend in backends:
        backend_name = backend.lower()
        durations = []
        error = None

        for run_idx in range(repeats):
            t0 = time.perf_counter()
            try:
                features, _, _ = compute_all_features_fast(
                    xyz_eval,
                    voxel_size=voxel_size,
                    k_neighbors=k_neighbors,
                    verbose=False,
                    backend=backend_name,
                    pgeof_scale=pgeof_scale,
                    pgeof_max_knn=pgeof_max_knn,
                )
                if run_idx == 0:
                    first_run_features[backend_name] = features
            except Exception as exc:
                error = str(exc)
                break
            durations.append(time.perf_counter() - t0)

        if error is not None:
            report["backends"][backend_name] = {
                "ok": False,
                "error": error,
            }
            continue

        f0 = first_run_features[backend_name]
        report["backends"][backend_name] = {
            "ok": True,
            "timing_seconds": {
                "runs": [float(v) for v in durations],
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
            },
            "feature_summary": {
                "verticality_mean": float(np.mean(f0.verticality)),
                "linearity_mean": float(np.mean(f0.linearity)),
                "planarity_mean": float(np.mean(f0.planarity)),
                "sphericity_mean": float(np.mean(f0.sphericity)),
            },
        }

    if {"voxel", "pgeof"}.issubset(first_run_features.keys()):
        fv = first_run_features["voxel"]
        fp = first_run_features["pgeof"]
        report["comparisons"]["pgeof_vs_voxel_mae"] = {
            "verticality": float(np.mean(np.abs(fp.verticality - fv.verticality))),
            "linearity": float(np.mean(np.abs(fp.linearity - fv.linearity))),
            "planarity": float(np.mean(np.abs(fp.planarity - fv.planarity))),
            "sphericity": float(np.mean(np.abs(fp.sphericity - fv.sphericity))),
        }

    return report


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


def estimate_local_radius(
    xyz: np.ndarray,
    search_radius: float = 0.15,
    search_height: float = 0.3,
    voxel_size: float = 0.1,
    k_neighbors: int = 20,
    verbose: bool = False
) -> np.ndarray:
    """
    Estimate local stem radius for each point based on horizontal point distribution.
    
    OPTIMIZED: Uses voxel subsampling to speed up query time. Instead of querying
    every point, we query voxel centers and broadcast results to points within each voxel.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    search_radius : float
        Horizontal search radius for neighbors (default: 0.15m).
    search_height : float
        Vertical search range ±height/2 (default: 0.3m).
    voxel_size : float
        Voxel size for subsampling (default: 0.1m).
    k_neighbors : int
        Minimum number of neighbors to consider (default: 20).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Estimated radius for each point (meters).
    """
    n_points = len(xyz)
    
    if verbose:
        print(f"Estimating local radius (optimized) for {n_points:,} points...")
        print(f"  Configuration: search_radius={search_radius}m, voxel_size={voxel_size}m")
    
    # ========================================================================
    # Step 1: Voxelize to reduce number of queries
    # ========================================================================
    if verbose:
        print("  Voxelizing point cloud...")
        
    # Compute grid indices
    grid_indices = np.floor(xyz / voxel_size).astype(np.int32)
    
    # Create unique keys
    # Map 3D indices to unique string or tuple or just use np.unique with axis
    # Using structured array for performance
    dtype = [('x', np.int32), ('y', np.int32), ('z', np.int32)]
    grid_indices_struct = np.empty(n_points, dtype=dtype)
    grid_indices_struct['x'] = grid_indices[:, 0]
    grid_indices_struct['y'] = grid_indices[:, 1]
    grid_indices_struct['z'] = grid_indices[:, 2]
    
    # Get unique voxels and inverse mapping (to map back to points)
    unique_voxels, inverse_indices = np.unique(grid_indices_struct, return_inverse=True)
    n_voxels = len(unique_voxels)
    
    # Compute voxel centroids
    voxel_centers = np.column_stack([
        unique_voxels['x'],
        unique_voxels['y'],
        unique_voxels['z']
    ]).astype(np.float32) * voxel_size + (voxel_size / 2)
    
    if verbose:
        print(f"  Voxel grid: {n_voxels:,} voxels ({100*n_voxels/n_points:.1f}% of original)")
        print("  Building KDTree on original points...")
        
    # ========================================================================
    # Step 2: Query radius for each voxel centroid
    # ========================================================================
    # We query the ORIGINAL points tree to get accurate neighbors
    tree = cKDTree(xyz)
    
    voxel_radii = np.zeros(n_voxels, dtype=np.float32)
    
    # Query in batches
    batch_size = 50000
    n_batches = (n_voxels + batch_size - 1) // batch_size
    
    if verbose:
        print(f"  Computing radius for {n_voxels:,} voxels in {n_batches} batches...")
        
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, n_voxels)
        batch_centers = voxel_centers[start:end]
        
        # 1. Broad search: find neighbors within search_radius
        # This returns list of lists
        neighbor_indices_list = tree.query_ball_point(batch_centers, r=search_radius)
        
        batch_radii = np.zeros(len(batch_centers), dtype=np.float32)
        
        for i, neighbor_indices in enumerate(neighbor_indices_list):
            if len(neighbor_indices) < k_neighbors:
                continue
                
            neighbors = xyz[neighbor_indices]
            center = batch_centers[i]
            
            # 2. Strict height filter
            dz = np.abs(neighbors[:, 2] - center[2])
            height_mask = dz <= (search_height / 2)
            
            valid_neighbors = neighbors[height_mask]
            
            if len(valid_neighbors) < k_neighbors:
                continue
                
            # 3. Horizontal distance
            d_xy = np.linalg.norm(valid_neighbors[:, :2] - center[:2], axis=1)
            
            # 4. Estimate radius (90th percentile)
            batch_radii[i] = np.percentile(d_xy, 90)
            
        voxel_radii[start:end] = batch_radii
        
        if verbose and (batch_idx + 1) % max(1, n_batches // 5) == 0:
            print(f"    Progress: {100*(batch_idx+1)/n_batches:.0f}%")
            
    # ========================================================================
    # Step 3: Map back to original points
    # ========================================================================
    if verbose:
        print("  Mapping results to original points...")
        
    point_radii = voxel_radii[inverse_indices]
    
    if verbose:
        print(f"  ✓ Radius estimation complete (optimized)")
        print(f"    Median radius: {np.median(point_radii[point_radii > 0]):.3f}m")
        
    return point_radii
