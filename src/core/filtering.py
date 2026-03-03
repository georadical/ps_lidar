"""
Filtering Module

Provides functions for point cloud noise filtering:
- Statistical Outlier Removal (SOR)
- Radius Outlier Removal
"""

import numpy as np
from dataclasses import dataclass
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
    
    Uses a highly optimized, multithreaded, and chunked `scipy.spatial.cKDTree`
    approach to keep memory usage low and speed high for 200M+ point clouds.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) array of points.
    k_neighbors : int
        Number of neighbors to analyze for each point.
    std_ratio : float
        Standard deviation multiplier threshold.
    verbose : bool
        Print progress information.

    Returns
    -------
    NoiseFilterResult
        Cleaned points and statistics.
    """
    from scipy.spatial import cKDTree
    import time    

    n_original = len(xyz)

    if n_original < k_neighbors + 1:
        warnings.warn(
            f"Not enough points ({n_original}) for SOR with k={k_neighbors}"
        )
        return NoiseFilterResult(
            clean_xyz=xyz.copy(),
            clean_indices=np.arange(n_original),
            noise_mask=np.zeros(n_original, dtype=bool),
            n_removed=0,
            removal_percentage=0.0,
        )

    if verbose:
        print(f"SOR filtering: {n_original:,} points, k={k_neighbors}, std={std_ratio}")
        t0 = time.perf_counter()

    # 1. Build tree (fast, multithreaded under the hood in modern scipy)
    tree = cKDTree(xyz)
    if verbose:
        print(f"  Tree built in {time.perf_counter() - t0:.1f}s")
        t1 = time.perf_counter()

    # 2. Query distances in chunks to avoid OOM
    # For a 218M cloud, returning all k-distances at once needs 218M * 11 * 8 bytes ≈ 19 GB just for distances
    chunk_size = 5_000_000
    n_chunks = int(np.ceil(n_original / chunk_size))
    
    mean_distances = np.zeros(n_original, dtype=np.float32)
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_original)
        
        # k_neighbors + 1 because the point itself is found at distance 0
        distances, _ = tree.query(
            xyz[start_idx:end_idx], 
            k=k_neighbors + 1, 
            workers=1
        )
        
        # average distance to the k neighbors (excluding the point itself)
        mean_distances[start_idx:end_idx] = np.mean(distances[:, 1:], axis=1).astype(np.float32)
        
        if verbose and i % 5 == 0 and i > 0:
            print(f"  Processed {i}/{n_chunks} chunks...")

    if verbose:
        print(f"  Distances computed in {time.perf_counter() - t1:.1f}s")

    # 3. Calculate global threshold
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    threshold = global_mean + (std_ratio * global_std)

    # 4. Filter
    # inliers are those with a mean distance LESS than the threshold
    inlier_mask = mean_distances <= threshold
    inlier_indices = np.where(inlier_mask)[0]
    
    noise_mask = ~inlier_mask
    n_removed = n_original - len(inlier_indices)
    removal_percentage = 100 * n_removed / n_original

    if verbose:
        print(f"  Global mean: {global_mean:.4f}m, std: {global_std:.4f}m, threshold: {threshold:.4f}m")
        print(f"  Removed: {n_removed:,} points ({removal_percentage:.2f}%)")
        print(f"  Remaining: {len(inlier_indices):,} points")

    out = NoiseFilterResult(
        clean_xyz=xyz[inlier_indices],
        clean_indices=inlier_indices,
        noise_mask=noise_mask,
        n_removed=n_removed,
        removal_percentage=removal_percentage,
    )
    
    # force gc
    del mean_distances, tree
    
    return out


def filter_noise_radius(
    xyz: np.ndarray,
    radius: float = 0.1,
    min_neighbors: int = 5,
    verbose: bool = False
) -> NoiseFilterResult:
    """
    Radius Outlier Removal for point cloud noise filtering.

    Removes points that have fewer than `min_neighbors` within `radius`.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) array of points.
    radius : float
        Search radius in meters.
    min_neighbors : int
        Minimum neighbors required to keep point.
    verbose : bool
        Print progress information.

    Returns
    -------
    NoiseFilterResult
        Cleaned points and statistics.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Open3D is required for radius filtering. Install with: pip install open3d"
        )

    n_original = len(xyz)

    if verbose:
        print(f"Radius filtering: {n_original:,} points, r={radius}m, min_n={min_neighbors}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pcd_clean, inlier_indices = pcd.remove_radius_outlier(
        nb_points=min_neighbors,
        radius=radius,
    )

    inlier_indices = np.array(inlier_indices)

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
        removal_percentage=removal_percentage,
    )
