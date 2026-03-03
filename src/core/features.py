"""
Geometric feature extraction module — pgeof backend only.

Provides fast, C++-backed geometric feature computation for point clouds
using the pgeof library (Point Geometric Features).

Features used in the pipeline:
- Verticality: identifies stems/trunks (vertical structures)
- Linearity:   identifies branches (linear structures, any orientation)
"""

from typing import Tuple, Optional
import numpy as np

try:
    import pgeof
    from pgeof import EFeatureID
    _HAS_PGEOF = True
except ImportError:
    _HAS_PGEOF = False


# ---------------------------------------------------------------------------
# Voxelization helper
# ---------------------------------------------------------------------------

def voxelize_cloud(
    xyz: np.ndarray,
    resolution_xy: float = 0.02,
    resolution_z: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Voxelize points and return voxel centroids plus point-to-voxel mapping.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point cloud.
    resolution_xy : float
        Horizontal voxel size.
    resolution_z : float
        Vertical voxel size.

    Returns
    -------
    voxel_centroids : np.ndarray
        (M, 3) voxel centroids.
    point_to_voxel : np.ndarray
        (N,) voxel index per input point.
    n_voxels : int
        Number of unique voxels.
    """
    resolution = np.array([resolution_xy, resolution_xy, resolution_z])
    voxel_indices = np.floor(xyz / resolution).astype(np.int64)

    # Unique voxels
    _, inverse, counts = np.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True,
    )

    n_voxels = len(counts)
    # Compute centroids by accumulating
    centroids = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(centroids, inverse, xyz)
    centroids /= counts[:, None]

    return centroids, inverse, n_voxels


# ---------------------------------------------------------------------------
# Feature computation — pgeof backend
# ---------------------------------------------------------------------------

def compute_verticality(
    xyz: np.ndarray,
    scale: float = 0.1,
    max_knn: int = 50000,
    voxel_resolution_xy: float = 0.02,
    voxel_resolution_z: float = 0.02,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute verticality for each point using pgeof (C++ backend).

    Operates on voxel centroids and re-projects results to original points.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point cloud.
    scale : float
        Radius for neighbourhood search during PCA.
    max_knn : int
        Maximum neighbours for radius search.
    voxel_resolution_xy, voxel_resolution_z : float
        Voxel resolution for subsampling.
    verbose : bool
        Print progress information.

    Returns
    -------
    verticality : np.ndarray
        (N,) verticality values in [0, 1].
    """
    if not _HAS_PGEOF:
        raise ImportError(
            "pgeof is required. Install with: pip install pgeof"
        )

    if verbose:
        print(f"  Computing verticality: {len(xyz):,} points, scale={scale}m")

    centroids, point_to_voxel, n_voxels = voxelize_cloud(
        xyz, resolution_xy=voxel_resolution_xy, resolution_z=voxel_resolution_z,
    )
    if verbose:
        print(f"  Voxelized: {len(xyz):,} → {n_voxels:,} voxels")

    vert = pgeof.compute_features_selected(
        centroids, scale, max_knn, [EFeatureID.Verticality],
    )
    # vert is (n_voxels, 1), flatten
    vert = vert.ravel()

    # Re-project to original points
    return vert[point_to_voxel]


def compute_linearity(
    xyz: np.ndarray,
    scale: float = 0.1,
    max_knn: int = 50000,
    voxel_resolution_xy: float = 0.02,
    voxel_resolution_z: float = 0.02,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute linearity for each point using pgeof (C++ backend).

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point cloud.
    scale : float
        Radius for neighbourhood search.
    max_knn : int
        Maximum neighbours.
    voxel_resolution_xy, voxel_resolution_z : float
        Voxel resolution.
    verbose : bool
        Print progress.

    Returns
    -------
    linearity : np.ndarray
        (N,) linearity values in [0, 1].
    """
    if not _HAS_PGEOF:
        raise ImportError(
            "pgeof is required. Install with: pip install pgeof"
        )

    if verbose:
        print(f"  Computing linearity: {len(xyz):,} points, scale={scale}m")

    centroids, point_to_voxel, n_voxels = voxelize_cloud(
        xyz, resolution_xy=voxel_resolution_xy, resolution_z=voxel_resolution_z,
    )
    if verbose:
        print(f"  Voxelized: {len(xyz):,} → {n_voxels:,} voxels")

    lin = pgeof.compute_features_selected(
        centroids, scale, max_knn, [EFeatureID.Linearity],
    )
    return lin.ravel()[point_to_voxel]


def compute_wood_features(
    xyz: np.ndarray,
    scale: float = 0.1,
    max_knn: int = 50000,
    voxel_resolution_xy: float = 0.02,
    voxel_resolution_z: float = 0.02,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both verticality and linearity in a single voxelization pass.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point cloud.
    scale : float
        Radius for neighbourhood search.
    max_knn : int
        Maximum neighbours.
    voxel_resolution_xy, voxel_resolution_z : float
        Voxel resolution.
    verbose : bool
        Print progress.

    Returns
    -------
    verticality : np.ndarray
        (N,) verticality values.
    linearity : np.ndarray
        (N,) linearity values.
    """
    if not _HAS_PGEOF:
        raise ImportError(
            "pgeof is required. Install with: pip install pgeof"
        )

    if verbose:
        print(f"  Computing wood features: {len(xyz):,} points, scale={scale}m")

    centroids, point_to_voxel, n_voxels = voxelize_cloud(
        xyz, resolution_xy=voxel_resolution_xy, resolution_z=voxel_resolution_z,
    )
    if verbose:
        print(f"  Voxelized: {len(xyz):,} → {n_voxels:,} voxels")

    features = pgeof.compute_features_selected(
        centroids, scale, max_knn,
        [EFeatureID.Verticality, EFeatureID.Linearity],
    )
    # features is (n_voxels, 2)
    vert = features[:, 0][point_to_voxel]
    lin = features[:, 1][point_to_voxel]

    if verbose:
        print(f"  Verticality: mean={vert.mean():.3f}, p95={np.percentile(vert, 95):.3f}")
        print(f"  Linearity:   mean={lin.mean():.3f}, p95={np.percentile(lin, 95):.3f}")

    return vert, lin
