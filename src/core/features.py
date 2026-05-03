"""
Geometric feature extraction module — pgeof backend only.

Provides fast, C++-backed geometric feature computation for point clouds
using the pgeof library (Point Geometric Features).

Features used in the pipeline:
- Verticality: identifies stems/trunks (vertical structures)
- Linearity:   identifies branches (linear structures, any orientation)
"""

import math
from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree

try:
    import pgeof
    from pgeof import EFeatureID
    _HAS_PGEOF = True
except ImportError:
    _HAS_PGEOF = False


_MIN_NEIGHBORS = 5
_NEIGHBORHOOD_FACTOR = 3.0
_EPS = 1e-12


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

    # Fast unique via structured-dtype view.
    # np.unique(axis=0) on (N, 3) is very slow because it sorts each
    # column independently.  Viewing the rows as a single structured
    # element turns this into a 1D unique, which is O(N log N) on a
    # contiguous key — typically 4-10× faster for large N.
    voxel_indices_c = np.ascontiguousarray(voxel_indices)
    row_dtype = np.dtype((np.void, voxel_indices_c.dtype.itemsize * voxel_indices_c.shape[1]))
    flat_view = voxel_indices_c.view(row_dtype).ravel()
    _, inverse, counts = np.unique(flat_view, return_inverse=True, return_counts=True)

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


def compute_verticality_mask_early_exit(
    xyz: np.ndarray,
    threshold: float,
    scale: float = 0.1,
    max_knn: int = 50000,
    voxel_resolution_xy: float = 0.02,
    voxel_resolution_z: float = 0.02,
    coarse_resolution: float = 0.08,
    margin: float = 0.1,
) -> Tuple[np.ndarray, dict]:
    """Two-tier verticality screening with voxel-level early exit.

    Tier 1 – **coarse pass** (fast, conservative):
        Voxelize at ``coarse_resolution``, run pgeof on coarse centroids.
        Coarse voxels with verticality ≥ ``threshold + margin`` are marked
        as **auto-keep**: their points will be kept regardless of the
        fine-pass result.

    Tier 2 – **fine pass** (only on non-auto-keep points):
        Run ``compute_verticality()`` at the full resolution on only the
        points NOT in auto-keep voxels.  Apply the normal threshold.

    The coarse pass **never rejects** a point that would otherwise be kept.
    The only possible difference from the baseline is that points in
    auto-keep voxels are kept even if their fine-pass verticality would
    have been below threshold — but this is extremely rare since auto-keep
    voxels have coarse vert ≥ threshold + margin.

    Parameters
    ----------
    xyz : (N, 3) point cloud.
    threshold : verticality threshold for keep/reject.
    scale : PCA neighbourhood radius.
    max_knn : maximum neighbours for radius search.
    voxel_resolution_xy, voxel_resolution_z : fine voxel resolution.
    coarse_resolution : coarse voxel resolution for tier-1 screening.
    margin : verticality margin above threshold for auto-keep.

    Returns
    -------
    keep_mask : np.ndarray
        (N,) boolean — True = keep, False = reject.
    stats : dict
        Tier-1 and tier-2 statistics.
    """
    if not _HAS_PGEOF:
        raise ImportError("pgeof is required. Install with: pip install pgeof")

    N = len(xyz)

    # --- Tier 1: coarse pass (keep-only) ---
    coarse_centroids, coarse_ptv, n_coarse = voxelize_cloud(
        xyz, resolution_xy=coarse_resolution, resolution_z=coarse_resolution,
    )
    coarse_vert = pgeof.compute_features_selected(
        coarse_centroids, scale, max_knn, [EFeatureID.Verticality],
    ).ravel()

    vox_auto_keep = coarse_vert >= (threshold + margin)
    point_auto_keep = vox_auto_keep[coarse_ptv]
    n_coarse_keep = int(vox_auto_keep.sum())
    n_points_coarse_keep = int(point_auto_keep.sum())

    # --- Tier 2: fine pass on remaining points only ---
    need_fine = ~point_auto_keep
    n_fine = int(need_fine.sum())

    keep_mask = np.ones(N, dtype=bool)          # auto-keep start as True

    if n_fine > 0:
        fine_indices = np.where(need_fine)[0]
        fine_pts = xyz[fine_indices]
        fine_vert = compute_verticality(
            fine_pts,
            scale=scale,
            max_knn=max_knn,
            voxel_resolution_xy=voxel_resolution_xy,
            voxel_resolution_z=voxel_resolution_z,
        )
        keep_mask[fine_indices] = fine_vert >= threshold

    stats = {
        "n_coarse_voxels": n_coarse,
        "n_coarse_keep": n_coarse_keep,
        "n_points_coarse_keep": n_points_coarse_keep,
        "n_points_fine_pass": n_fine,
    }
    return keep_mask, stats


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


def _compute_local_shape_features(
    centroids: np.ndarray,
    counts: np.ndarray,
    voxel_size: float,
) -> dict[str, np.ndarray]:
    """
    Compute local PCA shape features on voxel centroids and reproject later.

    This mirrors the geometry used by the sample-bank builder so the exported
    Module 7 features and the future ML dataset are consistent.
    """
    n_voxels = len(centroids)
    if n_voxels == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "planarity": empty,
            "sphericity": empty,
            "anisotropy": empty,
            "surface_variation": empty,
            "roughness": empty,
            "neighbor_count": empty,
            "volume_density": empty,
        }

    radius = max(voxel_size * _NEIGHBORHOOD_FACTOR, voxel_size + 0.05)
    sphere_volume = (4.0 / 3.0) * math.pi * (radius**3)
    tree = cKDTree(centroids)

    planarity = np.zeros(n_voxels, dtype=np.float64)
    sphericity = np.zeros(n_voxels, dtype=np.float64)
    anisotropy = np.zeros(n_voxels, dtype=np.float64)
    surface_variation = np.zeros(n_voxels, dtype=np.float64)
    roughness = np.zeros(n_voxels, dtype=np.float64)
    neighbor_count = np.zeros(n_voxels, dtype=np.float64)
    volume_density = np.zeros(n_voxels, dtype=np.float64)

    k_fallback = min(max(_MIN_NEIGHBORS, 3), n_voxels)

    for idx in range(n_voxels):
        neighbor_ids = tree.query_ball_point(centroids[idx], radius)
        if len(neighbor_ids) < _MIN_NEIGHBORS and k_fallback > 0:
            _, knn_ids = tree.query(centroids[idx], k=k_fallback)
            neighbor_ids = np.atleast_1d(knn_ids).astype(np.int64).tolist()

        neighbor_ids = sorted(set(int(i) for i in neighbor_ids))
        local_xyz = centroids[neighbor_ids]
        local_counts = counts[neighbor_ids]

        neighbor_count[idx] = float(len(neighbor_ids))
        volume_density[idx] = float(np.sum(local_counts) / max(sphere_volume, _EPS))

        if len(local_xyz) < 3:
            continue

        centered = local_xyz - np.mean(local_xyz, axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False, bias=True)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        lam1, lam2, lam3 = eigvals[::-1]

        if lam1 > _EPS:
            planarity[idx] = float((lam2 - lam3) / lam1)
            sphericity[idx] = float(lam3 / lam1)
            anisotropy[idx] = float((lam1 - lam3) / lam1)

        lam_sum = lam1 + lam2 + lam3
        if lam_sum > _EPS:
            surface_variation[idx] = float(lam3 / lam_sum)

        normal = eigvecs[:, 0]
        distances = centered @ normal
        roughness[idx] = float(np.sqrt(np.mean(distances**2)))

    return {
        "planarity": planarity,
        "sphericity": sphericity,
        "anisotropy": anisotropy,
        "surface_variation": surface_variation,
        "roughness": roughness,
        "neighbor_count": neighbor_count,
        "volume_density": volume_density,
    }


def compute_exportable_geometry_features(
    xyz: np.ndarray,
    scale: float = 0.1,
    max_knn: int = 50000,
    voxel_resolution_xy: float = 0.05,
    voxel_resolution_z: float = 0.05,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute a consistent set of pointwise geometric features for export.

    The shape features are computed on voxel centroids and then reprojected to
    the original points through the voxel mapping.
    """
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")

    n_points = len(xyz)
    if n_points == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "verticality": empty,
            "linearity": empty,
            "planarity": empty,
            "sphericity": empty,
            "anisotropy": empty,
            "surface_variation": empty,
            "roughness": empty,
            "neighbor_count": empty,
            "volume_density": empty,
        }

    verticality, linearity = compute_wood_features(
        xyz,
        scale=scale,
        max_knn=max_knn,
        voxel_resolution_xy=voxel_resolution_xy,
        voxel_resolution_z=voxel_resolution_z,
        verbose=verbose,
    )

    centroids, point_to_voxel, n_voxels = voxelize_cloud(
        xyz,
        resolution_xy=voxel_resolution_xy,
        resolution_z=voxel_resolution_z,
    )
    voxel_counts = np.bincount(point_to_voxel, minlength=n_voxels)
    shape_features = _compute_local_shape_features(
        centroids,
        voxel_counts,
        max(voxel_resolution_xy, voxel_resolution_z),
    )

    exportable = {
        "verticality": verticality.astype(np.float64, copy=False),
        "linearity": linearity.astype(np.float64, copy=False),
    }
    for name, voxel_values in shape_features.items():
        exportable[name] = voxel_values[point_to_voxel]

    return exportable
