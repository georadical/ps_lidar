"""
Multiscale geometric feature extraction for post-normalization workflows.

This module implements Brick 7.1: feature extraction only (no classification).
It computes covariance-based geometric descriptors across multiple scales and
returns point-level aggregated features for downstream CC/rules pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .features import _voxelize_cloud


@dataclass
class MultiScaleGeometricFeatures:
    """Point-level multiscale geometric features."""

    scales: Tuple[float, ...]
    verticality: np.ndarray
    linearity: np.ndarray
    planarity: np.ndarray
    sphericity: np.ndarray
    roughness: np.ndarray
    mean_curvature: np.ndarray
    gaussian_curvature: np.ndarray
    neighbor_count: np.ndarray
    surface_density: np.ndarray
    volume_density: np.ndarray
    per_scale: Optional[Dict[str, np.ndarray]] = None


def _validate_scales(scales: Sequence[float]) -> Tuple[float, ...]:
    if len(scales) == 0:
        raise ValueError("scales must contain at least one positive value")

    out = []
    for scale in scales:
        scale_f = float(scale)
        if scale_f <= 0:
            raise ValueError(f"All scales must be > 0, got {scale}")
        out.append(scale_f)

    # Sorted unique tuple for deterministic output
    return tuple(sorted(set(out)))


def _compute_covariance_descriptors(
    neighborhood: np.ndarray,
    scale: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute geometric descriptors from one neighborhood.

    Returns:
        verticality, linearity, planarity, sphericity, roughness,
        mean_curvature, gaussian_curvature

    Notes:
    - Mean/Gaussian curvature here are covariance-based proxies derived from
      local eigenvalue spread. They are designed as robust features, not as
      exact differential geometry over an explicit surface.
    """
    eps = 1e-12

    if len(neighborhood) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    centered = neighborhood - np.mean(neighborhood, axis=0)
    cov = np.cov(centered.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    idx = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[idx], a_min=0.0, a_max=None)
    eigvecs = eigvecs[:, idx]

    lambda1, lambda2, lambda3 = eigvals
    normal = eigvecs[:, 2]

    if normal[2] < 0:
        normal = -normal

    verticality = float(1.0 - abs(normal[2]))

    if lambda1 <= eps:
        return verticality, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    linearity = float((lambda1 - lambda2) / (lambda1 + eps))
    planarity = float((lambda2 - lambda3) / (lambda1 + eps))
    sphericity = float(lambda3 / (lambda1 + eps))

    roughness = float(np.sqrt(lambda3))

    # Covariance-based curvature proxies.
    k1 = float(np.sqrt(lambda2 + eps) / (scale + eps))
    k2 = float(np.sqrt(lambda3 + eps) / (scale + eps))
    mean_curvature = 0.5 * (k1 + k2)
    gaussian_curvature = k1 * k2

    return (
        verticality,
        linearity,
        planarity,
        sphericity,
        roughness,
        mean_curvature,
        gaussian_curvature,
    )


def compute_multiscale_geometric_features(
    xyz: np.ndarray,
    scales: Sequence[float] = (0.10, 0.20, 0.40),
    voxel_size: float = 0.10,
    min_neighbors: int = 8,
    return_per_scale: bool = False,
    verbose: bool = False,
) -> MultiScaleGeometricFeatures:
    """
    Compute multiscale geometric features for each point.

    Workflow:
    1. Voxelize input cloud for speed.
    2. For each scale, compute neighborhood descriptors on voxel centroids.
    3. Aggregate (mean across scales) and re-project to original points.

    Parameters
    ----------
    xyz
        Input coordinates with shape (N, 3).
    scales
        Neighborhood radii in meters.
    voxel_size
        Voxelization size in meters (used before descriptor computation).
    min_neighbors
        Minimum neighbors target. If a radius neighborhood is too small, KNN
        fallback is used for stability.
    return_per_scale
        If True, includes per-scale matrices in `result.per_scale`.
    verbose
        Print progress.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got shape {xyz.shape}")

    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    scales_t = _validate_scales(scales)
    n_points = len(xyz)

    if n_points == 0:
        empty = np.array([], dtype=np.float32)
        return MultiScaleGeometricFeatures(
            scales=scales_t,
            verticality=empty,
            linearity=empty,
            planarity=empty,
            sphericity=empty,
            roughness=empty,
            mean_curvature=empty,
            gaussian_curvature=empty,
            neighbor_count=empty,
            surface_density=empty,
            volume_density=empty,
            per_scale={} if return_per_scale else None,
        )

    voxel_xyz, inverse_indices, n_voxels = _voxelize_cloud(xyz, voxel_size=voxel_size)
    n_scales = len(scales_t)

    if verbose:
        print(
            "Computing multiscale geometric features: "
            f"{n_points:,} points -> {n_voxels:,} voxels, {n_scales} scales"
        )

    tree = cKDTree(voxel_xyz)

    vert_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    lin_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    pla_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    sph_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    rou_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    mcu_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    gcu_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    nei_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    sde_v = np.zeros((n_voxels, n_scales), dtype=np.float32)
    vde_v = np.zeros((n_voxels, n_scales), dtype=np.float32)

    k_fallback = int(max(3, min(min_neighbors, n_voxels)))

    for scale_idx, scale in enumerate(scales_t):
        if verbose:
            print(f"  Scale {scale_idx + 1}/{n_scales}: r={scale:.3f}m")

        neighbors_per_voxel = tree.query_ball_point(voxel_xyz, r=scale, workers=-1)

        for i in range(n_voxels):
            neighbors = neighbors_per_voxel[i]
            if len(neighbors) < 3 and k_fallback > 1:
                _, knn_indices = tree.query(voxel_xyz[i], k=k_fallback, workers=-1)
                neighbors = np.atleast_1d(knn_indices).tolist()

            n_neighbors = len(neighbors)
            nei_v[i, scale_idx] = float(n_neighbors)

            neighborhood = voxel_xyz[neighbors]
            (
                vert,
                lin,
                pla,
                sph,
                rou,
                mcu,
                gcu,
            ) = _compute_covariance_descriptors(neighborhood, scale=scale)

            vert_v[i, scale_idx] = vert
            lin_v[i, scale_idx] = lin
            pla_v[i, scale_idx] = pla
            sph_v[i, scale_idx] = sph
            rou_v[i, scale_idx] = rou
            mcu_v[i, scale_idx] = mcu
            gcu_v[i, scale_idx] = gcu

            area = np.pi * (scale**2)
            volume = (4.0 / 3.0) * np.pi * (scale**3)
            sde_v[i, scale_idx] = float(n_neighbors / area)
            vde_v[i, scale_idx] = float(n_neighbors / volume)

        if verbose:
            print("    done")

    def _mean_to_points(values_voxel: np.ndarray) -> np.ndarray:
        return np.mean(values_voxel, axis=1, dtype=np.float32)[inverse_indices]

    per_scale = None
    if return_per_scale:
        per_scale = {
            "verticality": vert_v[inverse_indices],
            "linearity": lin_v[inverse_indices],
            "planarity": pla_v[inverse_indices],
            "sphericity": sph_v[inverse_indices],
            "roughness": rou_v[inverse_indices],
            "mean_curvature": mcu_v[inverse_indices],
            "gaussian_curvature": gcu_v[inverse_indices],
            "neighbor_count": nei_v[inverse_indices],
            "surface_density": sde_v[inverse_indices],
            "volume_density": vde_v[inverse_indices],
        }

    return MultiScaleGeometricFeatures(
        scales=scales_t,
        verticality=_mean_to_points(vert_v),
        linearity=_mean_to_points(lin_v),
        planarity=_mean_to_points(pla_v),
        sphericity=_mean_to_points(sph_v),
        roughness=_mean_to_points(rou_v),
        mean_curvature=_mean_to_points(mcu_v),
        gaussian_curvature=_mean_to_points(gcu_v),
        neighbor_count=_mean_to_points(nei_v),
        surface_density=_mean_to_points(sde_v),
        volume_density=_mean_to_points(vde_v),
        per_scale=per_scale,
    )
