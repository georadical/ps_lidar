"""
Sampling and Spatial Clipping Module

Provides functions for extracting spatial subsets of point clouds,
simulating field sampling protocols (e.g., circular plots).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import warnings


@dataclass
class CircularPlotResult:
    """
    Result of a circular plot clipping operation.
    
    Attributes:
        center_x: X coordinate of the plot center.
        center_y: Y coordinate of the plot center.
        radius: Radius used for clipping in meters.
        indices: Array of point indices within the circular plot.
        n_points: Number of points in the clipped plot.
        area_m2: Theoretical area of the plot in square meters.
        area_ha: Theoretical area of the plot in hectares.
    """
    center_x: float
    center_y: float
    radius: float
    indices: np.ndarray
    n_points: int
    area_m2: float
    area_ha: float


@dataclass
class ReflectiveCluster:
    """
    A single detected high-intensity cluster.
    """
    x: float
    y: float
    z: float
    n_points: int
    mean_intensity: float


@dataclass
class ReflectiveMat:
    """
    A detected reflective mat (may consist of 1 or 2 clusters).
    
    For checkerboard mats: center is midpoint between 2 diagonal clusters.
    For circular mats: center is the single cluster centroid.
    """
    center_x: float
    center_y: float
    center_z: float
    n_clusters: int
    clusters: List[ReflectiveCluster]
    total_points: int


def get_centroid(xyz: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the 2D centroid (X, Y) of a point cloud.
    
    Args:
        xyz: (N, 3) numpy array with coordinates.
        
    Returns:
        Tuple of (center_x, center_y).
    """
    return float(np.mean(xyz[:, 0])), float(np.mean(xyz[:, 1]))


def _detect_intensity_clusters(
    xyz: np.ndarray,
    intensity: np.ndarray,
    intensity_percentile: float = 99.5,
    cluster_radius: float = 0.2,
    min_points: int = 10
) -> List[ReflectiveCluster]:
    """
    Detect individual high-intensity clusters.
    
    Args:
        xyz: (N, 3) numpy array.
        intensity: (N,) numpy array.
        intensity_percentile: Threshold percentile.
        cluster_radius: Radius to group nearby high-intensity points.
        min_points: Minimum points per cluster.
        
    Returns:
        List of ReflectiveCluster objects.
    """
    threshold = np.percentile(intensity, intensity_percentile)
    high_mask = intensity >= threshold
    high_xyz = xyz[high_mask]
    high_int = intensity[high_mask]
    
    if len(high_xyz) < min_points:
        return []
    
    clusters = []
    remaining = np.ones(len(high_xyz), dtype=bool)
    
    while np.sum(remaining) >= min_points:
        remaining_idx = np.where(remaining)[0]
        seed_idx = remaining_idx[0]
        seed_xy = high_xyz[seed_idx, :2]
        
        # Find neighbors
        dists = np.sqrt(np.sum((high_xyz[remaining, :2] - seed_xy)**2, axis=1))
        local_mask = dists <= cluster_radius
        
        if np.sum(local_mask) >= min_points:
            cluster_idx = remaining_idx[local_mask]
            cluster_xyz = high_xyz[cluster_idx]
            cluster_int = high_int[cluster_idx]
            
            clusters.append(ReflectiveCluster(
                x=float(np.mean(cluster_xyz[:, 0])),
                y=float(np.mean(cluster_xyz[:, 1])),
                z=float(np.mean(cluster_xyz[:, 2])),
                n_points=len(cluster_idx),
                mean_intensity=float(np.mean(cluster_int))
            ))
            remaining[cluster_idx] = False
        else:
            remaining[seed_idx] = False
    
    return clusters


def detect_reflective_mats(
    xyz: np.ndarray,
    intensity: np.ndarray,
    mat_size: float = 0.6,
    intensity_percentile: float = 99.5,
    cluster_radius: float = 0.2,
    min_points: int = 10
) -> List[ReflectiveMat]:
    """
    Detect reflective mats (checkerboard or circular patterns).
    
    For checkerboard mats: groups 2 diagonal clusters within mat_size.
    For circular mats: each cluster is its own mat.
    
    Args:
        xyz: (N, 3) numpy array with coordinates.
        intensity: (N,) numpy array with intensity values.
        mat_size: Maximum distance between clusters of the same mat (default: 0.6m).
        intensity_percentile: Percentile threshold for high intensity.
        cluster_radius: Radius to group nearby points into a single cluster.
        min_points: Minimum points required per cluster.
        
    Returns:
        List of ReflectiveMat objects, sorted by distance to cloud centroid.
    """
    # Step 1: Detect individual clusters
    clusters = _detect_intensity_clusters(
        xyz, intensity, intensity_percentile, cluster_radius, min_points
    )
    
    if not clusters:
        return []
    
    # Step 2: Group clusters into mats based on proximity
    used = [False] * len(clusters)
    mats = []
    
    for i, c1 in enumerate(clusters):
        if used[i]:
            continue
        
        # Find all clusters within mat_size distance
        group = [c1]
        used[i] = True
        
        for j, c2 in enumerate(clusters):
            if used[j]:
                continue
            dist = np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
            if dist <= mat_size:
                group.append(c2)
                used[j] = True
        
        # Calculate mat center (midpoint of all cluster centroids)
        cx = np.mean([c.x for c in group])
        cy = np.mean([c.y for c in group])
        cz = np.mean([c.z for c in group])
        total_pts = sum(c.n_points for c in group)
        
        mats.append(ReflectiveMat(
            center_x=float(cx),
            center_y=float(cy),
            center_z=float(cz),
            n_clusters=len(group),
            clusters=group,
            total_points=total_pts
        ))
    
    # Step 3: Sort mats by distance to cloud centroid
    cloud_cx, cloud_cy = get_centroid(xyz)
    mats.sort(key=lambda m: (m.center_x - cloud_cx)**2 + (m.center_y - cloud_cy)**2)
    
    return mats


def find_plot_center(
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    mat_size: float = 0.6,
    **kwargs
) -> Tuple[float, float, str]:
    """
    Automatically find the plot center using the best available method.
    
    Priority:
    1. Center of reflective mat nearest to geometric centroid (if intensity provided)
    2. Geometric centroid of the point cloud (fallback)
    
    Args:
        xyz: (N, 3) numpy array with coordinates.
        intensity: Optional (N,) numpy array with intensity values.
        mat_size: Size of reflective mats in meters (default: 0.6).
        **kwargs: Additional arguments for mat detection.
        
    Returns:
        Tuple of (center_x, center_y, method_used).
    """
    centroid_x, centroid_y = get_centroid(xyz)
    
    if intensity is not None:
        mats = detect_reflective_mats(xyz, intensity, mat_size=mat_size, **kwargs)
        
        if mats:
            # First mat is closest to centroid (already sorted)
            best_mat = mats[0]
            method = f"reflective_mat ({best_mat.n_clusters} clusters)"
            return best_mat.center_x, best_mat.center_y, method
        else:
            warnings.warn(
                "No se detectaron mats reflectivos. Usando centroide geométrico.",
                UserWarning
            )
    
    return centroid_x, centroid_y, "geometric_centroid"


def validate_gps_center(
    center_x: float,
    center_y: float,
    xyz: np.ndarray,
    has_crs: bool = False,
    tolerance: float = 0.0
) -> bool:
    """
    Validate that GPS coordinates fall within the point cloud bounds.
    
    Args:
        center_x: X coordinate.
        center_y: Y coordinate.
        xyz: (N, 3) numpy array.
        has_crs: Whether the point cloud has a CRS.
        tolerance: Buffer around bounds in meters.
        
    Returns:
        True if valid.
        
    Raises:
        ValueError: If coordinates outside bounds.
    """
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    
    x_ok = (x_min - tolerance) <= center_x <= (x_max + tolerance)
    y_ok = (y_min - tolerance) <= center_y <= (y_max + tolerance)
    
    if not has_crs:
        warnings.warn(
            "La nube no tiene CRS. Verifica que las coordenadas sean correctas.",
            UserWarning
        )
    
    if not (x_ok and y_ok):
        raise ValueError(
            f"Coordenada ({center_x:.2f}, {center_y:.2f}) fuera de bounds.\n"
            f"X: [{x_min:.2f}, {x_max:.2f}], Y: [{y_min:.2f}, {y_max:.2f}]"
        )
    
    return True


def clip_circular_plot(
    xyz: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float
) -> CircularPlotResult:
    """
    Extract indices of points within a circular radius from a 2D center.
    
    Args:
        xyz: (N, 3) numpy array.
        center_x: X coordinate of plot center.
        center_y: Y coordinate of plot center.
        radius: Radius in meters.
        
    Returns:
        CircularPlotResult with indices and metadata.
    """
    dx = xyz[:, 0] - center_x
    dy = xyz[:, 1] - center_y
    dist_sq = dx**2 + dy**2
    radius_sq = radius**2
    
    indices = np.where(dist_sq <= radius_sq)[0]
    area_m2 = np.pi * radius_sq
    
    return CircularPlotResult(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        indices=indices,
        n_points=len(indices),
        area_m2=area_m2,
        area_ha=area_m2 / 10000.0
    )


# Legacy alias for backwards compatibility
def detect_reflective_targets(*args, **kwargs):
    """Deprecated: Use detect_reflective_mats() instead."""
    warnings.warn(
        "detect_reflective_targets() is deprecated. Use detect_reflective_mats().",
        DeprecationWarning
    )
    return detect_reflective_mats(*args, **kwargs)
