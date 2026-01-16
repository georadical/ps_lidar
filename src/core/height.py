"""
Height Normalization Module

Calculates normalized heights (Z relative to ground) for vegetation points
using fast grid-based DTM interpolation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Literal


@dataclass
class HeightNormalizationResult:
    """
    Result of height normalization operation.
    
    Attributes:
        xyz_normalized: (N, 3) array with normalized Z values.
        z_original: Original Z values before normalization.
        z_ground: Interpolated ground heights at each point.
        z_normalized: Normalized heights (z_original - z_ground).
        n_points: Number of points normalized.
        grid_resolution: Resolution of the DTM grid used.
    """
    xyz_normalized: np.ndarray
    z_original: np.ndarray
    z_ground: np.ndarray
    z_normalized: np.ndarray
    n_points: int
    grid_resolution: float
    
    @property
    def height_stats(self) -> dict:
        """Return basic statistics of normalized heights."""
        return {
            "min": float(np.nanmin(self.z_normalized)),
            "max": float(np.nanmax(self.z_normalized)),
            "mean": float(np.nanmean(self.z_normalized)),
            "std": float(np.nanstd(self.z_normalized)),
        }


def _create_dtm_grid(
    ground_xyz: np.ndarray,
    resolution: float,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Create a regular grid DTM from ground points.
    
    Returns:
        Tuple of (dtm_grid, x_min, y_min, x_max, y_max)
    """
    if bounds is None:
        x_min, y_min = ground_xyz[:, 0].min(), ground_xyz[:, 1].min()
        x_max, y_max = ground_xyz[:, 0].max(), ground_xyz[:, 1].max()
    else:
        x_min, y_min, x_max, y_max = bounds
    
    # Calculate grid dimensions
    n_cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    n_rows = int(np.ceil((y_max - y_min) / resolution)) + 1
    
    # Initialize grid with NaN
    dtm_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
    dtm_count = np.zeros((n_rows, n_cols), dtype=np.int32)
    
    # Assign ground points to grid cells
    col_idx = ((ground_xyz[:, 0] - x_min) / resolution).astype(np.int32)
    row_idx = ((ground_xyz[:, 1] - y_min) / resolution).astype(np.int32)
    
    # Clip to valid range
    col_idx = np.clip(col_idx, 0, n_cols - 1)
    row_idx = np.clip(row_idx, 0, n_rows - 1)
    
    # Accumulate Z values
    np.add.at(dtm_sum, (row_idx, col_idx), ground_xyz[:, 2])
    np.add.at(dtm_count, (row_idx, col_idx), 1)
    
    # Calculate mean Z per cell
    with np.errstate(divide='ignore', invalid='ignore'):
        dtm_grid = dtm_sum / dtm_count
    
    # Fill empty cells with nearest neighbor
    empty_mask = dtm_count == 0
    if np.any(empty_mask):
        from scipy.ndimage import distance_transform_edt
        # Find distance to nearest filled cell and its index
        _, indices = distance_transform_edt(empty_mask, return_indices=True)
        dtm_grid[empty_mask] = dtm_grid[indices[0, empty_mask], indices[1, empty_mask]]
    
    return dtm_grid, x_min, y_min, x_max, y_max


def normalize_heights(
    vegetation_xyz: np.ndarray,
    ground_xyz: np.ndarray,
    resolution: float = 0.5,
) -> HeightNormalizationResult:
    """
    Normalize vegetation heights relative to the ground surface.
    
    Uses a fast grid-based DTM approach:
    1. Creates a regular grid from ground points
    2. Averages ground Z in each cell
    3. Looks up grid cell for each vegetation point
    
    Args:
        vegetation_xyz: (N, 3) array of vegetation point coordinates.
        ground_xyz: (M, 3) array of ground point coordinates.
        resolution: Grid cell size in meters (default: 0.5m).
    
    Returns:
        HeightNormalizationResult with normalized coordinates and metadata.
    
    Raises:
        ValueError: If input arrays have wrong shape or too few ground points.
    """
    if vegetation_xyz.ndim != 2 or vegetation_xyz.shape[1] < 3:
        raise ValueError(f"vegetation_xyz must be (N, 3), got {vegetation_xyz.shape}")
    if ground_xyz.ndim != 2 or ground_xyz.shape[1] < 3:
        raise ValueError(f"ground_xyz must be (M, 3), got {ground_xyz.shape}")
    if len(ground_xyz) < 3:
        raise ValueError("Need at least 3 ground points for interpolation")
    
    # Get bounds from vegetation (may extend beyond ground)
    all_xyz = np.vstack([vegetation_xyz, ground_xyz])
    x_min, y_min = all_xyz[:, 0].min(), all_xyz[:, 1].min()
    x_max, y_max = all_xyz[:, 0].max(), all_xyz[:, 1].max()
    bounds = (x_min, y_min, x_max, y_max)
    
    # Create DTM grid
    dtm_grid, x_min, y_min, x_max, y_max = _create_dtm_grid(
        ground_xyz, resolution, bounds
    )
    
    # Look up grid cell for each vegetation point
    veg_col = ((vegetation_xyz[:, 0] - x_min) / resolution).astype(np.int32)
    veg_row = ((vegetation_xyz[:, 1] - y_min) / resolution).astype(np.int32)
    
    # Clip to valid range
    veg_col = np.clip(veg_col, 0, dtm_grid.shape[1] - 1)
    veg_row = np.clip(veg_row, 0, dtm_grid.shape[0] - 1)
    
    # Get ground height at each vegetation point
    z_ground = dtm_grid[veg_row, veg_col]
    
    # Calculate normalized heights
    veg_z = vegetation_xyz[:, 2]
    z_normalized = veg_z - z_ground
    
    # Create normalized XYZ array
    xyz_normalized = vegetation_xyz.copy()
    xyz_normalized[:, 2] = z_normalized
    
    return HeightNormalizationResult(
        xyz_normalized=xyz_normalized,
        z_original=veg_z.copy(),
        z_ground=z_ground,
        z_normalized=z_normalized,
        n_points=len(vegetation_xyz),
        grid_resolution=resolution,
    )


def get_normalized_vegetation(
    xyz: np.ndarray,
    ground_indices: np.ndarray,
    off_ground_indices: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, HeightNormalizationResult]:
    """
    Convenience function to normalize vegetation heights from ground filter result.
    
    Args:
        xyz: Full (N, 3) point cloud array.
        ground_indices: Indices of ground points (from classify_ground).
        off_ground_indices: Indices of vegetation points.
        **kwargs: Additional arguments passed to normalize_heights.
    
    Returns:
        Tuple of (normalized_vegetation_xyz, HeightNormalizationResult).
    """
    ground_xyz = xyz[ground_indices]
    vegetation_xyz = xyz[off_ground_indices]
    
    result = normalize_heights(vegetation_xyz, ground_xyz, **kwargs)
    
    return result.xyz_normalized, result
