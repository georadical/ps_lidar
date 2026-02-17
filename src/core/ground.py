"""
Ground Filtering Module

Separates ground points from vegetation/objects using the 
Cloth Simulation Filter (CSF) algorithm.

Reference:
    Zhang, W., et al. (2016). An Easy-to-Use Airborne LiDAR Data Filtering Method 
    Based on Cloth Simulation. Remote Sensing.
"""

import CSF
import numpy as np
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional


@dataclass
class GroundFilterResult:
    """
    Results from ground filtering operation.
    
    Attributes:
        ground_indices: Indices of points classified as ground
        off_ground_indices: Indices of points classified as vegetation/objects
        cloth_nodes: (M, 3) array of cloth surface positions (the DTM approximation)
        n_ground: Number of ground points
        n_off_ground: Number of off-ground points
    """
    ground_indices: np.ndarray
    off_ground_indices: np.ndarray
    cloth_nodes: Optional[np.ndarray]
    
    @property
    def n_ground(self) -> int:
        return len(self.ground_indices)
    
    @property
    def n_off_ground(self) -> int:
        return len(self.off_ground_indices)
    
    @property
    def ground_ratio(self) -> float:
        total = self.n_ground + self.n_off_ground
        return self.n_ground / total if total > 0 else 0.0


def classify_ground(
    xyz: np.ndarray, 
    cloth_resolution: float = 1.0, 
    rigidness: int = 1, 
    time_step: float = 0.65, 
    class_threshold: float = 0.5, 
    iterations: int = 500,
    slope_smooth: bool = True,
    export_cloth: bool = False,
    cloth_nodes_path: Optional[str] = None,
) -> GroundFilterResult:
    """
    Classifies points as ground or off-ground using the CSF algorithm.
    
    The Cloth Simulation Filter works by simulating a piece of cloth falling
    from above onto the inverted point cloud. Points close to the final cloth
    position are classified as ground.
    
    Args:
        xyz: (N, 3) numpy array of point coordinates.
        cloth_resolution: Distance between cloth nodes (m). 
            Lower values capture more terrain detail but are slower.
            Typical: 0.5-2.0m for forest plots.
        rigidness: Constraint on cloth movement:
            1 = Flat terrain (most flexible cloth)
            2 = Relief terrain (moderate)
            3 = Steep terrain (rigid cloth)
        time_step: Simulation time step. Higher = faster but less stable.
            Typical: 0.5-0.65
        class_threshold: Distance threshold to classify points as ground (m).
            Points within this distance of the cloth are ground.
        iterations: Maximum simulation iterations.
        slope_smooth: If True, applies post-processing to handle steep slopes better.
            Recommended for forest terrain.
        export_cloth: If True, requests CSF cloth export.
        cloth_nodes_path: Optional path to read exported cloth nodes from.
            If omitted, cloth_nodes are not loaded to avoid implicit dependence
            on process working directory.
        
    Returns:
        GroundFilterResult containing indices and optional cloth nodes.
        
    Raises:
        ValueError: If input array has wrong shape.
    """
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Expected (N, 3) array, got shape {xyz.shape}")

    csf = CSF.CSF()
    
    # Configure parameters
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold
    csf.params.interations = iterations  # Note: CSF library has typo 'interations'
    csf.params.bSloopSmooth = slope_smooth
    
    # Pass points to CSF (requires double precision, contiguous array)
    points = np.ascontiguousarray(xyz[:, :3].astype(np.float64))
    csf.setPointCloud(points)
    
    # Prepare output containers
    ground_indices = CSF.VecInt()
    off_ground_indices = CSF.VecInt()
    
    # Execute filtering
    csf.do_filtering(ground_indices, off_ground_indices, exportCloth=export_cloth)
    
    # Get cloth nodes if requested and explicit path is provided
    cloth_nodes = None
    if export_cloth:
        if cloth_nodes_path is None:
            warnings.warn(
                "export_cloth=True but cloth_nodes_path is not set; "
                "cloth_nodes will not be loaded.",
                UserWarning,
                stacklevel=2,
            )
        else:
            cloth_file = Path(cloth_nodes_path)
            try:
                cloth_data = np.loadtxt(cloth_file)
                if cloth_data.size > 0:
                    cloth_nodes = cloth_data.reshape(-1, 3)
            except (FileNotFoundError, ValueError):
                warnings.warn(
                    f"Could not load cloth nodes from '{cloth_file}'.",
                    UserWarning,
                    stacklevel=2,
                )
    
    return GroundFilterResult(
        ground_indices=np.array(ground_indices),
        off_ground_indices=np.array(off_ground_indices),
        cloth_nodes=cloth_nodes,
    )


def get_ground_mask(xyz: np.ndarray, **kwargs) -> np.ndarray:
    """
    Returns a boolean mask where True indicates a ground point.
    
    This is a convenience wrapper around classify_ground for when
    you only need a mask rather than separate index arrays.
    
    Args:
        xyz: (N, 3) numpy array of point coordinates.
        **kwargs: Additional arguments passed to classify_ground.
        
    Returns:
        Boolean array of shape (N,) where True = ground point.
    """
    result = classify_ground(xyz, **kwargs)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[result.ground_indices] = True
    return mask


def get_ground_points(xyz: np.ndarray, **kwargs) -> np.ndarray:
    """
    Returns only the ground points from the input cloud.
    
    Args:
        xyz: (N, 3) numpy array of point coordinates.
        **kwargs: Additional arguments passed to classify_ground.
        
    Returns:
        (M, 3) array containing only ground points.
    """
    result = classify_ground(xyz, **kwargs)
    return xyz[result.ground_indices]
