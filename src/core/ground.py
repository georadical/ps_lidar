"""
Ground Filtering Module

Separates ground points from vegetation/objects using the 
Cloth Simulation Filter (CSF) algorithm.
"""

import CSF
import numpy as np
from typing import Tuple


def classify_ground(
    xyz: np.ndarray, 
    cloth_resolution: float = 1.0, 
    rigidness: int = 1, 
    time_step: float = 0.6, 
    class_threshold: float = 0.5, 
    inter_iterations: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classifies points as ground or off-ground using the CSF algorithm.
    
    Args:
        xyz: (N, 3) numpy array of point coordinates.
        cloth_resolution: Distance between cloth nodes (m). 
                          Lower values capture more terrain detail but are slower.
        rigidness: Constraint on cloth movement (1: flat, 2: relief, 3: steep terrain).
        time_step: Stability parameter for the simulation.
        class_threshold: Distance threshold to classify points as ground (m).
        inter_iterations: Maximum number of internal iterations for the simulation.
        
    Returns:
        A tuple of (ground_indices, off_ground_indices) as numpy arrays.
    """
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Expected (N, 3) array, got shape {xyz.shape}")

    csf = CSF.CSF()
    
    # Setting parameters
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold
    csf.params.inter_iterations = inter_iterations
    
    # Pass points to CSF (needs double precision)
    csf.setPointCloud(xyz.astype(np.float64))
    
    # Prepare output containers
    ground_indices = CSF.VecInt()
    off_ground_indices = CSF.VecInt()
    
    # Execute
    csf.do_filtering(ground_indices, off_ground_indices)
    
    return np.array(ground_indices), np.array(off_ground_indices)


def get_ground_mask(xyz: np.ndarray, **kwargs) -> np.ndarray:
    """
    Returns a boolean mask where True indicates a ground point.
    """
    ground_idx, _ = classify_ground(xyz, **kwargs)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[ground_idx] = True
    return mask
