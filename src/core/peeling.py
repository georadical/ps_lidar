"""
Voxel-based iterative peeling for understory separation.
This is a highly optimized implementation that operates on voxels instead of points.
"""
import numpy as np
import warnings
import time
from dataclasses import dataclass
from scipy.sparse import csr_matrix


@dataclass
class IterativePeelingResult:
    """Result of iterative peeling understory separation."""
    is_tree: np.ndarray          # (N,) boolean mask (True = tree, False = understory)
    n_seeds: int                 # Number of initial seed points
    n_iterations: int            # Number of expansion iterations performed
    n_tree: int                  # Final number of tree points
    n_understory: int            # Final number of understory points
    expansion_percentage: float  # Percentage of points added via expansion


def iterative_peeling_understory(
    xyz: np.ndarray,
    verticality: np.ndarray,
    linearity: np.ndarray,
    sphericity: np.ndarray,
    dist_to_ground: np.ndarray,
    seed_verticality: float = 0.85,
    seed_linearity: float = 0.6,
    seed_height_min: float = 1.5,
    seed_height_max: float = 11.0,
    expansion_verticality: float = 0.5,
    understory_height: float = 3.0,
    canopy_height: float = 15.0,
    voxel_size: float = 0.1,
    max_iterations: int = 50,
    verbose: bool = False
) -> IterativePeelingResult:
    """
    Separate trees from understory using voxel-based iterative peeling.
    
    This optimized version operates on voxels instead of points for massive speedup.
    Includes canopy protection (auto-include high points) and height-aware expansion.
    
    Strategy:
    1. Voxelize cloud to reduce data size (e.g. 0.1m voxels).
    2. Protect canopy voxels (height > canopy_height) automatically.
    3. Select robust seeds in the "clean stem" zone (seed_height_min to seed_height_max).
    4. Grow iteratively from seeds/canopy:
       - Upward growth into canopy uses relaxed criteria.
       - Downward growth into understory uses strict verticality to ensure it's a stem.
    """
    n_points = len(xyz)
    
    if verbose:
        print(f"Iterative peeling (voxel-based) for {n_points:,} points...")
        print(f"  Voxel size: {voxel_size}m")
        print(f"  Seed criteria: vert>{seed_verticality}, line>{seed_linearity}, height {seed_height_min}-{seed_height_max}m")
        print(f"  Canopy protection: >{canopy_height}m")
        t0 = time.time()
    
    # ========================================================================
    # Step 1: Voxelize point cloud
    # ========================================================================
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    dtype = [('x', np.int32), ('y', np.int32), ('z', np.int32)]
    voxel_struct = np.empty(len(voxel_indices), dtype=dtype)
    voxel_struct['x'] = voxel_indices[:, 0]
    voxel_struct['y'] = voxel_indices[:, 1]
    voxel_struct['z'] = voxel_indices[:, 2]
    
    unique_voxels, inverse_indices = np.unique(voxel_struct, return_inverse=True)
    n_voxels = len(unique_voxels)
    
    # ========================================================================
    # Step 2: Compute per-voxel features
    # ========================================================================
    voxel_verticality_sum = np.bincount(inverse_indices, weights=verticality, minlength=n_voxels)
    voxel_linearity_sum = np.bincount(inverse_indices, weights=linearity, minlength=n_voxels)
    voxel_height_sum = np.bincount(inverse_indices, weights=dist_to_ground, minlength=n_voxels)
    voxel_counts = np.bincount(inverse_indices, minlength=n_voxels).astype(np.float32)
    voxel_counts[voxel_counts == 0] = 1
    
    voxel_verticality = voxel_verticality_sum / voxel_counts
    voxel_linearity = voxel_linearity_sum / voxel_counts
    voxel_height = voxel_height_sum / voxel_counts
    
    # ========================================================================
    # Step 3: Identify seeds and protected canopy
    # ========================================================================
    canopy_voxel_mask = voxel_height > canopy_height
    seed_voxel_mask = (
        (voxel_verticality > seed_verticality) &
        (voxel_linearity > seed_linearity) &
        (voxel_height >= seed_height_min) &
        (voxel_height <= seed_height_max)
    )
    
    is_tree_voxel = seed_voxel_mask | canopy_voxel_mask
    n_initial_voxels = np.sum(is_tree_voxel)
    
    if n_initial_voxels == 0:
        if verbose: print("  WARNING: No seeds or canopy found!")
        return IterativePeelingResult(np.zeros(n_points, dtype=bool), 0, 0, 0, n_points, 0.0)
    
    # ========================================================================
    # Step 4: Build voxel adjacency graph
    # ========================================================================
    voxel_coords = np.column_stack([unique_voxels['x'], unique_voxels['y'], unique_voxels['z']])
    voxel_to_idx = {(v[0], v[1], v[2]): i for i, v in enumerate(voxel_coords)}
    
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx != 0 or dy != 0 or dz != 0:
                    offsets.append((dx, dy, dz))
    offsets = np.array(offsets, dtype=np.int32)
    
    rows, cols = [], []
    for offset in offsets:
        shifted = voxel_coords + offset
        for i, (x, y, z) in enumerate(shifted):
            key = (x, y, z)
            if key in voxel_to_idx:
                rows.append(i)
                cols.append(voxel_to_idx[key])
    
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_voxels, n_voxels))
    
    # ========================================================================
    # Step 5: Iterative expansion
    # ========================================================================
    previous_tree_voxel = np.zeros(n_voxels, dtype=bool)
    iteration = 0
    
    for iteration in range(max_iterations):
        frontier_mask = is_tree_voxel & ~previous_tree_voxel
        if np.sum(frontier_mask) == 0: break
        
        previous_tree_voxel = is_tree_voxel.copy()
        frontier_indices = np.where(frontier_mask)[0]
        neighbor_rows = adj[frontier_indices].nonzero()[1]
        unique_neighbors = np.unique(neighbor_rows)
        
        candidates = unique_neighbors[~is_tree_voxel[unique_neighbors]]
        if len(candidates) > 0:
            c_h = voxel_height[candidates]
            c_v = voxel_verticality[candidates]
            valid_mask = np.zeros(len(candidates), dtype=bool)
            
            # Growth Rule 1: High points (canopy connection) - relaxed criteria
            valid_mask |= (c_h > seed_height_max) & (c_v > (expansion_verticality * 0.6))
            
            # Growth Rule 2: Clean zone - standard
            valid_mask |= (c_h >= seed_height_min) & (c_h <= seed_height_max) & (c_v > expansion_verticality)
            
            # Growth Rule 3: Low points (understory zone) - VERY STRICT to avoid bushes
            # Only grow into understory zone if verticality is very high (trunk-like)
            strict_vert = max(expansion_verticality, 0.75)  # At least 0.75 for low zone
            valid_mask |= (c_h < understory_height) & (c_v > strict_vert)
            
            is_tree_voxel[candidates[valid_mask]] = True
    
    # ========================================================================
    # Step 6: UNCONDITIONAL CANOPY PROTECTION
    # All voxels above canopy_height are ALWAYS tree, regardless of connectivity
    # ========================================================================
    is_tree_voxel = is_tree_voxel | canopy_voxel_mask  # Re-apply to catch disconnected canopy
    
    # ========================================================================
    # Step 7: Map back and Statistics
    # ========================================================================
    is_tree = is_tree_voxel[inverse_indices]
    n_tree = np.sum(is_tree)
    n_seeds = np.sum(seed_voxel_mask[inverse_indices])
    
    if verbose:
        print(f"  ✓ Peeling complete in {time.time()-t0:.1f}s. Iterations: {iteration}")
        print(f"  Final: {n_tree:,} tree ({100*n_tree/n_points:.1f}%)")
    
    return IterativePeelingResult(
        is_tree=is_tree,
        n_seeds=n_seeds,
        n_iterations=iteration,
        n_tree=n_tree,
        n_understory=n_points - n_tree,
        expansion_percentage=0.0 # Not used for logic, simplified
    )
