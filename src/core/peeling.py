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


@dataclass
class SliceFilterResult:
    """Result of slice-based understory filtering."""
    is_tree: np.ndarray          # (N,) boolean mask (True = tree, False = understory)
    n_tree: int                  # Final number of tree points
    n_understory: int            # Final number of understory points
    n_protected: int             # Points in upper slice (untouched)
    n_filtered_lower: int        # Points removed from lower slice


def slice_filter_understory(
    xyz: np.ndarray,
    verticality: np.ndarray,
    linearity: np.ndarray,
    dist_to_ground: np.ndarray,
    slice_height: float = 3.0,
    seed_verticality: float = 0.7,
    seed_linearity: float = 0.4,
    expansion_verticality: float = 0.5,
    voxel_size: float = 0.1,
    max_iterations: int = 30,
    verbose: bool = False
) -> SliceFilterResult:
    """
    Slice-based understory filtering.
    
    Strategy:
    1. Split cloud into lower slice (<slice_height) and upper slice (>=slice_height)
    2. Upper slice is 100% PROTECTED - never touched
    3. Apply peeling ONLY to lower slice to identify tree trunks
    4. Merge results: upper_slice + lower_slice_trees
    
    This guarantees canopy protection since it never enters the filter.
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    verticality : np.ndarray
        Verticality feature (0-1) for each point.
    linearity : np.ndarray
        Linearity feature (0-1) for each point.
    dist_to_ground : np.ndarray
        Distance to ground for each point (meters).
    slice_height : float
        Height threshold for slicing. Points below this are filtered (default: 3.0m).
    seed_verticality : float
        Minimum verticality for seed selection in lower slice (default: 0.7).
    seed_linearity : float
        Minimum linearity for seed selection (default: 0.4).
    expansion_verticality : float
        Minimum verticality for expansion (default: 0.5).
    voxel_size : float
        Size of voxels for discretization (default: 0.1m).
    max_iterations : int
        Maximum expansion iterations (default: 30).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    SliceFilterResult
        Result object containing is_tree mask and statistics.
    """
    n_points = len(xyz)
    
    if verbose:
        print(f"Slice-based understory filtering for {n_points:,} points...")
        print(f"  Slice height: {slice_height}m")
        t0 = time.time()
    
    # ========================================================================
    # Step 1: Split into lower and upper slices
    # ========================================================================
    lower_mask = dist_to_ground < slice_height
    upper_mask = ~lower_mask
    
    n_lower = np.sum(lower_mask)
    n_upper = np.sum(upper_mask)
    
    if verbose:
        print(f"  Lower slice (<{slice_height}m): {n_lower:,} points")
        print(f"  Upper slice (>={slice_height}m): {n_upper:,} points (PROTECTED)")
    
    # Initialize result: upper slice is ALL tree (protected)
    is_tree = np.zeros(n_points, dtype=bool)
    is_tree[upper_mask] = True
    
    if n_lower == 0:
        if verbose:
            print("  No points in lower slice. Done.")
        return SliceFilterResult(
            is_tree=is_tree,
            n_tree=n_upper,
            n_understory=0,
            n_protected=n_upper,
            n_filtered_lower=0
        )
    
    # ========================================================================
    # Step 2: Extract lower slice data
    # ========================================================================
    lower_indices = np.where(lower_mask)[0]
    lower_xyz = xyz[lower_mask]
    lower_vert = verticality[lower_mask]
    lower_lin = linearity[lower_mask]
    lower_height = dist_to_ground[lower_mask]
    
    # ========================================================================
    # Step 3: Voxelize lower slice
    # ========================================================================
    voxel_indices = np.floor(lower_xyz / voxel_size).astype(np.int32)
    dtype = [('x', np.int32), ('y', np.int32), ('z', np.int32)]
    voxel_struct = np.empty(len(voxel_indices), dtype=dtype)
    voxel_struct['x'] = voxel_indices[:, 0]
    voxel_struct['y'] = voxel_indices[:, 1]
    voxel_struct['z'] = voxel_indices[:, 2]
    
    unique_voxels, inverse_indices = np.unique(voxel_struct, return_inverse=True)
    n_voxels = len(unique_voxels)
    
    # Per-voxel features
    voxel_vert_sum = np.bincount(inverse_indices, weights=lower_vert, minlength=n_voxels)
    voxel_lin_sum = np.bincount(inverse_indices, weights=lower_lin, minlength=n_voxels)
    voxel_counts = np.bincount(inverse_indices, minlength=n_voxels).astype(np.float32)
    voxel_counts[voxel_counts == 0] = 1
    
    voxel_vert = voxel_vert_sum / voxel_counts
    voxel_lin = voxel_lin_sum / voxel_counts
    
    # ========================================================================
    # Step 4: Identify seeds (high verticality + linearity = trunk base)
    # ========================================================================
    seed_voxel_mask = (voxel_vert > seed_verticality) & (voxel_lin > seed_linearity)
    n_seeds = np.sum(seed_voxel_mask)
    
    if n_seeds == 0:
        if verbose:
            print("  WARNING: No seeds found in lower slice. Keeping all as understory.")
        return SliceFilterResult(
            is_tree=is_tree,
            n_tree=n_upper,
            n_understory=n_lower,
            n_protected=n_upper,
            n_filtered_lower=n_lower
        )
    
    if verbose:
        print(f"  Seeds in lower slice: {n_seeds:,} voxels")
    
    # ========================================================================
    # Step 5: Build adjacency graph for lower slice voxels
    # ========================================================================
    voxel_coords = np.column_stack([unique_voxels['x'], unique_voxels['y'], unique_voxels['z']])
    voxel_to_idx = {(v[0], v[1], v[2]): i for i, v in enumerate(voxel_coords)}
    
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx != 0 or dy != 0 or dz != 0:
                    offsets.append((dx, dy, dz))
    
    rows, cols = [], []
    for offset in offsets:
        shifted = voxel_coords + np.array(offset, dtype=np.int32)
        for i, (x, y, z) in enumerate(shifted):
            if (x, y, z) in voxel_to_idx:
                rows.append(i)
                cols.append(voxel_to_idx[(x, y, z)])
    
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_voxels, n_voxels))
    
    # ========================================================================
    # Step 6: Region growing from seeds
    # ========================================================================
    is_tree_voxel = seed_voxel_mask.copy()
    previous_tree_voxel = np.zeros(n_voxels, dtype=bool)
    
    for iteration in range(max_iterations):
        frontier_mask = is_tree_voxel & ~previous_tree_voxel
        if np.sum(frontier_mask) == 0:
            break
        
        previous_tree_voxel = is_tree_voxel.copy()
        frontier_indices = np.where(frontier_mask)[0]
        neighbor_rows = adj[frontier_indices].nonzero()[1]
        unique_neighbors = np.unique(neighbor_rows)
        
        candidates = unique_neighbors[~is_tree_voxel[unique_neighbors]]
        if len(candidates) > 0:
            valid = candidates[voxel_vert[candidates] > expansion_verticality]
            is_tree_voxel[valid] = True
    
    # ========================================================================
    # Step 7: Map voxel results to lower slice points
    # ========================================================================
    is_tree_lower = is_tree_voxel[inverse_indices]
    
    # Update global mask: lower slice points that are tree
    is_tree[lower_indices[is_tree_lower]] = True
    
    # ========================================================================
    # Step 8: Statistics
    # ========================================================================
    n_tree_lower = np.sum(is_tree_lower)
    n_understory_lower = n_lower - n_tree_lower
    n_tree_total = np.sum(is_tree)
    
    if verbose:
        print(f"  ✓ Slice filtering complete in {time.time()-t0:.1f}s")
        print(f"  Lower slice: {n_tree_lower:,} tree, {n_understory_lower:,} understory removed")
        print(f"  Total: {n_tree_total:,} tree ({100*n_tree_total/n_points:.1f}%)")
    
    return SliceFilterResult(
        is_tree=is_tree,
        n_tree=n_tree_total,
        n_understory=n_points - n_tree_total,
        n_protected=n_upper,
        n_filtered_lower=n_understory_lower
    )


def postfilter_clusters(
    xyz: np.ndarray,
    is_tree: np.ndarray,
    dist_to_ground: np.ndarray,
    filter_height: float = 3.0,
    min_cluster_points: int = 100,
    voxel_size: float = 0.15,
    verbose: bool = False
) -> np.ndarray:
    """
    Post-filter clusters in lower zone that don't connect to upper zone.
    
    Strategy:
    1. Find all "tree" voxels in lower zone (<filter_height)
    2. Find connected components
    3. For each cluster, check if ANY voxel connects to upper zone
    4. Remove clusters that have no vertical connection (isolated understory)
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates.
    is_tree : np.ndarray
        Boolean mask from previous filtering (True = tree).
    dist_to_ground : np.ndarray
        Height above ground for each point.
    filter_height : float
        Height threshold for filtering. Only clusters below this are evaluated.
    min_cluster_points : int
        Minimum points for a cluster to be kept (regardless of connectivity).
    voxel_size : float
        Voxel size for connectivity analysis.
    verbose : bool
        Print progress.
    
    Returns
    -------
    np.ndarray
        Updated is_tree mask with isolated low clusters removed.
    """
    from scipy.sparse.csgraph import connected_components
    
    n_points = len(xyz)
    is_tree_out = is_tree.copy()
    
    if verbose:
        print(f"Post-filtering clusters below {filter_height}m...")
        t0 = time.time()
    
    # ========================================================================
    # Step 1: Identify tree points in lower zone
    # ========================================================================
    lower_tree_mask = is_tree & (dist_to_ground < filter_height)
    upper_tree_mask = is_tree & (dist_to_ground >= filter_height)
    
    n_lower_tree = np.sum(lower_tree_mask)
    if n_lower_tree == 0:
        if verbose:
            print("  No tree points in lower zone to filter.")
        return is_tree_out
    
    if verbose:
        print(f"  Lower zone tree points: {n_lower_tree:,}")
        print(f"  Upper zone tree points: {np.sum(upper_tree_mask):,}")
    
    # ========================================================================
    # Step 2: Voxelize ALL tree points (lower + upper)
    # ========================================================================
    tree_indices = np.where(is_tree)[0]
    tree_xyz = xyz[is_tree]
    tree_heights = dist_to_ground[is_tree]
    
    voxel_indices = np.floor(tree_xyz / voxel_size).astype(np.int32)
    dtype = [('x', np.int32), ('y', np.int32), ('z', np.int32)]
    voxel_struct = np.empty(len(voxel_indices), dtype=dtype)
    voxel_struct['x'] = voxel_indices[:, 0]
    voxel_struct['y'] = voxel_indices[:, 1]
    voxel_struct['z'] = voxel_indices[:, 2]
    
    unique_voxels, inverse_indices = np.unique(voxel_struct, return_inverse=True)
    n_voxels = len(unique_voxels)
    
    # Per-voxel average height
    voxel_height_sum = np.bincount(inverse_indices, weights=tree_heights, minlength=n_voxels)
    voxel_counts = np.bincount(inverse_indices, minlength=n_voxels).astype(np.float32)
    voxel_counts[voxel_counts == 0] = 1
    voxel_height = voxel_height_sum / voxel_counts
    
    # Which voxels are in lower zone?
    lower_voxel_mask = voxel_height < filter_height
    upper_voxel_mask = ~lower_voxel_mask
    
    if verbose:
        print(f"  Voxels: {n_voxels} total, {np.sum(lower_voxel_mask)} lower, {np.sum(upper_voxel_mask)} upper")
    
    # ========================================================================
    # Step 3: Build adjacency graph
    # ========================================================================
    voxel_coords = np.column_stack([unique_voxels['x'], unique_voxels['y'], unique_voxels['z']])
    voxel_to_idx = {(v[0], v[1], v[2]): i for i, v in enumerate(voxel_coords)}
    
    rows, cols = [], []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx != 0 or dy != 0 or dz != 0:
                    shifted = voxel_coords + np.array([dx, dy, dz], dtype=np.int32)
                    for i, (x, y, z) in enumerate(shifted):
                        if (x, y, z) in voxel_to_idx:
                            rows.append(i)
                            cols.append(voxel_to_idx[(x, y, z)])
    
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_voxels, n_voxels))
    
    # ========================================================================
    # Step 4: Find connected components
    # ========================================================================
    n_components, labels = connected_components(adj, directed=False)
    
    if verbose:
        print(f"  Found {n_components} connected components")
    
    # ========================================================================
    # Step 5: For each component, check if it has upper zone voxels
    # ========================================================================
    # Component is "connected to upper" if ANY of its voxels is in upper zone
    component_has_upper = np.zeros(n_components, dtype=bool)
    for comp_id in range(n_components):
        comp_mask = labels == comp_id
        if np.any(upper_voxel_mask[comp_mask]):
            component_has_upper[comp_id] = True
    
    # Count points per component
    point_labels = labels[inverse_indices]
    component_point_counts = np.bincount(point_labels, minlength=n_components)
    
    # ========================================================================
    # Step 6: Remove lower zone points from components without upper connection
    # ========================================================================
    n_removed = 0
    for comp_id in range(n_components):
        if component_has_upper[comp_id]:
            continue  # This component connects to trunk, keep all
        
        if component_point_counts[comp_id] >= min_cluster_points:
            continue  # Large cluster, might be a short tree, keep it
        
        # This is a small, isolated cluster in lower zone -> remove
        comp_point_mask = point_labels == comp_id
        # Map back to original indices
        remove_original_indices = tree_indices[comp_point_mask]
        # Only remove points in lower zone
        for idx in remove_original_indices:
            if dist_to_ground[idx] < filter_height:
                is_tree_out[idx] = False
                n_removed += 1
    
    if verbose:
        print(f"  ✓ Removed {n_removed:,} points from isolated low clusters")
        print(f"  Post-filter complete in {time.time()-t0:.1f}s")
    
    return is_tree_out
