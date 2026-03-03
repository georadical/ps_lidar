"""
Branch Extraction Module — Brick 8

Extracts tree branches from a height-normalized point cloud using
linearity-based filtering and topological connectivity to trunks.

Pipeline:
  1. Compute linearity on non-trunk points (pgeof)
  2. Filter by linearity threshold (high linearity = wood)
  3. Voxelize candidates
  4. Build 26-neighbour connectivity graph
  5. Connected components — keep only components connected to trunk
  6. Filter by max_branch_length from nearest trunk axis

Field parameters:
  - max_branch_length: longest expected branch in the plot
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .features import compute_linearity
from .trunk_extraction import TrunkExtractionResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BranchExtractionConfig:
    """
    Configuration for branch extraction.

    Field parameters:
        max_branch_length: Maximum expected branch length (measured in field).

    Algorithm parameters:
        linearity_threshold: Minimum linearity to consider a point as wood.
        verticality_scale: Neighbourhood radius for PCA via pgeof.
        connectivity_radius: Voxel edge length for connectivity graph.
        min_branch_points: Minimum points for a valid branch cluster.
        max_sphericity: Maximum sphericity (filters out foliage clusters).
    """
    # Field parameters
    max_branch_length: float = 5.0      # m

    # Algorithm parameters
    linearity_threshold: float = 0.5
    verticality_scale: float = 0.1      # m
    connectivity_radius: float = 0.05   # m (voxel size for connectivity)
    min_branch_points: int = 50
    max_sphericity: Optional[float] = None  # if set, filter high-sphericity points


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BranchExtractionResult:
    """Result of branch extraction."""
    wood_mask: np.ndarray           # (N,) bool — trunk + branches
    branch_only_mask: np.ndarray    # (N,) bool — only branches (not trunk)
    n_branch_clusters: int          # number of branch clusters found
    n_branch_points: int            # total branch points
    config: BranchExtractionConfig  # config used


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_branches(
    xyz: np.ndarray,
    trunk_result: TrunkExtractionResult,
    config: Optional[BranchExtractionConfig] = None,
    verbose: bool = False,
) -> BranchExtractionResult:
    """
    Extract tree branches from a height-normalized point cloud.

    Starts from trunk extraction results and identifies linear structures
    connected to trunk axes.

    Pipeline:
      1. Select non-trunk points
      2. Compute linearity (pgeof)
      3. Filter by linearity threshold
      4. Voxelize and build connectivity graph
      5. Keep only components connected to trunk
      6. Filter by max_branch_length

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) height-normalized point cloud.
    trunk_result : TrunkExtractionResult
        Output from extract_trunks().
    config : BranchExtractionConfig, optional
        Configuration. Uses defaults if None.
    verbose : bool
        Print progress information.

    Returns
    -------
    BranchExtractionResult
        Contains wood_mask (trunk+branches), branch_only_mask, etc.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if config is None:
        config = BranchExtractionConfig()

    n_total = len(xyz)
    trunk_mask = trunk_result.trunk_mask

    if verbose:
        print(f"Branch extraction: {n_total:,} points")
        print(f"  Trunk points: {trunk_mask.sum():,}")
        print(f"  Linearity threshold: {config.linearity_threshold}")
        print(f"  Max branch length: {config.max_branch_length}m")

    # ======================================================================
    # Step 1: Select candidate points (non-trunk, above ground)
    # ======================================================================
    candidate_mask = ~trunk_mask & (xyz[:, 2] > 0.3)  # above ground, not trunk
    candidate_indices = np.where(candidate_mask)[0]
    candidate_xyz = xyz[candidate_mask]

    if verbose:
        print(f"  Candidate points: {len(candidate_xyz):,}")

    if len(candidate_xyz) < config.min_branch_points:
        if verbose:
            print("  Not enough candidate points for branch detection")
        return BranchExtractionResult(
            wood_mask=trunk_mask.copy(),
            branch_only_mask=np.zeros(n_total, dtype=bool),
            n_branch_clusters=0,
            n_branch_points=0,
            config=config,
        )

    # ======================================================================
    # Step 2: Compute linearity
    # ======================================================================
    linearity = compute_linearity(
        candidate_xyz,
        scale=config.verticality_scale,
        voxel_resolution_xy=config.connectivity_radius,
        voxel_resolution_z=config.connectivity_radius,
        verbose=verbose,
    )

    # ======================================================================
    # Step 3: Filter by linearity
    # ======================================================================
    linear_mask = linearity >= config.linearity_threshold
    linear_indices = candidate_indices[linear_mask]
    linear_xyz = xyz[linear_indices]

    if verbose:
        print(f"  Linear points: {len(linear_xyz):,} "
              f"({100 * len(linear_xyz) / max(len(candidate_xyz), 1):.1f}%)")

    if len(linear_xyz) < config.min_branch_points:
        if verbose:
            print("  Not enough linear points for branch detection")
        return BranchExtractionResult(
            wood_mask=trunk_mask.copy(),
            branch_only_mask=np.zeros(n_total, dtype=bool),
            n_branch_clusters=0,
            n_branch_points=0,
            config=config,
        )

    # ======================================================================
    # Step 4: Voxelize and build connectivity graph
    # ======================================================================
    # Combine trunk + linear candidate points for a unified connectivity graph
    trunk_indices = np.where(trunk_mask)[0]
    combined_indices = np.concatenate([trunk_indices, linear_indices])
    combined_xyz = xyz[combined_indices]

    # Track which combined points are trunk vs candidate
    is_trunk_in_combined = np.zeros(len(combined_indices), dtype=bool)
    is_trunk_in_combined[:len(trunk_indices)] = True

    voxel_size = config.connectivity_radius
    voxel_idx = np.floor(combined_xyz / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    n_voxels = len(unique_voxels)

    if verbose:
        print(f"  Connectivity graph: {n_voxels:,} voxels")

    # Determine which voxels contain trunk points
    trunk_voxel_ids = np.unique(inverse[is_trunk_in_combined])

    # Build 26-neighbour adjacency
    voxel_to_id = {tuple(v): i for i, v in enumerate(unique_voxels)}

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx != 0 or dy != 0 or dz != 0:
                    offsets.append((dx, dy, dz))

    rows = []
    cols = []
    for i, voxel in enumerate(unique_voxels):
        vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
        for dx, dy, dz in offsets:
            j = voxel_to_id.get((vx + dx, vy + dy, vz + dz))
            if j is not None:
                rows.append(i)
                cols.append(j)

    if rows:
        adjacency = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n_voxels, n_voxels),
        )
        n_components, voxel_labels = connected_components(
            adjacency, directed=False, return_labels=True,
        )
    else:
        n_components = n_voxels
        voxel_labels = np.arange(n_voxels, dtype=np.int32)

    if verbose:
        print(f"  Connected components: {n_components:,}")

    # ======================================================================
    # Step 5: Keep components connected to trunk
    # ======================================================================
    trunk_component_ids = np.unique(voxel_labels[trunk_voxel_ids])

    connected_voxel_mask = np.isin(voxel_labels, trunk_component_ids)
    connected_point_mask = connected_voxel_mask[inverse]

    # Points that are connected and are NOT trunk = branches
    branch_in_combined = connected_point_mask & ~is_trunk_in_combined
    branch_original_indices = combined_indices[branch_in_combined]

    n_connected_components = len(trunk_component_ids)
    if verbose:
        print(f"  Trunk-connected components: {n_connected_components:,}")
        print(f"  Branch candidates (connected): {len(branch_original_indices):,}")

    # ======================================================================
    # Step 6: Filter by max_branch_length
    # ======================================================================
    if len(branch_original_indices) > 0 and len(trunk_indices) > 0:
        trunk_tree = cKDTree(xyz[trunk_indices])
        branch_xyz = xyz[branch_original_indices]
        dist_to_trunk, _ = trunk_tree.query(branch_xyz, k=1, workers=-1)

        within_range = dist_to_trunk <= config.max_branch_length
        branch_original_indices = branch_original_indices[within_range]

        if verbose:
            n_filtered = (~within_range).sum()
            print(f"  Filtered (too far from trunk): {n_filtered:,}")

    # ======================================================================
    # Build result
    # ======================================================================
    branch_only_mask = np.zeros(n_total, dtype=bool)
    if len(branch_original_indices) > 0:
        branch_only_mask[branch_original_indices] = True

    wood_mask = trunk_mask | branch_only_mask
    n_branch_points = branch_only_mask.sum()

    # Count branch clusters
    if n_branch_points > 0:
        # Simple connected component count on branch voxels
        branch_voxel_idx = np.floor(xyz[branch_only_mask] / voxel_size).astype(np.int32)
        unique_branch_voxels = np.unique(branch_voxel_idx, axis=0)
        n_branch_clusters = min(len(unique_branch_voxels), n_connected_components)
    else:
        n_branch_clusters = 0

    if verbose:
        print(f"  Final branch points: {n_branch_points:,}")
        print(f"  Total wood points: {wood_mask.sum():,} "
              f"({100 * wood_mask.sum() / n_total:.1f}%)")

    return BranchExtractionResult(
        wood_mask=wood_mask,
        branch_only_mask=branch_only_mask,
        n_branch_clusters=n_branch_clusters,
        n_branch_points=int(n_branch_points),
        config=config,
    )
