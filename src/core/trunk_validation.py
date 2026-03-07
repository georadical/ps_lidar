"""
Trunk validation — point-level cylinder scrubbing.

Instead of accepting/rejecting entire trees, this module SCRUBS each
trunk point-by-point: for each tree, it slices the trunk into horizontal
sections, fits a circle per section, and keeps only points that fall
within the fitted radius + a small offset. Points outside are reclassified
as non-trunk (branches/understory).

After scrubbing, trees with too few remaining trunk points or too short
total height are removed entirely.

This approach:
  - Cleans real trunks of attached understory/branches
  - Does NOT remove entire trees (no neighbour damage)
  - Residual understory-only trees shrink to ~0 points → filtered by count/height
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

from src.core.trunk_extraction import TrunkExtractionResult


@dataclass
class TrunkScrubConfig:
    """
    Configuration for point-level trunk scrubbing.

    section_height: Height of each horizontal slice (metres).
    radius_offset: Extra radius beyond fitted circle to keep (metres).
        Controls how aggressively to scrub. Larger = more permissive.
    min_points_per_section: Minimum points to attempt circle fitting.
        Sections with fewer points are kept as-is (no filtering).
    min_trunk_points_after: Minimum trunk points remaining after scrubbing
        for a tree to be kept.
    min_trunk_height: Minimum vertical extent of remaining trunk points (metres).
    dbh_max: Maximum expected trunk diameter (metres). Sections with fitted
        diameter larger than this * safety_factor are fully cleaned.
    safety_factor: Multiplier on dbh_max for absolute maximum diameter.
    percentile: Percentile for robust radius estimation (default 75th).
        Using percentile instead of mean avoids influence from outlier
        understory points.
    """
    section_height: float = 1.0         # metres
    radius_offset: float = 0.03         # metres (3cm tolerance)
    min_points_per_section: int = 30
    min_trunk_points_after: int = 200
    min_trunk_height: float = 5.0       # metres
    dbh_max: float = 0.80               # metres (from field)
    safety_factor: float = 1.5
    percentile: float = 75.0            # robust radius percentile


@dataclass
class TreeScrubResult:
    """Per-tree scrubbing diagnostics."""
    tree_id: int
    points_before: int
    points_after: int
    points_removed: int
    n_sections: int
    n_sections_scrubbed: int
    section_radii: List[float]    # fitted radius per section
    height_before: float
    height_after: float
    removed_entirely: bool
    removal_reason: str           # "" if kept


@dataclass
class TrunkScrubResult:
    """Result of trunk scrubbing."""
    trunk_mask: np.ndarray           # (N,) bool — updated trunk mask
    tree_ids: np.ndarray             # (N,) int — updated tree ids
    n_trees_before: int
    n_trees_after: int
    n_trees_removed: int
    total_points_before: int
    total_points_after: int
    total_points_scrubbed: int
    removed_tree_ids: List[int]
    tree_results: List[TreeScrubResult]
    config: TrunkScrubConfig


def _project_to_perpendicular_plane(
    points: np.ndarray,
    axis_direction: np.ndarray,
) -> np.ndarray:
    """
    Project 3D points onto the plane perpendicular to the axis direction.
    Returns 2D coordinates in the plane.
    """
    axis = axis_direction / np.linalg.norm(axis_direction)

    if abs(axis[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    u = ref - np.dot(ref, axis) * axis
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    return np.column_stack([points @ u, points @ v])


def _fit_circle_robust(
    points_2d: np.ndarray,
    percentile: float = 75.0,
) -> Tuple[np.ndarray, float]:
    """
    Robust circle fit using median centre and percentile radius.

    Returns:
        centre_2d: (2,) estimated centre
        radius: estimated radius at the given percentile
    """
    # Use median as robust centre estimate (resistant to outliers)
    centre = np.median(points_2d, axis=0)

    # Compute radial distances from centre
    radii = np.linalg.norm(points_2d - centre, axis=1)

    # Use percentile radius — more robust than mean
    # The 75th percentile captures the "shell" of the cylinder
    # while being resistant to outlier understory points
    radius = np.percentile(radii, percentile)

    return centre, float(radius)


def _scrub_single_tree(
    trunk_points: np.ndarray,
    global_indices: np.ndarray,
    axis_direction: np.ndarray,
    tree_id: int,
    config: TrunkScrubConfig,
) -> Tuple[np.ndarray, TreeScrubResult]:
    """
    Scrub a single tree's trunk points section by section.

    Returns:
        keep_mask: boolean mask over the GLOBAL indices (True = keep as trunk)
        result: diagnostics for this tree
    """
    n_before = len(trunk_points)
    z_min = trunk_points[:, 2].min()
    z_max = trunk_points[:, 2].max()
    height_before = z_max - z_min

    # Start with all points marked to keep
    keep = np.ones(len(trunk_points), dtype=bool)
    section_radii = []
    n_scrubbed = 0
    max_radius = (config.dbh_max * config.safety_factor) / 2.0

    # Slice into horizontal sections
    section_edges = np.arange(z_min, z_max, config.section_height)

    for z_low in section_edges:
        z_high = z_low + config.section_height
        section_mask = (trunk_points[:, 2] >= z_low) & (trunk_points[:, 2] < z_high)
        n_section = section_mask.sum()

        if n_section < config.min_points_per_section:
            # Too few points for reliable fitting — keep all
            continue

        section_pts = trunk_points[section_mask]

        # Project onto plane perpendicular to tree axis
        pts_2d = _project_to_perpendicular_plane(section_pts, axis_direction)

        # Robust circle fit
        centre, radius = _fit_circle_robust(pts_2d, config.percentile)

        # Clamp radius to maximum expected
        radius = min(radius, max_radius)
        section_radii.append(radius)

        # Keep only points within radius + offset
        scrub_radius = radius + config.radius_offset
        distances = np.linalg.norm(pts_2d - centre, axis=1)
        outside = distances > scrub_radius

        if outside.any():
            # Find which points in the section to remove
            section_indices = np.where(section_mask)[0]
            remove_indices = section_indices[outside]
            keep[remove_indices] = False
            n_scrubbed += 1

    n_after = keep.sum()
    points_removed = n_before - n_after

    # Compute height of remaining points
    if n_after > 0:
        remaining_z = trunk_points[keep, 2]
        height_after = remaining_z.max() - remaining_z.min()
    else:
        height_after = 0.0

    # Check if tree should be removed entirely
    removed = False
    reason = ""
    if n_after < config.min_trunk_points_after:
        removed = True
        reason = f"too_few_points={n_after}<{config.min_trunk_points_after}"
        keep[:] = False
    elif height_after < config.min_trunk_height:
        removed = True
        reason = f"too_short={height_after:.1f}<{config.min_trunk_height}"
        keep[:] = False

    result = TreeScrubResult(
        tree_id=tree_id,
        points_before=n_before,
        points_after=int(keep.sum()),
        points_removed=n_before - int(keep.sum()),
        n_sections=len(section_edges),
        n_sections_scrubbed=n_scrubbed,
        section_radii=section_radii,
        height_before=height_before,
        height_after=height_after,
        removed_entirely=removed,
        removal_reason=reason,
    )

    return keep, result


def scrub_trunks(
    xyz: np.ndarray,
    trunk_result: TrunkExtractionResult,
    config: TrunkScrubConfig,
) -> TrunkScrubResult:
    """
    Scrub trunk points using section-wise cylinder fitting.

    For each tree, slices trunk into horizontal sections. Per section,
    fits a circle and keeps only points within the fitted radius + offset.
    Points outside are reclassified as non-trunk.

    After scrubbing, trees with too few remaining points or insufficient
    height are removed entirely.

    Args:
        xyz: (N, 3) full point cloud (height-normalised).
        trunk_result: Output of extract_trunks().
        config: Scrubbing parameters.

    Returns:
        TrunkScrubResult with updated masks and per-tree diagnostics.
    """
    print(f"\nTrunk scrubbing: {trunk_result.n_trees} trees")
    print(f"  Section height: {config.section_height}m")
    print(f"  Radius offset:  {config.radius_offset}m")
    print(f"  Max radius:     {(config.dbh_max * config.safety_factor) / 2:.3f}m")
    print(f"  Min points:     {config.min_trunk_points_after}")
    print(f"  Min height:     {config.min_trunk_height}m")

    trunk_mask = trunk_result.trunk_mask.copy()
    tree_ids = trunk_result.tree_ids.copy()
    total_before = int(trunk_mask.sum())

    # Build axis lookup
    axis_by_id = {}
    for ax in trunk_result.tree_axes:
        axis_by_id[ax["tree_id"]] = ax

    tree_results = []
    removed_ids = []
    unique_trees = sorted(set(tree_ids[trunk_mask]) - {-1})

    for tid in unique_trees:
        # Get trunk points for this tree
        tree_trunk_mask = trunk_mask & (tree_ids == tid)
        global_indices = np.where(tree_trunk_mask)[0]
        trunk_pts = xyz[global_indices]

        # Get axis direction
        ax = axis_by_id.get(tid)
        axis_dir = ax["direction"] if ax is not None else np.array([0, 0, 1.0])

        # Scrub this tree
        keep, result = _scrub_single_tree(
            trunk_pts, global_indices, axis_dir, tid, config
        )
        tree_results.append(result)

        # Apply scrubbing to global masks
        remove_indices = global_indices[~keep]
        if len(remove_indices) > 0:
            trunk_mask[remove_indices] = False

        if result.removed_entirely:
            # Also clear tree_ids for this tree
            tree_ids[tree_ids == tid] = -1
            removed_ids.append(tid)

        # Print summary per tree
        pct = (result.points_removed / max(result.points_before, 1)) * 100
        mean_r = np.mean(result.section_radii) if result.section_radii else 0
        if result.removed_entirely:
            print(f"  Tree {tid:3d}: ✗ REMOVED "
                  f"({result.removal_reason})")
        else:
            print(f"  Tree {tid:3d}: ✓ scrubbed "
                  f"{result.points_before:,} → {result.points_after:,} pts "
                  f"(-{pct:.0f}%), "
                  f"r̄={mean_r:.3f}m, "
                  f"h={result.height_after:.1f}m")

    total_after = int(trunk_mask.sum())
    n_after = trunk_result.n_trees - len(removed_ids)

    print(f"\n  Summary:")
    print(f"    Trees:  {trunk_result.n_trees} → {n_after} "
          f"({len(removed_ids)} removed)")
    print(f"    Points: {total_before:,} → {total_after:,} "
          f"({total_before - total_after:,} scrubbed)")

    return TrunkScrubResult(
        trunk_mask=trunk_mask,
        tree_ids=tree_ids,
        n_trees_before=trunk_result.n_trees,
        n_trees_after=n_after,
        n_trees_removed=len(removed_ids),
        total_points_before=total_before,
        total_points_after=total_after,
        total_points_scrubbed=total_before - total_after,
        removed_tree_ids=removed_ids,
        tree_results=tree_results,
        config=config,
    )
