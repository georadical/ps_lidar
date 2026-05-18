"""
Trunk Validation — Stem Cleaning + Sectioning (inspired by 3DFin).

This module implements the post-extraction refinement of trunk points:

Step 1: clean_stems()
    Includes optional fine-grained profiling (emitted via verbose mode and
    stored in StemCleaningResult.profile) to diagnose runtime bottlenecks.

    Takes trunk_mask points and runs a SECOND verticality pass to remove
    non-vertical material (branches, attached understory, foliage) that
    survived the initial trunk extraction. This is 3DFin's Step 4.

Step 2: compute_stem_sections()
    Slices each cleaned stem into horizontal sections and fits circles
    using least-squares optimisation (scipy.optimize.leastsq). Returns
    per-section centres, radii, and quality metrics. This is 3DFin's Step 5.

The output of Step 2 can later be used for:
  - Per-section adaptive trunk mask refinement (Step 3, future)
  - DAP estimation
  - Taper curves
  - Feature extraction (sweep, knots, fluting, etc.)
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Tuple, Optional, Literal
import numpy as np
from scipy import optimize
from scipy.ndimage import median_filter

from src.core.features import (
    voxelize_cloud,
    compute_verticality,
    compute_verticality_mask_early_exit,
)
from src.core.trunk_extraction import TrunkExtractionResult
from src.core.ellipse_fitting import EllipseFitConfig, _fit_ellipse_check


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StemCleaningConfig:
    """
    Configuration for stem cleaning and sectioning.

    Verticality pass parameters:
        verticality_threshold: Minimum verticality to keep a point as stem.
        verticality_scale: Neighbourhood radius for PCA.
        voxel_resolution_xy: Horizontal voxel size.
        voxel_resolution_z: Vertical voxel size.

    Sectioning parameters (like 3DFin):
        section_len: Distance between section centres (metres).
        section_wid: Half-width of each section slice (metres).
            Points within [section_h - wid, section_h + wid] are used.
        min_points_section: Minimum points for circle fitting.
        r_min: Minimum valid fitted radius (metres).
        r_max: Maximum valid fitted radius (metres).
        n_sectors: Number of angular sectors for quality check.
        min_sectors: Minimum occupied sectors for valid fit.
        sector_width: Width around fitted circle to check sectors (metres).
        inner_circle_ratio: Ratio for inner circle quality test.
        max_inner_points: Maximum points in inner circle before flagging.
        minimum_height: Lowest height for sections (metres).
        maximum_height: Highest height for sections (metres).
    """
    # 2nd verticality pass
    verticality_threshold: float = 0.7
    verticality_scale: float = 0.1
    voxel_resolution_xy: float = 0.02
    voxel_resolution_z: float = 0.02
    # Coarser voxel resolution for the 2nd verticality pass (clean_stems).
    # Set to e.g. 0.04 for ~1.2× speedup (but ~0.5% mask difference).
    # Defaults to None → reuses the primary voxel_resolution_xy / _z values
    # for exact output equivalence with the pre-optimisation behaviour.
    voxel_resolution_xy_2nd: Optional[float] = None
    voxel_resolution_z_2nd: Optional[float] = None

    # Sectioning
    section_len: float = 0.2       # distance between sections (m)
    section_wid: float = 0.05      # half-width of section slice (m)
    min_points_section: int = 80
    r_min: float = 0.03            # minimum radius (m)
    r_max: float = 0.50            # maximum radius (m)
    n_sectors: int = 16
    min_sectors: int = 9
    sector_width: float = 0.02     # width around circle for sector check (m)
    inner_circle_ratio: float = 0.5
    max_inner_points: int = 5
    minimum_height: float = 0.3    # lowest section (m)
    maximum_height: float = 25.0   # highest section (m)
    cluster_eps: float = 0.02      # DBSCAN eps for section clustering (m)

    # Execution mode
    mode: Literal["global", "suspicious_only"] = "global"
    suspicious_axis_tilt_deg: float = 10.0
    suspicious_stripe_circularity_max: float = 0.45
    suspicious_trunk_to_stripe_ratio_min: float = 10.0
    suspicious_point_count_min: int = 25_000
    global_fallback_point_ratio: float = 0.60
    global_fallback_tree_ratio: float = 0.50

    # Parallel execution (suspicious_only mode only).
    # Uses ThreadPoolExecutor — pgeof (C++ backend) releases the GIL,
    # so threads achieve true parallelism for the verticality computation.
    parallel: bool = False
    parallel_max_workers: Optional[int] = None  # None = cpu_count()

    # Voxel-level early exit — two-tier screening.
    # Coarse pass auto-keeps obviously vertical voxels; remaining voxels run
    # the full-resolution fine pass.
    #
    # Real-data benchmark (T460298A, 17.5M points, 32 trees, mode="global"):
    #   speedup       ~1.11x   (below the 1.20x bar to promote to default)
    #   workload red. ~0.0%    (auto-keep barely fires on this dataset)
    #   mask diff     ~1.82%   (>0.5% — output drift vs baseline)
    #   recall loss   104,416 points dropped that the baseline kept
    # Conclusion: kept opt-in, NOT default. Bidirectional auto-decisions
    # (auto-keep + auto-reject) would be needed to deliver the original
    # 10–50× ambition of the plan; not pursued for now (low marginal value).
    voxel_early_exit: bool = False
    voxel_early_exit_coarse_resolution: float = 0.08
    voxel_early_exit_margin: float = 0.1


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StemCleaningResult:
    """Result of stem cleaning (2nd verticality pass)."""
    stem_mask: np.ndarray         # (N,) bool — updated stem mask
    n_points_before: int
    n_points_after: int
    n_points_removed: int
    per_tree_stats: List[Dict[str, Any]]
    mode_used: str = "global"
    n_trees_processed: int = 0
    n_trees_skipped: int = 0
    n_points_processed_verticality: int = 0
    used_global_fallback: bool = False
    profile: Optional[Dict[str, Any]] = None


@dataclass
class SectionResult:
    """Result of stem sectioning (circle fitting)."""
    X_c: np.ndarray               # (n_trees, n_sections) X centres
    Y_c: np.ndarray               # (n_trees, n_sections) Y centres
    R: np.ndarray                  # (n_trees, n_sections) radii
    check: np.ndarray             # (n_trees, n_sections) validity flag
    sector_pct: np.ndarray        # (n_trees, n_sections) sector occupancy %
    sections: np.ndarray          # (n_sections,) section heights
    tree_ids: List[int]           # tree IDs in order
    config: StemCleaningConfig


# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------

def _format_profile_summary(profile: Dict[str, Any]) -> str:
    """Format a profiling dict into a human-readable summary table."""
    lines = ["\n  ── Profiling breakdown ──"]
    total = profile.get("total_seconds", 0.0)
    stages = [
        ("input_preparation", "Input preparation"),
        ("suspicious_scoring", "Suspicious scoring"),
        ("verticality_computation", "Verticality computation"),
        ("threshold_filtering", "Threshold filtering"),
        ("mask_update", "Mask update"),
        ("stats_aggregation", "Stats aggregation"),
    ]
    for key, label in stages:
        secs = profile.get(f"{key}_seconds", 0.0)
        if secs > 0.0 or key in ("verticality_computation",):
            pct = (secs / total * 100.0) if total > 0 else 0.0
            lines.append(f"    {label:30s} {secs:8.3f}s  ({pct:5.1f}%)")
    lines.append(f"    {'TOTAL':30s} {total:8.3f}s")
    lines.append(f"    Points → verticality:       {profile.get('n_points_verticality', 0):,}")
    if profile.get("per_tree_timings"):
        lines.append(f"    Trees processed individually: {len(profile['per_tree_timings'])}")
        for t in profile["per_tree_timings"]:
            lines.append(
                f"      Tree {t['tree_id']:3d}: "
                f"{t['n_points']:>7,} pts, "
                f"vert={t['verticality_seconds']:.3f}s, "
                f"filter={t['filtering_seconds']:.3f}s, "
                f"total={t['tree_total_seconds']:.3f}s"
            )
    lines.append("  ── end profiling ──")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: Stem cleaning — 2nd verticality pass
# ---------------------------------------------------------------------------

def _axis_tilt_deg(direction: np.ndarray) -> float:
    """Angle between an axis direction and the global vertical axis."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return 90.0
    direction = direction / norm
    vertical_alignment = np.clip(abs(direction[2]), 0.0, 1.0)
    return float(np.degrees(np.arccos(vertical_alignment)))


def _unique_tree_ids(tree_ids: np.ndarray, stem_mask: np.ndarray) -> List[int]:
    """Sorted tree IDs present in a stem mask."""
    return sorted(set(tree_ids[stem_mask]) - {-1})


def _build_per_tree_stats(
    tree_ids: np.ndarray,
    stem_mask_before: np.ndarray,
    stem_mask_after: np.ndarray,
    unique_trees: List[int],
) -> List[Dict[str, Any]]:
    """Build per-tree before/after cleaning statistics."""
    per_tree = []
    for tid in unique_trees:
        tree_before = stem_mask_before & (tree_ids == tid)
        tree_after = stem_mask_after & (tree_ids == tid)
        n_before = int(tree_before.sum())
        n_after = int(tree_after.sum())
        pct_removed = ((n_before - n_after) / max(n_before, 1)) * 100.0
        per_tree.append(
            {
                "tree_id": tid,
                "before": n_before,
                "after": n_after,
                "removed": n_before - n_after,
                "pct_removed": pct_removed,
            }
        )
    return per_tree


def _score_suspicious_trees(
    trunk_mask: np.ndarray,
    tree_ids: np.ndarray,
    tree_axes: List[Dict[str, Any]],
    config: StemCleaningConfig,
) -> Tuple[List[int], int, int, List[int]]:
    """
    Select trees that justify a second verticality pass.

    Returns:
        suspicious_tree_ids, suspicious_points, total_trunk_points, unique_tree_ids
    """
    unique_trees = _unique_tree_ids(tree_ids, trunk_mask)
    axis_by_tid = {int(axis["tree_id"]): axis for axis in tree_axes}

    suspicious_tree_ids: List[int] = []
    suspicious_points = 0
    total_trunk_points = int(trunk_mask.sum())

    for tid in unique_trees:
        before_points = int(np.sum(trunk_mask & (tree_ids == tid)))
        axis = axis_by_tid.get(tid, {})
        stripe_circularity = float(axis.get("stripe_circularity", np.nan))
        stripe_points = int(axis.get("stripe_points", axis.get("n_points", 0)))
        trunk_to_stripe_ratio = float(before_points / max(stripe_points, 1))
        axis_tilt = _axis_tilt_deg(axis.get("direction", np.array([0.0, 0.0, 1.0])))

        is_suspicious = (
            axis_tilt > config.suspicious_axis_tilt_deg
            or (np.isfinite(stripe_circularity) and stripe_circularity < config.suspicious_stripe_circularity_max)
            or trunk_to_stripe_ratio > config.suspicious_trunk_to_stripe_ratio_min
            or before_points >= config.suspicious_point_count_min
        )
        if is_suspicious:
            suspicious_tree_ids.append(tid)
            suspicious_points += before_points

    return suspicious_tree_ids, suspicious_points, total_trunk_points, unique_trees


def _clean_stems_global(
    xyz: np.ndarray,
    trunk_result: TrunkExtractionResult,
    config: StemCleaningConfig,
    verbose: bool = True,
) -> StemCleaningResult:
    """Reference implementation: apply the second verticality pass to all trunk points."""
    t_total_start = time.perf_counter()

    # --- input preparation ---
    t0 = time.perf_counter()
    trunk_mask = trunk_result.trunk_mask.copy()
    tree_ids = trunk_result.tree_ids
    n_before = int(trunk_mask.sum())
    unique_trees = _unique_tree_ids(tree_ids, trunk_result.trunk_mask)
    trunk_indices = np.where(trunk_mask)[0]
    trunk_pts = xyz[trunk_indices]
    t_input_prep = time.perf_counter() - t0

    if verbose:
        print(f"\nStem cleaning (2nd verticality pass):")
        print("  Mode: global")
        print(f"  Input: {n_before:,} trunk points")
        print(f"  Verticality threshold: {config.verticality_threshold}")
        print(f"  Scale: {config.verticality_scale}m")

    if len(trunk_pts) == 0:
        profile = {
            "total_seconds": time.perf_counter() - t_total_start,
            "input_preparation_seconds": t_input_prep,
            "verticality_computation_seconds": 0.0,
            "threshold_filtering_seconds": 0.0,
            "mask_update_seconds": 0.0,
            "stats_aggregation_seconds": 0.0,
            "n_points_verticality": 0,
        }
        return StemCleaningResult(
            stem_mask=trunk_mask,
            n_points_before=0,
            n_points_after=0,
            n_points_removed=0,
            per_tree_stats=[],
            mode_used="global",
            n_trees_processed=0,
            n_trees_skipped=0,
            n_points_processed_verticality=0,
            used_global_fallback=False,
            profile=profile,
        )

    if verbose:
        print(f"  Computing verticality on {len(trunk_pts):,} trunk points...")

    # --- verticality computation + filtering ---
    vox_xy = config.voxel_resolution_xy_2nd or config.voxel_resolution_xy
    vox_z = config.voxel_resolution_z_2nd or config.voxel_resolution_z

    t0 = time.perf_counter()
    if config.voxel_early_exit:
        keep_mask, ee_stats = compute_verticality_mask_early_exit(
            trunk_pts,
            threshold=config.verticality_threshold,
            scale=config.verticality_scale,
            voxel_resolution_xy=vox_xy,
            voxel_resolution_z=vox_z,
            coarse_resolution=config.voxel_early_exit_coarse_resolution,
            margin=config.voxel_early_exit_margin,
        )
    else:
        vert = compute_verticality(
            trunk_pts,
            scale=config.verticality_scale,
            voxel_resolution_xy=vox_xy,
            voxel_resolution_z=vox_z,
        )
        keep_mask = vert >= config.verticality_threshold
        ee_stats = None
    t_vert = time.perf_counter() - t0

    t0 = time.perf_counter()
    remove_indices = trunk_indices[~keep_mask]
    t_filter = time.perf_counter() - t0

    # --- mask update ---
    t0 = time.perf_counter()
    trunk_mask[remove_indices] = False
    n_after = int(trunk_mask.sum())
    t_mask = time.perf_counter() - t0

    # --- stats aggregation ---
    t0 = time.perf_counter()
    per_tree = _build_per_tree_stats(tree_ids, trunk_result.trunk_mask, trunk_mask, unique_trees)
    t_stats = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start

    profile = {
        "total_seconds": t_total,
        "input_preparation_seconds": t_input_prep,
        "verticality_computation_seconds": t_vert,
        "threshold_filtering_seconds": t_filter,
        "mask_update_seconds": t_mask,
        "stats_aggregation_seconds": t_stats,
        "n_points_verticality": len(trunk_pts),
    }

    if verbose:
        for row in per_tree:
            print(
                f"    Tree {row['tree_id']:3d}: {row['before']:,} → {row['after']:,} "
                f"pts (-{row['pct_removed']:.0f}%)"
            )
        print(
            f"  Summary: {n_before:,} → {n_after:,} "
            f"({n_before - n_after:,} non-vertical points removed)"
        )
        print(_format_profile_summary(profile))

    return StemCleaningResult(
        stem_mask=trunk_mask,
        n_points_before=n_before,
        n_points_after=n_after,
        n_points_removed=n_before - n_after,
        per_tree_stats=per_tree,
        mode_used="global",
        n_trees_processed=len(unique_trees),
        n_trees_skipped=0,
        n_points_processed_verticality=len(trunk_pts),
        used_global_fallback=False,
        profile=profile,
    )


def _process_single_tree(
    tid: int,
    tree_indices: np.ndarray,
    tree_pts: np.ndarray,
    verticality_scale: float,
    vox_xy: float,
    vox_z: float,
    verticality_threshold: float,
    early_exit: bool = False,
    early_exit_coarse_res: float = 0.08,
    early_exit_margin: float = 0.1,
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """Process one tree: compute verticality, threshold, return removal indices.

    This helper is used by both serial and parallel paths in
    ``suspicious_only`` mode.  It is intentionally a module-level function
    so that it works with concurrent.futures (must be picklable for
    ProcessPoolExecutor, though we default to ThreadPoolExecutor).
    """
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    if early_exit:
        keep_mask, _ = compute_verticality_mask_early_exit(
            tree_pts,
            threshold=verticality_threshold,
            scale=verticality_scale,
            voxel_resolution_xy=vox_xy,
            voxel_resolution_z=vox_z,
            coarse_resolution=early_exit_coarse_res,
            margin=early_exit_margin,
        )
    else:
        vert = compute_verticality(
            tree_pts,
            scale=verticality_scale,
            voxel_resolution_xy=vox_xy,
            voxel_resolution_z=vox_z,
        )
        keep_mask = vert >= verticality_threshold
    t_vert = time.perf_counter() - t0

    t0 = time.perf_counter()
    remove_indices = tree_indices[~keep_mask]
    t_filter = time.perf_counter() - t0

    timing = {
        "tree_id": tid,
        "n_points": len(tree_pts),
        "verticality_seconds": t_vert,
        "filtering_seconds": t_filter,
        "mask_update_seconds": 0.0,              # filled by caller
        "tree_total_seconds": time.perf_counter() - t_start,
    }
    return tid, remove_indices, timing


def _process_trees_parallel(
    tree_work: List[Tuple[int, np.ndarray, np.ndarray]],
    verticality_scale: float,
    vox_xy: float,
    vox_z: float,
    verticality_threshold: float,
    max_workers: Optional[int],
    early_exit: bool = False,
    early_exit_coarse_res: float = 0.08,
    early_exit_margin: float = 0.1,
) -> List[Tuple[int, np.ndarray, Dict[str, Any]]]:
    """Run per-tree verticality in parallel using ThreadPoolExecutor.

    ThreadPoolExecutor is chosen over ProcessPoolExecutor because pgeof
    (C++ backend) releases the GIL, giving true parallelism without the
    cost of serialising large numpy arrays across process boundaries.
    """
    results: List[Tuple[int, np.ndarray, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_tree, tid, tree_indices, tree_pts,
                verticality_scale, vox_xy, vox_z, verticality_threshold,
                early_exit, early_exit_coarse_res, early_exit_margin,
            ): tid
            for tid, tree_indices, tree_pts in tree_work
        }
        for future in futures:
            results.append(future.result())
    # Sort by tree_id so output order is deterministic
    results.sort(key=lambda r: r[0])
    return results


def clean_stems(
    xyz: np.ndarray,
    trunk_result: TrunkExtractionResult,
    config: StemCleaningConfig,
    verbose: bool = True,
) -> StemCleaningResult:
    """
    Clean trunk points with a second verticality pass.

    In `global` mode the second pass runs on all trunk points.
    In `suspicious_only` mode it runs only on trees marked as suspicious,
    with an automatic fallback to the global path if the plot is mostly
    suspicious anyway.

    When verbose=True, a profiling breakdown is printed after the summary.
    The profile dict is also stored on StemCleaningResult.profile.
    """
    if config.mode == "global":
        return _clean_stems_global(xyz, trunk_result, config, verbose=verbose)

    if config.mode != "suspicious_only":
        raise ValueError(f"Unsupported stem cleaning mode: {config.mode}")

    t_total_start = time.perf_counter()

    # --- input preparation ---
    t0 = time.perf_counter()
    trunk_mask_before = trunk_result.trunk_mask.copy()
    tree_ids = trunk_result.tree_ids
    n_before = int(trunk_mask_before.sum())
    t_input_prep = time.perf_counter() - t0

    # --- suspicious scoring ---
    t0 = time.perf_counter()
    suspicious_ids, suspicious_points, total_trunk_points, unique_trees = _score_suspicious_trees(
        trunk_mask_before,
        tree_ids,
        trunk_result.tree_axes,
        config,
    )

    tree_ratio = len(suspicious_ids) / max(len(unique_trees), 1)
    point_ratio = suspicious_points / max(total_trunk_points, 1)
    use_global_fallback = (
        point_ratio > config.global_fallback_point_ratio
        or tree_ratio > config.global_fallback_tree_ratio
    )
    t_suspicious = time.perf_counter() - t0

    if use_global_fallback:
        if verbose:
            print("\nStem cleaning (2nd verticality pass):")
            print("  Mode request: suspicious_only")
            print(
                f"  Fallback to global: suspicious trees={len(suspicious_ids)}/{len(unique_trees)}, "
                f"suspicious points={suspicious_points:,}/{total_trunk_points:,}"
            )
        result = _clean_stems_global(xyz, trunk_result, config, verbose=verbose)
        # Merge suspicious-scoring time into the profile from _clean_stems_global
        if result.profile is not None:
            result.profile["suspicious_scoring_seconds"] = t_suspicious
        return replace(result, used_global_fallback=True)

    trunk_mask_after = trunk_mask_before.copy()
    use_parallel = config.parallel and len(suspicious_ids) > 1

    if verbose:
        print("\nStem cleaning (2nd verticality pass):")
        print(f"  Mode: suspicious_only{' (parallel)' if use_parallel else ''}")
        print(f"  Input: {n_before:,} trunk points")
        print(f"  Verticality threshold: {config.verticality_threshold}")
        print(f"  Scale: {config.verticality_scale}m")
        print(f"  Suspicious trees: {len(suspicious_ids)}/{len(unique_trees)}")
        print(f"  Suspicious points: {suspicious_points:,}/{total_trunk_points:,}")

    # Resolve voxel resolutions once
    vox_xy = config.voxel_resolution_xy_2nd or config.voxel_resolution_xy
    vox_z = config.voxel_resolution_z_2nd or config.voxel_resolution_z

    # Pre-compute per-tree indices (needed by both serial and parallel paths)
    tree_work: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for tid in suspicious_ids:
        tree_mask = trunk_mask_before & (tree_ids == tid)
        tree_indices = np.where(tree_mask)[0]
        tree_pts = xyz[tree_indices]
        if len(tree_pts) > 0:
            tree_work.append((tid, tree_indices, tree_pts))

    processed_points = 0
    t_vert_total = 0.0
    t_filter_total = 0.0
    t_mask_total = 0.0
    per_tree_timings: List[Dict[str, Any]] = []

    if use_parallel:
        # --- parallel path: ThreadPoolExecutor ---
        # pgeof (C++) releases the GIL → threads achieve true parallelism.
        tree_results = _process_trees_parallel(
            tree_work, config.verticality_scale, vox_xy, vox_z,
            config.verticality_threshold, config.parallel_max_workers,
            config.voxel_early_exit, config.voxel_early_exit_coarse_resolution,
            config.voxel_early_exit_margin,
        )
        for tid, remove_idx, timing in tree_results:
            processed_points += timing["n_points"]
            t_vert_total += timing["verticality_seconds"]
            t_filter_total += timing["filtering_seconds"]
            per_tree_timings.append(timing)
            # Mask update is always serial (trivial cost)
            t0 = time.perf_counter()
            trunk_mask_after[remove_idx] = False
            t_mask_tree = time.perf_counter() - t0
            t_mask_total += t_mask_tree
            per_tree_timings[-1]["mask_update_seconds"] = t_mask_tree
            per_tree_timings[-1]["tree_total_seconds"] += t_mask_tree
    else:
        # --- serial path ---
        for tid, tree_indices, tree_pts in tree_work:
            if verbose:
                print(f"  Computing verticality on tree {tid} ({len(tree_pts):,} pts)...")
            _, remove_indices, timing = _process_single_tree(
                tid, tree_indices, tree_pts,
                config.verticality_scale, vox_xy, vox_z,
                config.verticality_threshold,
                config.voxel_early_exit,
                config.voxel_early_exit_coarse_resolution,
                config.voxel_early_exit_margin,
            )
            processed_points += timing["n_points"]
            t_vert_total += timing["verticality_seconds"]
            t_filter_total += timing["filtering_seconds"]

            t0 = time.perf_counter()
            trunk_mask_after[remove_indices] = False
            t_mask_tree = time.perf_counter() - t0
            t_mask_total += t_mask_tree

            timing["mask_update_seconds"] = t_mask_tree
            timing["tree_total_seconds"] += t_mask_tree
            per_tree_timings.append(timing)

    # --- stats aggregation ---
    t0 = time.perf_counter()
    n_after = int(trunk_mask_after.sum())
    per_tree = _build_per_tree_stats(tree_ids, trunk_mask_before, trunk_mask_after, unique_trees)
    t_stats = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start

    profile = {
        "total_seconds": t_total,
        "input_preparation_seconds": t_input_prep,
        "suspicious_scoring_seconds": t_suspicious,
        "verticality_computation_seconds": t_vert_total,
        "threshold_filtering_seconds": t_filter_total,
        "mask_update_seconds": t_mask_total,
        "stats_aggregation_seconds": t_stats,
        "n_points_verticality": processed_points,
        "n_trees_suspicious": len(suspicious_ids),
        "n_trees_skipped": len(unique_trees) - len(suspicious_ids),
        "per_tree_timings": per_tree_timings,
        "parallel": use_parallel,
    }

    if verbose:
        for row in per_tree:
            print(
                f"    Tree {row['tree_id']:3d}: {row['before']:,} -> {row['after']:,} "
                f"pts (-{row['pct_removed']:.0f}%)"
            )
        print(
            f"  Summary: {n_before:,} -> {n_after:,} "
            f"({n_before - n_after:,} non-vertical points removed)"
        )
        print(_format_profile_summary(profile))

    return StemCleaningResult(
        stem_mask=trunk_mask_after,
        n_points_before=n_before,
        n_points_after=n_after,
        n_points_removed=n_before - n_after,
        per_tree_stats=per_tree,
        mode_used="suspicious_only",
        n_trees_processed=len(suspicious_ids),
        n_trees_skipped=len(unique_trees) - len(suspicious_ids),
        n_points_processed_verticality=processed_points,
        used_global_fallback=False,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Circle fitting (least squares, like 3DFin/dendromatics)
# ---------------------------------------------------------------------------

def _fit_circle_ls(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a circle to 2D points using least-squares (scipy.optimize.leastsq).
    Returns (centre_xy, mean_radius).
    """
    def _calc_R(X, Y, xc, yc):
        return np.sqrt((X - xc)**2 + (Y - yc)**2)

    def _residuals(c, X, Y):
        Ri = _calc_R(X, Y, *c)
        return Ri - Ri.mean()

    x0 = np.array([X.mean(), Y.mean()])
    centre, _ = optimize.leastsq(_residuals, x0, args=(X, Y), maxfev=2000)
    radius = _calc_R(X, Y, *centre).mean()
    return centre, float(radius)


def _sector_occupancy(
    X: np.ndarray, Y: np.ndarray,
    xc: float, yc: float, R: float,
    n_sectors: int, min_sectors: int, width: float,
) -> Tuple[float, bool]:
    """
    Check sector occupancy around the fitted circle.
    Returns (pct_occupied, is_valid).
    """
    X_red = X - xc
    Y_red = Y - yc
    radial = np.sqrt(X_red**2 + Y_red**2)
    angular = np.arctan2(X_red, Y_red)

    # Points near the circle
    near = (radial > (R - width)) & (radial < (R + width))
    if near.sum() == 0:
        return 0.0, False

    sectors = np.floor(angular[near] / (2 * np.pi / n_sectors))
    n_occupied = len(np.unique(sectors))
    pct = n_occupied * 100 / n_sectors
    return pct, n_occupied >= min_sectors


def _inner_circle_count(
    X: np.ndarray, Y: np.ndarray,
    xc: float, yc: float, R: float, ratio: float,
) -> int:
    """Count points inside inner circle (radius * ratio)."""
    dist = np.sqrt((X - xc)**2 + (Y - yc)**2)
    return int(np.sum(dist < R * ratio))


def _point_clustering_largest(X: np.ndarray, Y: np.ndarray, eps: float):
    """DBSCAN on XY, return points from largest cluster."""
    from sklearn.cluster import DBSCAN
    xy = np.column_stack([X, Y])
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(xy)
    if labels.max() < 0:
        return X, Y
    # Find largest cluster
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest = unique[np.argmax(counts)]
    mask = labels == largest
    return X[mask], Y[mask]


def _fit_circle_check(
    X: np.ndarray, Y: np.ndarray,
    config: StemCleaningConfig,
    is_retry: bool = False,
) -> Tuple[float, float, float, int, float]:
    """
    Fit circle with quality checks (like 3DFin's fit_circle_check).
    Returns (xc, yc, R, check_status, sector_pct).

    check_status: 0 = valid, 1 = checked (retry), 2 = not enough points
    """
    if len(X) < config.min_points_section:
        return 0.0, 0.0, 0.0, 2, 0.0

    centre, R = _fit_circle_ls(X, Y)
    xc, yc = centre

    n_inner = _inner_circle_count(X, Y, xc, yc, R, config.inner_circle_ratio)
    sector_pct, sectors_ok = _sector_occupancy(
        X, Y, xc, yc, R,
        config.n_sectors, config.min_sectors, config.sector_width,
    )

    # Check if fit is good
    fit_bad = (
        n_inner > config.max_inner_points
        or R < config.r_min
        or R > config.r_max
        or not sectors_ok
    )

    if fit_bad and not is_retry:
        # Retry with largest cluster
        Xg, Yg = _point_clustering_largest(X, Y, config.cluster_eps)
        if len(Xg) >= config.min_points_section:
            return _fit_circle_check(Xg, Yg, config, is_retry=True)
        else:
            return 0.0, 0.0, 0.0, 1, 0.0

    return xc, yc, R, (1 if is_retry else 0), sector_pct


# ---------------------------------------------------------------------------
# Step 2: Compute sections (circle fitting per section, per tree)
# ---------------------------------------------------------------------------

def compute_stem_sections(
    xyz: np.ndarray,
    stem_mask: np.ndarray,
    tree_ids: np.ndarray,
    config: StemCleaningConfig,
    verbose: bool = True,
) -> SectionResult:
    """
    Compute stem sections with circle fitting for each tree.

    For each tree, slices the cleaned stem points into horizontal sections
    and fits circles using least-squares. Returns per-section centres,
    radii, and quality metrics.

    Args:
        xyz: (N, 3) full height-normalised point cloud.
        stem_mask: (N,) boolean mask — cleaned stem points.
        tree_ids: (N,) tree ID per point.
        config: Sectioning parameters.
        verbose: Print progress.

    Returns:
        SectionResult with per-tree, per-section circle fits.
    """
    sections = np.arange(
        config.minimum_height,
        config.maximum_height,
        config.section_len,
    )
    n_sections = len(sections)

    # Get unique tree IDs from stem mask
    unique_trees = sorted(set(tree_ids[stem_mask]) - {-1})
    n_trees = len(unique_trees)

    if verbose:
        print(f"\nStem sectioning:")
        print(f"  Trees: {n_trees}")
        print(f"  Sections: {n_sections} "
              f"({config.minimum_height}m → {config.maximum_height}m, "
              f"step={config.section_len}m, width=±{config.section_wid}m)")

    # Allocate output arrays
    X_c = np.zeros((n_trees, n_sections))
    Y_c = np.zeros((n_trees, n_sections))
    R = np.zeros((n_trees, n_sections))
    check = np.zeros((n_trees, n_sections))
    sector_pct = np.zeros((n_trees, n_sections))

    for i, tid in enumerate(unique_trees):
        # Get stem points for this tree
        tree_mask = stem_mask & (tree_ids == tid)
        tree_pts = xyz[tree_mask]

        n_valid = 0
        radii_valid = []

        for j, sh in enumerate(sections):
            # Select points in this section slice
            z_low = sh - config.section_wid
            z_high = sh + config.section_wid
            sec_mask = (tree_pts[:, 2] >= z_low) & (tree_pts[:, 2] < z_high)

            sec_X = tree_pts[sec_mask, 0]
            sec_Y = tree_pts[sec_mask, 1]

            # Fit circle with checks
            xc, yc, r, chk, spct = _fit_circle_check(sec_X, sec_Y, config)

            X_c[i, j] = xc
            Y_c[i, j] = yc
            R[i, j] = r
            check[i, j] = chk
            sector_pct[i, j] = spct

            if r > 0:
                n_valid += 1
                radii_valid.append(r)

        if verbose:
            mean_r = np.mean(radii_valid) if radii_valid else 0
            mean_d = mean_r * 2
            print(f"    Tree {tid:3d}: {n_valid}/{n_sections} valid sections, "
                  f"mean_diam={mean_d:.3f}m")

    return SectionResult(
        X_c=X_c,
        Y_c=Y_c,
        R=R,
        check=check,
        sector_pct=sector_pct,
        sections=sections,
        tree_ids=unique_trees,
        config=config,
    )


# ---------------------------------------------------------------------------
# Step 2 (variant): RANSAC ellipse sections (Mejora 1, Phase 1C — EF.1)
# ---------------------------------------------------------------------------

def _stem_config_to_ellipse_config(config: StemCleaningConfig) -> EllipseFitConfig:
    """Map the relevant fields of ``StemCleaningConfig`` (used by the
    circle-fit path) onto an :class:`EllipseFitConfig` (used by the
    ellipse-fit path). RANSAC and aspect-ratio knobs fall back to the
    sensible defaults declared on ``EllipseFitConfig`` itself.
    """
    return EllipseFitConfig(
        min_points_section=config.min_points_section,
        r_min=config.r_min,
        r_max=config.r_max,
        inner_ratio=config.inner_circle_ratio,
        max_inner_points=config.max_inner_points,
        n_sectors=config.n_sectors,
        min_sectors=config.min_sectors,
        sector_width=config.sector_width,
        cluster_eps=config.cluster_eps,
        # ransac_n_iters, ransac_tau_sampson, min_inlier_fraction,
        # min_aspect_ratio: kept at EllipseFitConfig defaults — they have
        # no equivalent in StemCleaningConfig and changing them requires
        # an explicit override at the caller.
    )


def compute_stem_sections_ellipse(
    xyz: np.ndarray,
    stem_mask: np.ndarray,
    tree_ids: np.ndarray,
    config: StemCleaningConfig,
    verbose: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> SectionResult:
    """RANSAC-ellipse drop-in alternative to :func:`compute_stem_sections`.

    Same loop structure (per-tree, per-section horizontal slice), same
    output dataclass, but the per-section fit is the ellipse pipeline
    from ``src.core.ellipse_fitting``:

      1. Fitzgibbon-Halíř-Flusser algebraic hypothesis (EL.1) inside a
         RANSAC loop with Sampson scoring (EL.2, EL.3).
      2. Levenberg-Marquardt refit on the orthogonal distance over the
         consensus set (EL.4).
      3. Quality checks (radius range, inner-empty, sector occupancy,
         aspect ratio, inlier fraction) and a DBSCAN-retry path identical
         to the circle wrapper (EL.5 — ``_fit_ellipse_check``).

    The ``SectionResult.R`` field stores the **equivalent radius**
    ``√(a · b)`` of each section's ellipse so that
    :func:`build_centerline_from_sections` and any downstream code that
    consumes ``R`` continues to work unchanged. The full geometric
    description ``(a, b, theta)`` per section is not yet stored on
    ``SectionResult`` — that extension is sub-fase EF.2.

    Parameters
    ----------
    xyz, stem_mask, tree_ids, config : same as :func:`compute_stem_sections`.
    verbose : bool, default True
        Print per-tree progress to stdout.
    rng : np.random.Generator, optional
        Random number generator forwarded to the RANSAC loop. Pass a
        seeded generator for reproducibility (recommended for the
        notebook gate comparisons).

    Returns
    -------
    SectionResult
        Same dataclass as the circle-fit path; ``R = √(a·b)`` per
        section. Sections with no valid fit have ``R = 0``.
    """
    ellipse_cfg = _stem_config_to_ellipse_config(config)

    sections = np.arange(
        config.minimum_height,
        config.maximum_height,
        config.section_len,
    )
    n_sections = len(sections)

    unique_trees = sorted(set(tree_ids[stem_mask]) - {-1})
    n_trees = len(unique_trees)

    if verbose:
        print(f"\nStem sectioning (RANSAC ellipse, Mejora 1 Phase 1C — EF.1):")
        print(f"  Trees: {n_trees}")
        print(f"  Sections: {n_sections} "
              f"({config.minimum_height}m → {config.maximum_height}m, "
              f"step={config.section_len}m, width=±{config.section_wid}m)")
        print(f"  Per-section fit: RANSAC ellipse "
              f"(n_iters={ellipse_cfg.ransac_n_iters}, "
              f"τ_Sampson={ellipse_cfg.ransac_tau_sampson}m)")

    X_c = np.zeros((n_trees, n_sections))
    Y_c = np.zeros((n_trees, n_sections))
    R = np.zeros((n_trees, n_sections))
    check = np.zeros((n_trees, n_sections))
    sector_pct = np.zeros((n_trees, n_sections))

    for i, tid in enumerate(unique_trees):
        tree_mask = stem_mask & (tree_ids == tid)
        tree_pts = xyz[tree_mask]

        n_valid = 0
        radii_valid: List[float] = []

        for j, sh in enumerate(sections):
            z_low = sh - config.section_wid
            z_high = sh + config.section_wid
            sec_mask = (tree_pts[:, 2] >= z_low) & (tree_pts[:, 2] < z_high)

            sec_X = tree_pts[sec_mask, 0]
            sec_Y = tree_pts[sec_mask, 1]

            xc, yc, a, b, _theta, chk, spct = _fit_ellipse_check(
                sec_X, sec_Y, ellipse_cfg, rng=rng,
            )

            r_equiv = float(np.sqrt(a * b)) if (a > 0.0 and b > 0.0) else 0.0

            X_c[i, j] = xc
            Y_c[i, j] = yc
            R[i, j] = r_equiv
            check[i, j] = chk
            sector_pct[i, j] = spct

            if r_equiv > 0.0:
                n_valid += 1
                radii_valid.append(r_equiv)

        if verbose:
            mean_r = float(np.mean(radii_valid)) if radii_valid else 0.0
            mean_d = mean_r * 2.0
            print(f"    Tree {tid:3d}: {n_valid}/{n_sections} valid sections, "
                  f"mean_diam={mean_d:.3f}m")

    return SectionResult(
        X_c=X_c,
        Y_c=Y_c,
        R=R,
        check=check,
        sector_pct=sector_pct,
        sections=sections,
        tree_ids=unique_trees,
        config=config,
    )


# ---------------------------------------------------------------------------
# Step 2b: Centerline from section circle centres (Improvement 1, Phase 1A.REAL)
# ---------------------------------------------------------------------------

def build_centerline_from_sections(
    section_result: SectionResult,
    min_valid_sections: int = 3,
    smooth_window: int = 3,
) -> List[Optional[np.ndarray]]:
    """
    Build a piecewise centerline per tree from section circle centres.

    For each tree, concatenates the `(X_c, Y_c, z_section)` triplets of the
    sections whose circle fit succeeded into a polyline sorted ascending by Z,
    then applies a moving-median filter across adjacent sections to suppress
    per-fit jitter (improvement F1).

    A section is considered valid when its fitted radius is strictly positive
    (``R > 0``) — this is the same criterion ``compute_stem_sections`` uses
    internally to report "N valid sections" in its verbose output. The
    ``check`` field is intentionally NOT used because its value ``1`` is
    ambiguous (returned both for valid retry fits and for retry failures);
    ``R > 0`` distinguishes the two cases unambiguously.

    Smoothing (F1)
    --------------
    Independent per-section LS circle fits produce small XY jitter (a few cm)
    even on geometrically straight trunks, because each fit is solved in
    isolation with no awareness of the adjacent sections. A moving median of
    odd width ``smooth_window`` (default 3) is applied per coordinate
    (``X_c`` and ``Y_c``; ``z_section`` is left untouched) with
    ``mode='nearest'`` for edge replication. ``mode='nearest'`` is chosen
    deliberately: it makes the filter the identity transform on monotonic
    series (a leaning trunk stays leaning), and only flattens local spikes
    surrounded by quieter neighbours — exactly the failure mode it is
    meant to suppress. ``smooth_window=1`` disables smoothing entirely.

    Parameters
    ----------
    section_result : SectionResult
        Output of :func:`compute_stem_sections`. Provides ``X_c``, ``Y_c``,
        ``R`` arrays of shape ``(n_trees, n_sections)``, the ``sections``
        height grid of shape ``(n_sections,)``, and the ``tree_ids`` list.
    min_valid_sections : int, default 3
        Minimum number of valid sections a tree must have to produce a
        centerline. Trees below this threshold get ``None`` (treated as
        no-data downstream).
    smooth_window : int, default 3
        Odd-sized moving-median window applied to ``X_c`` and ``Y_c`` along
        the section dimension. Must be ``>= 1``. ``1`` disables smoothing.

    Returns
    -------
    list of (ndarray or None)
        One entry per tree, indexed positionally to match
        ``section_result.tree_ids``. Each entry is either:
          - an ``(K, 3) float64`` array of control points
            ``(X_c, Y_c, z_section)`` sorted by Z, where ``K`` is the number
            of valid sections for that tree (``K >= min_valid_sections``); or
          - ``None`` if the tree has fewer than ``min_valid_sections`` valid
            sections.
    """
    if min_valid_sections < 2:
        raise ValueError(
            f"min_valid_sections must be >= 2 to form a polyline; got {min_valid_sections}"
        )
    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError(
            f"smooth_window must be a positive odd integer; got {smooth_window}"
        )

    sections_z = np.asarray(section_result.sections, dtype=np.float64)
    n_trees = len(section_result.tree_ids)

    centerlines: List[Optional[np.ndarray]] = []
    for i in range(n_trees):
        valid_mask = section_result.R[i] > 0
        n_valid = int(valid_mask.sum())
        if n_valid < min_valid_sections:
            centerlines.append(None)
            continue

        xc = section_result.X_c[i, valid_mask].astype(np.float64, copy=False)
        yc = section_result.Y_c[i, valid_mask].astype(np.float64, copy=False)
        zc = sections_z[valid_mask]

        # Sections come from np.arange (monotonic increasing) but sort
        # defensively in case a future change reorders them.
        order = np.argsort(zc)
        xc, yc, zc = xc[order], yc[order], zc[order]

        if smooth_window > 1:
            xc = median_filter(xc, size=smooth_window, mode="nearest")
            yc = median_filter(yc, size=smooth_window, mode="nearest")

        polyline = np.column_stack([xc, yc, zc])
        centerlines.append(polyline)

    return centerlines


# ---------------------------------------------------------------------------
# Step 3: Tree-level filters (Height and Edge)
# ---------------------------------------------------------------------------

@dataclass
class TreeFilterConfig:
    """
    Configuration for final tree-level filtering (Stage B & D).
    
    Parameters:
        plot_center_x: X coordinate of plot centre.
        plot_center_y: Y coordinate of plot centre.
        max_distance_from_center: Maximum allowed distance from plot centre to 
                                  tree centroid (metres) before it's considered an edge tree.
    """
    plot_center_x: float = 0.0
    plot_center_y: float = 0.0
    max_distance_from_center: float = 15.5  # Slightly less than plot radius to catch edge intersections


@dataclass
class TreeFilterResult:
    """Result of final tree filtering."""
    stem_mask: np.ndarray         # (N,) bool — updated stem mask with remaining valid trees
    tree_ids: np.ndarray          # (N,) int — updated tree IDs (-1 for filtered)
    n_trees_before: int
    n_trees_after: int
    n_trees_removed: int
    trees_removed_edge: List[int]


def filter_trees(
    xyz: np.ndarray,
    stem_mask: np.ndarray,
    tree_ids: np.ndarray,
    config: TreeFilterConfig,
    verbose: bool = True,
) -> TreeFilterResult:
    """
    Filter trees by distance from plot centre.
    
    Trees that are too far from the
    centre (cut edge trees) are completely removed from the stem_mask and tree_ids.

    Args:
        xyz: (N, 3) full height-normalised point cloud.
        stem_mask: (N,) boolean mask of cleaned stem points.
        tree_ids: (N,) tree ID per point.
        config: Filter parameters.
        verbose: Print progress.

    Returns:
        TreeFilterResult with updated masks.
    """
    updated_stem_mask = stem_mask.copy()
    updated_tree_ids = tree_ids.copy()
    
    unique_trees = sorted(set(tree_ids[stem_mask]) - {-1})
    n_before = len(unique_trees)
    
    if verbose:
        print(f"\nTree-level filtering:")
        print(f"  Input: {n_before} trees")
        print(f"  Max distance from centre ({config.plot_center_x:.2f}, {config.plot_center_y:.2f}): {config.max_distance_from_center}m")

    removed_edge = []

    
    for tid in unique_trees:
        tree_pts_mask = stem_mask & (tree_ids == tid)
        x_vals = xyz[tree_pts_mask, 0]
        y_vals = xyz[tree_pts_mask, 1]
        
        # 1. Edge filter
        centroid_x = x_vals.mean()
        centroid_y = y_vals.mean()
        distance = np.sqrt((centroid_x - config.plot_center_x)**2 + (centroid_y - config.plot_center_y)**2)
        
        if distance > config.max_distance_from_center:
            removed_edge.append(tid)
            updated_stem_mask[tree_pts_mask] = False
            updated_tree_ids[tree_pts_mask] = -1
            if verbose:
                print(f"    Tree {tid:3d}: ✗ REJECTED (edge tree: dist={distance:.1f}m > {config.max_distance_from_center}m)")
            continue
            
        if verbose:
            print(f"    Tree {tid:3d}: ✓ valid (dist={distance:.1f}m)")

    n_after = len(unique_trees) - len(removed_edge)
    
    if verbose:
        print(f"  Summary: {n_before} → {n_after} trees ({len(removed_edge)} edge)")

    return TreeFilterResult(
        stem_mask=updated_stem_mask,
        tree_ids=updated_tree_ids,
        n_trees_before=n_before,
        n_trees_after=n_after,
        n_trees_removed=n_before - n_after,
        trees_removed_edge=removed_edge,
    )
