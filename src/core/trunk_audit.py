"""
Trunk audit helpers.

Builds a per-candidate audit table from the current trunk extraction pipeline
without changing any extraction or filtering decisions.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .trunk_extraction import TrunkExtractionResult
from .trunk_validation import StemCleaningResult, SectionResult


def _longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run of True values."""
    longest = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _axis_tilt_deg(direction: np.ndarray) -> float:
    """Angle between the axis direction and the vertical axis."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-12
    vertical_alignment = np.clip(abs(direction[2]), 0.0, 1.0)
    return float(np.degrees(np.arccos(vertical_alignment)))


def _valid_section_mask(section_result: SectionResult, row_idx: int) -> np.ndarray:
    """Sections accepted by the fitting routine, not merely retried."""
    return (section_result.check[row_idx] == 0) & (section_result.R[row_idx] > 0)


def _centerline_steps(section_result: SectionResult, row_idx: int) -> np.ndarray:
    """XY step lengths between consecutive valid section centres."""
    valid = _valid_section_mask(section_result, row_idx)
    if valid.sum() < 2:
        return np.empty(0, dtype=np.float64)

    centres = np.column_stack([section_result.X_c[row_idx, valid], section_result.Y_c[row_idx, valid]])
    return np.linalg.norm(np.diff(centres, axis=0), axis=1)


def build_trunk_audit_table(
    xyz: np.ndarray,
    trunk_result: TrunkExtractionResult,
    cleaning_result: StemCleaningResult,
    section_result: SectionResult,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> pd.DataFrame:
    """
    Build a diagnostic table with one row per detected trunk candidate.

    The table is intended for manual QA and threshold tuning. It does not
    apply any new filtering logic.
    """
    clean_stats: Dict[int, Dict[str, Any]] = {
        int(item["tree_id"]): item for item in cleaning_result.per_tree_stats
    }
    section_rows = {int(tree_id): idx for idx, tree_id in enumerate(section_result.tree_ids)}

    rows = []
    for axis in trunk_result.tree_axes:
        tree_id = int(axis["tree_id"])
        assigned_mask = trunk_result.tree_ids == tree_id
        trunk_mask = trunk_result.trunk_mask & assigned_mask

        clean_before = int(trunk_mask.sum())
        clean_after = clean_before
        clean_removed_pct = 0.0
        if tree_id in clean_stats:
            clean_before = int(clean_stats[tree_id]["before"])
            clean_after = int(clean_stats[tree_id]["after"])
            clean_removed_pct = float(clean_stats[tree_id]["pct_removed"])

        valid_sections_total = 0
        valid_sections_consecutive_max = 0
        mean_radius = np.nan
        radius_cv = np.nan
        centerline_step_mean = np.nan
        centerline_step_max = np.nan
        if tree_id in section_rows:
            sec_idx = section_rows[tree_id]
            valid_sections = _valid_section_mask(section_result, sec_idx)
            valid_radii = section_result.R[sec_idx, valid_sections]
            valid_sections_total = int(valid_sections.sum())
            valid_sections_consecutive_max = _longest_true_run(valid_sections)
            if valid_radii.size > 0:
                mean_radius = float(np.mean(valid_radii))
                radius_cv = float(np.std(valid_radii) / (mean_radius + 1e-12))
            steps = _centerline_steps(section_result, sec_idx)
            if steps.size > 0:
                centerline_step_mean = float(np.mean(steps))
                centerline_step_max = float(np.max(steps))

        centroid = np.asarray(axis["centroid"], dtype=np.float64)
        distance_to_center = float(np.linalg.norm(centroid[:2] - np.array([center_x, center_y])))
        stripe_diameter = float(axis.get("stripe_diameter", np.nan))
        seed_radius_ref = float(axis.get("seed_radius_ref", np.nan))
        seed_vs_half_diam = np.nan
        if np.isfinite(stripe_diameter) and stripe_diameter > 0 and np.isfinite(seed_radius_ref):
            seed_vs_half_diam = float(seed_radius_ref / max(stripe_diameter / 2.0, 1e-12))

        stripe_points = int(axis.get("stripe_points", axis.get("n_points", 0)))
        stripe_z_span = float(axis.get("stripe_z_span", axis["z_max"] - axis["z_min"]))
        growth_points_kept = int(axis.get("growth_points_kept", trunk_mask.sum()))
        growth_ratio = float(growth_points_kept / max(stripe_points, 1))
        stripe_density_z = float(stripe_points / max(stripe_z_span, 1e-12))

        rows.append({
            "tree_id": tree_id,
            "stripe_points": stripe_points,
            "stripe_diameter": stripe_diameter,
            "stripe_circularity": float(axis.get("stripe_circularity", np.nan)),
            "stripe_z_span": stripe_z_span,
            "axis_source": axis.get("axis_source", "pca"),
            "basal_anchor_found": bool(axis.get("basal_anchor_found", False)),
            "basal_anchor_applied": bool(axis.get("basal_anchor_applied", False)),
            "basal_anchor_candidates_total": int(axis.get("basal_anchor_candidates_total", 0)),
            "basal_anchor_candidates_validated": int(axis.get("basal_anchor_candidates_validated", 0)),
            "basal_anchor_x": float(axis.get("basal_anchor_x", np.nan)),
            "basal_anchor_y": float(axis.get("basal_anchor_y", np.nan)),
            "basal_anchor_z": float(axis.get("basal_anchor_z", np.nan)),
            "basal_anchor_radius": float(axis.get("basal_anchor_radius", np.nan)),
            "basal_anchor_circularity": float(axis.get("basal_anchor_circularity", np.nan)),
            "basal_anchor_sector_pct": float(axis.get("basal_anchor_sector_pct", np.nan)),
            "basal_anchor_inner_fraction": float(axis.get("basal_anchor_inner_fraction", np.nan)),
            "basal_anchor_support_slices": int(axis.get("basal_anchor_support_slices", 0)),
            "basal_anchor_center_offset": float(axis.get("basal_anchor_center_offset", np.nan)),
            "basal_anchor_model": axis.get("basal_anchor_model", "none"),
            "basal_anchor_arc_coverage": float(axis.get("basal_anchor_arc_coverage", np.nan)),
            "basal_anchor_fit_residual": float(axis.get("basal_anchor_fit_residual", np.nan)),
            "basal_anchor_track_center_std": float(axis.get("basal_anchor_track_center_std", np.nan)),
            "basal_anchor_track_radius_cv": float(axis.get("basal_anchor_track_radius_cv", np.nan)),
            "basal_anchor_track_id": int(axis.get("basal_anchor_track_id", -1)),
            "basal_anchor_track_score": float(axis.get("basal_anchor_track_score", np.nan)),
            "basal_anchor_match_passed": bool(axis.get("basal_anchor_match_passed", False)),
            "basal_anchor_match_score": float(axis.get("basal_anchor_match_score", np.nan)),
            "basal_anchor_match_xy_distance": float(axis.get("basal_anchor_match_xy_distance", np.nan)),
            "basal_anchor_match_radius_ratio": float(axis.get("basal_anchor_match_radius_ratio", np.nan)),
            "basal_anchor_match_tilt_deg": float(axis.get("basal_anchor_match_tilt_deg", np.nan)),
            "axis_direction_delta_deg": float(axis.get("axis_direction_delta_deg", 0.0)),
            "seed_selection_mode": axis.get("seed_selection_mode", ""),
            "seed_points": int(axis.get("seed_points", 0)),
            "seed_z_min": float(axis.get("seed_z_min", np.nan)),
            "seed_z_max": float(axis.get("seed_z_max", np.nan)),
            "seed_z_span": float(axis.get("seed_z_span", np.nan)),
            "seed_mean_circularity": float(axis.get("seed_mean_circularity", np.nan)),
            "seed_min_circularity": float(axis.get("seed_min_circularity", np.nan)),
            "seed_radius_cv": float(axis.get("seed_radius_cv", np.nan)),
            "seed_center_step_mean": float(axis.get("seed_center_step_mean", np.nan)),
            "seed_center_step_max": float(axis.get("seed_center_step_max", np.nan)),
            "seed_mean_dominant_fraction": float(axis.get("seed_mean_dominant_fraction", np.nan)),
            "seed_min_dominant_fraction": float(axis.get("seed_min_dominant_fraction", np.nan)),
            "seed_n_slices": int(axis.get("seed_n_slices", 0)),
            "seed_radius_ref": seed_radius_ref,
            "seed_vs_half_diam": seed_vs_half_diam,
            "plo_seed_mode": axis.get("plo_seed_mode", ""),
            "plo_seed_voxel_count": int(axis.get("plo_seed_voxel_count", 0)),
            "plo_seed_empty_neighbor_ratio": float(axis.get("plo_seed_empty_neighbor_ratio", np.nan)),
            "plo_seed_vertical_support": int(axis.get("plo_seed_vertical_support", 0)),
            "growth_slices_up": int(axis.get("growth_slices_up", 0)),
            "growth_slices_down": int(axis.get("growth_slices_down", 0)),
            "growth_points_kept": growth_points_kept,
            "growth_ratio": growth_ratio,
            "growth_stop_reason_up": axis.get("growth_stop_reason_up", ""),
            "growth_stop_reason_down": axis.get("growth_stop_reason_down", ""),
            "plo_growth_voxel_count": int(axis.get("plo_growth_voxel_count", 0)),
            "plo_growth_layers_up": int(axis.get("plo_growth_layers_up", axis.get("growth_slices_up", 0))),
            "plo_growth_layers_down": int(axis.get("plo_growth_layers_down", axis.get("growth_slices_down", 0))),
            "plo_growth_max_gap_hit": int(axis.get("plo_growth_max_gap_hit", 0)),
            "plo_growth_reached_ground_proxy": bool(axis.get("plo_growth_reached_ground_proxy", False)),
            "stripe_density_z": stripe_density_z,
            "axis_tilt_deg": _axis_tilt_deg(axis["direction"]),
            "assigned_points_total": int(assigned_mask.sum()),
            "trunk_points": int(trunk_mask.sum()),
            "clean_before": clean_before,
            "clean_after": clean_after,
            "clean_removed_pct": clean_removed_pct,
            "valid_sections_total": valid_sections_total,
            "valid_sections_consecutive_max": valid_sections_consecutive_max,
            "mean_radius": mean_radius,
            "radius_cv": radius_cv,
            "centerline_step_mean": centerline_step_mean,
            "centerline_step_max": centerline_step_max,
            "distance_to_center": distance_to_center,
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("tree_id").reset_index(drop=True)


def export_trunk_audit_table(df: pd.DataFrame, output_path: Path) -> Path:
    """Export an audit table to CSV or XLSX based on file suffix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False, sheet_name="Trunk Audit")
    else:
        raise ValueError(f"Unsupported audit export format: {output_path.suffix}")

    return output_path
