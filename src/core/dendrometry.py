"""
Dendrometric metrics — per-tree DBH, total height, ovality at DBH.

Downstream consumer of ``TrunkExtractionResult`` (assigns ``tree_id`` to
points) and ``SectionResult`` (per-section ellipse / circle fits). Sits
strictly outside the fitting pipeline so it can evolve independently of
the upstream geometric work.

This is Phase 1C / EF.4 of Mejora 1:
  * DBH (diameter at breast height, 1.30 m): derived from the per-section
    ellipse semi-axes at the section nearest to 1.30 m. Uses the equivalent
    diameter ``2 · √(a · b)`` so circle and ellipse fits give consistent
    DBH on truly circular cross-sections.
  * Total height: ``z_max − z_min`` over all points belonging to the tree
    (regardless of whether they passed the section fits — the height is
    a cloud-extent metric, not a section metric).
  * Ovality at DBH: ``a / b`` with ``a ≥ b`` from the DBH section. The
    HQP forestry criterion classifies a trunk as "oval" when this ratio
    exceeds 1.2.

Caveats
-------
- Total height is biased downward on inclined/curved stems when the
  upstream tree-ID assignment uses a straight cylinder (Phase 1A.REAL).
  Points whose XY drifts outside that cylinder are unassigned and don't
  contribute to ``z_max``. Phase 1B (curved-cylinder / tracking assignment)
  resolves this; the algorithm here is unchanged, only its inputs improve.
- DBH and ovality require the section nearest to 1.30 m to have a
  successful fit (``R > 0`` and ``a, b > 0``). If the fit failed at that
  height — too few points, sector occupancy below threshold, etc. —
  the metric is marked invalid for that tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.core.trunk_extraction import TrunkExtractionResult
from src.core.trunk_validation import SectionResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TreeMetrics:
    """Per-tree dendrometric metrics (EF.4)."""
    tree_id: int

    # --- DBH and ovality at breast height (1.30 m) ---
    valid_at_dbh: bool            # True if the section at DBH height has a valid fit
    dbh_section_height: float     # actual height of the section used (m); 0 if invalid
    dbh: float                    # equivalent diameter 2·√(a·b) at DBH section (m)
    dbh_major: float              # 2 · semi-major axis at DBH (m)
    dbh_minor: float              # 2 · semi-minor axis at DBH (m)
    ovality_at_dbh: float         # a / b at DBH (≥ 1.0); 0.0 if invalid
    is_oval_at_dbh: bool          # ovality_at_dbh > ovality_threshold (HQP criterion)

    # --- Cloud-extent metrics ---
    z_base: float                 # z_min over the tree_id's points (m)
    z_top: float                  # z_max over the tree_id's points (m)
    height_total: float           # z_top − z_base (m); 0 if the tree has no points

    # --- Provenance ---
    n_points: int                 # number of points carrying this tree_id
    n_valid_sections: int         # number of sections with R > 0 for this tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_section_index_at_height(
    sections: np.ndarray,
    target_z: float,
    max_distance: float,
) -> int:
    """Return the index of the section whose height is closest to
    ``target_z``, provided the distance is within ``max_distance``.
    Returns ``-1`` if no section is close enough or the array is empty.
    """
    if sections.size == 0:
        return -1
    diffs = np.abs(sections - target_z)
    idx = int(np.argmin(diffs))
    if diffs[idx] > max_distance:
        return -1
    return idx


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_tree_metrics(
    xyz: np.ndarray,
    trunk_result: TrunkExtractionResult,
    section_result: SectionResult,
    dbh_height: float = 1.30,
    dbh_max_distance: float = 0.15,
    ovality_threshold: float = 1.20,
) -> List[TreeMetrics]:
    """Compute dendrometric metrics for every tree present in
    ``section_result.tree_ids``.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        The full height-normalised point cloud. Used only to read the
        z-extent of each tree (no geometric processing).
    trunk_result : TrunkExtractionResult
        Provides ``tree_ids`` (per-point tree assignment from the
        upstream pipeline). Used to filter the cloud by tree.
    section_result : SectionResult
        Per-section fits. Must have ``a`` and ``b`` populated (EF.2);
        a result from either the circle path or the ellipse path works
        — the circle path stores ``a = b = R`` so the API is uniform.
    dbh_height : float, default 1.30
        Target height for DBH measurement, in metres.
    dbh_max_distance : float, default 0.15
        Maximum allowed distance (m) between ``dbh_height`` and the
        nearest section. Beyond this, the DBH is marked invalid.
        Default 0.15 m covers up to a 0.30 m sectioning step.
    ovality_threshold : float, default 1.20
        HQP criterion for classifying a stem as "oval" at DBH:
        ``a / b > ovality_threshold``.

    Returns
    -------
    list of TreeMetrics
        One entry per tree in ``section_result.tree_ids``, in the same
        order. Trees with no points (defensive case, should not happen
        in practice) get an entry with ``n_points = 0``,
        ``height_total = 0`` and ``valid_at_dbh = False``.
    """
    sections = np.asarray(section_result.sections, dtype=np.float64)
    tree_ids_array = np.asarray(trunk_result.tree_ids)
    xyz_z = xyz[:, 2]

    has_ellipse_fields = (
        section_result.a is not None
        and section_result.b is not None
    )

    metrics: List[TreeMetrics] = []
    for i, tid in enumerate(section_result.tree_ids):
        tid_int = int(tid)

        # --- Cloud-extent metrics ---
        tree_point_mask = tree_ids_array == tid_int
        n_points = int(tree_point_mask.sum())
        if n_points > 0:
            tree_z = xyz_z[tree_point_mask]
            z_base = float(tree_z.min())
            z_top = float(tree_z.max())
            height_total = z_top - z_base
        else:
            z_base = 0.0
            z_top = 0.0
            height_total = 0.0

        # --- Section count ---
        r_row = section_result.R[i]
        n_valid_sections = int((r_row > 0.0).sum())

        # --- DBH section lookup ---
        idx = _find_section_index_at_height(
            sections, dbh_height, dbh_max_distance,
        )
        valid_at_dbh = False
        dbh_section_height = 0.0
        dbh = 0.0
        dbh_major = 0.0
        dbh_minor = 0.0
        ovality_at_dbh = 0.0
        is_oval_at_dbh = False

        if idx >= 0 and section_result.R[i, idx] > 0.0 and has_ellipse_fields:
            a_val = float(section_result.a[i, idx])  # type: ignore[index]
            b_val = float(section_result.b[i, idx])  # type: ignore[index]
            if a_val > 0.0 and b_val > 0.0:
                # SectionResult convention: a ≥ b on valid sections.
                valid_at_dbh = True
                dbh_section_height = float(sections[idx])
                dbh_major = 2.0 * a_val
                dbh_minor = 2.0 * b_val
                dbh = 2.0 * float(np.sqrt(a_val * b_val))
                ovality_at_dbh = a_val / b_val
                is_oval_at_dbh = ovality_at_dbh > ovality_threshold

        metrics.append(TreeMetrics(
            tree_id=tid_int,
            valid_at_dbh=valid_at_dbh,
            dbh_section_height=dbh_section_height,
            dbh=dbh,
            dbh_major=dbh_major,
            dbh_minor=dbh_minor,
            ovality_at_dbh=ovality_at_dbh,
            is_oval_at_dbh=is_oval_at_dbh,
            z_base=z_base,
            z_top=z_top,
            height_total=height_total,
            n_points=n_points,
            n_valid_sections=n_valid_sections,
        ))

    return metrics


def tree_metrics_to_dataframe(metrics: List[TreeMetrics]):
    """Convert a list of :class:`TreeMetrics` to a pandas DataFrame for
    Excel / CSV export. Pandas is imported lazily so this module stays
    importable without it.
    """
    import pandas as pd

    rows = []
    for m in metrics:
        rows.append({
            "Tree_ID": m.tree_id,
            "N_points": m.n_points,
            "Z_base": round(m.z_base, 3),
            "Z_top": round(m.z_top, 3),
            "Height_total": round(m.height_total, 3),
            "Valid_at_DBH": m.valid_at_dbh,
            "DBH_section_height": round(m.dbh_section_height, 3),
            "DBH": round(m.dbh, 4),
            "DBH_major": round(m.dbh_major, 4),
            "DBH_minor": round(m.dbh_minor, 4),
            "Ovality_at_DBH": round(m.ovality_at_dbh, 4),
            "Is_oval_at_DBH": m.is_oval_at_dbh,
            "N_valid_sections": m.n_valid_sections,
        })
    return pd.DataFrame(rows)
