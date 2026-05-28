"""
Per-tree stem-extraction coverage metrics + Interpine HQP stem description.

Two related outputs produced by this module, intended to land as extra
sheets in ``tree_inventory.xlsx`` alongside the main per-tree summary
produced by :mod:`src.core.dendrometry`:

1. ``Coverage`` sheet — one row per tree, summarises how much of the
   trunk the curved-cylinder labelling actually captured from the
   point cloud. Used to compare against the SED10 threshold
   (RH 0.70-0.85 in radiata pine) and decide whether each tree's
   extraction is sufficient for HQP grading.

2. ``Stem_Description`` sheet — multi-row-per-tree, mirroring the
   Interpine PlotSafe ``StemDescription`` CSV schema (see
   ``memory/reference_interpine_hqp_codes.md``). Position rows carry a
   diameter at each valid section; feature rows carry stem-level codes
   (``Sw`` for sweep classification from the centerline; ``F`` for
   ovality if ``Is_oval_at_DBH`` is True; ``Br`` reserved for the
   future branch module).

Sweep classification follows the HQP LiDAR quickcard mapping documented
in the memory file; the algorithm computes the max lateral deviation of
the centerline polyline from a straight base-to-top reference line,
expresses it as a fraction of SED_obs, and maps to the Interpine code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from src.core.trunk_extraction import TrunkExtractionResult
from src.core.trunk_validation import SectionResult
from src.core.dendrometry import TreeMetrics


# ===========================================================================
# Coverage metrics
# ===========================================================================

@dataclass
class CoverageMetrics:
    """Per-tree stem-extraction coverage summary.

    All length-like fields in metres unless suffixed with ``_cm``.
    """
    tree_id: int
    ht_cloud_m: float       # max z of cloud points within crown buffer at the tree's basal xy
    ht_stem_m: float        # length of the extracted (curved-cylinder labelled) stem
    rh_obs: float           # ht_stem_m / ht_cloud_m (0-1)
    dbh_cm: float           # DBH at 1.30 m, in cm (mirror of dendrometry.DBH * 100)
    sed_obs_cm: float       # diameter at the topmost valid section, in cm
    sed_obs_height_m: float # z of the topmost valid section (provenance for sed_obs_cm)
    valid_sed: bool         # True if at least one valid section exists


def _topmost_valid_section_index(
    R_row: np.ndarray,
) -> int:
    """Return the **highest** section index for which the per-section
    radius is positive (i.e. the fit succeeded). Returns ``-1`` if no
    valid section exists.
    """
    valid = np.where(R_row > 0.0)[0]
    if valid.size == 0:
        return -1
    return int(valid.max())


def compute_coverage_metrics(
    xyz: np.ndarray,
    trunk_result: TrunkExtractionResult,
    section_result: SectionResult,
    tree_metrics: Sequence[TreeMetrics],
    crown_buffer_radius: float = 2.5,
) -> List[CoverageMetrics]:
    """Compute per-tree coverage metrics for the ``Coverage`` xlsx sheet.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        Full height-normalised point cloud. Used to read ``z`` values
        in the crown buffer column around each tree (for ``HT_cloud``).
    trunk_result : TrunkExtractionResult
        Provides ``tree_ids`` and ``tree_axes`` (basal centroid per tree).
    section_result : SectionResult
        Per-section fits. Provides ``R`` (and ``a, b`` when ellipse
        mode is active) to derive SED at the topmost valid section.
    tree_metrics : sequence of TreeMetrics
        Precomputed dendrometric metrics from
        :func:`src.core.dendrometry.compute_tree_metrics`. Provides
        ``height_total`` (= ``ht_stem_m``) and ``dbh``.
    crown_buffer_radius : float, default 2.5 (m)
        XY radius around each tree's basal centroid in which to read
        ``z`` from the cloud to estimate ``ht_cloud_m``. 2.5 m is a
        reasonable proxy for a radiata pine crown radius; in dense
        plots this may absorb a sliver of neighbouring canopy. Adjust
        per plot density if needed.

    Returns
    -------
    list of CoverageMetrics
        One entry per tree in ``section_result.tree_ids``, in the same
        order as ``tree_metrics``.

    Notes
    -----
    - ``HT_cloud`` is a **cloud-derived** estimate. It is not a
      manually-measured tree height; the user's protocol samples 3 trees
      per plot for ground-truth heights (used for calibration, not
      reported here).
    - ``RH_obs`` should be compared against the SED10 thresholds from
      the radiata pine literature (RH 0.70-0.85; Goulding & Murray 1976;
      Kimberley & Beets 2007). Anything < RH 0.70 means the merchantable
      bole is incomplete.
    - ``SED_obs`` reports the diameter of the *topmost extracted*
      section, not a fixed diameter threshold. To compare directly
      against the SED10 = 10 cm threshold, check whether
      ``sed_obs_cm <= 10.0``; if so, the extraction reached the
      merchantable top; if not, more taper is still wood-thick enough
      to be measurable but the pipeline cut off earlier.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3); got {xyz.shape}")
    if crown_buffer_radius <= 0.0:
        raise ValueError(
            f"crown_buffer_radius must be positive; got {crown_buffer_radius}"
        )

    x_all = xyz[:, 0]
    y_all = xyz[:, 1]
    z_all = xyz[:, 2]

    has_ellipse = (
        section_result.a is not None
        and section_result.b is not None
    )

    # Build a fast lookup of basal centroids by tree_id from trunk_result.
    centroids_by_id = {
        int(ax["tree_id"]): (float(ax["centroid"][0]), float(ax["centroid"][1]))
        for ax in trunk_result.tree_axes
    }

    metrics: List[CoverageMetrics] = []
    r2 = crown_buffer_radius * crown_buffer_radius

    for i, tm in enumerate(tree_metrics):
        tid = int(tm.tree_id)

        # --- HT_cloud_m: max z within crown buffer around basal centroid ---
        if tid in centroids_by_id:
            cx, cy = centroids_by_id[tid]
            dx = x_all - cx
            dy = y_all - cy
            in_buf = (dx * dx + dy * dy) <= r2
            if in_buf.any():
                ht_cloud_m = float(z_all[in_buf].max())
            else:
                ht_cloud_m = float(tm.z_top)  # fallback: no cloud in buffer
        else:
            ht_cloud_m = float(tm.z_top)

        # --- HT_stem_m: extracted stem length (from dendrometry) ---
        ht_stem_m = float(tm.height_total)

        # --- RH_obs ---
        rh_obs = ht_stem_m / ht_cloud_m if ht_cloud_m > 0.0 else 0.0

        # --- DBH_cm (mirror of main sheet's DBH * 100) ---
        dbh_cm = float(tm.dbh) * 100.0

        # --- SED_obs_cm at the topmost valid section ---
        r_row = section_result.R[i]
        idx_top = _topmost_valid_section_index(r_row)
        if idx_top >= 0:
            if has_ellipse:
                a_val = float(section_result.a[i, idx_top])  # type: ignore[index]
                b_val = float(section_result.b[i, idx_top])  # type: ignore[index]
                diameter_m = 2.0 * float(np.sqrt(max(a_val * b_val, 0.0)))
            else:
                diameter_m = 2.0 * float(r_row[idx_top])
            sed_obs_cm = diameter_m * 100.0
            sed_obs_height_m = float(section_result.sections[idx_top])
            valid_sed = True
        else:
            sed_obs_cm = 0.0
            sed_obs_height_m = 0.0
            valid_sed = False

        metrics.append(CoverageMetrics(
            tree_id=tid,
            ht_cloud_m=ht_cloud_m,
            ht_stem_m=ht_stem_m,
            rh_obs=rh_obs,
            dbh_cm=dbh_cm,
            sed_obs_cm=sed_obs_cm,
            sed_obs_height_m=sed_obs_height_m,
            valid_sed=valid_sed,
        ))

    return metrics


def coverage_metrics_to_dataframe(metrics: Sequence[CoverageMetrics]):
    """Convert a list of :class:`CoverageMetrics` to a pandas DataFrame
    for Excel export. Pandas is imported lazily so this module stays
    importable without it.
    """
    import pandas as pd

    rows = []
    for m in metrics:
        rows.append({
            "Tree_ID": m.tree_id,
            "HT_cloud_m": round(m.ht_cloud_m, 3),
            "HT_stem_m": round(m.ht_stem_m, 3),
            "RH_obs": round(m.rh_obs, 4),
            "DBH_cm": round(m.dbh_cm, 2),
            "SED_obs_cm": round(m.sed_obs_cm, 2) if m.valid_sed else float("nan"),
            "SED_obs_height_m": round(m.sed_obs_height_m, 3) if m.valid_sed else float("nan"),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Sweep classification (Interpine HQP, from centerline polyline)
# ===========================================================================

def _max_deviation_from_baseline(centerline: np.ndarray) -> float:
    """Max perpendicular distance from any centerline node to the
    straight line connecting the base node (lowest z) to the top node
    (highest z). Returns 0.0 if the polyline has fewer than 3 points
    or its base-top line has zero length.

    Distances are computed in 3D so curvature in any direction counts.
    """
    if centerline.ndim != 2 or centerline.shape[1] != 3 or centerline.shape[0] < 3:
        return 0.0

    z_col = centerline[:, 2]
    i_base = int(np.argmin(z_col))
    i_top = int(np.argmax(z_col))
    if i_base == i_top:
        return 0.0

    p0 = centerline[i_base]
    p1 = centerline[i_top]
    axis = p1 - p0
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-9:
        return 0.0

    rel = centerline - p0
    # Component along the axis
    t = rel @ axis / (axis_len * axis_len)
    closest = p0 + t[:, None] * axis
    dist = np.linalg.norm(centerline - closest, axis=1)
    return float(dist.max())


def _polyline_length_z(centerline: np.ndarray) -> float:
    """Vertical extent of the polyline (z_top − z_base). Used as a
    proxy for the stem length when classifying ``L`` vs ``S`` (the
    Interpine 6.1 m threshold)."""
    if centerline.ndim != 2 or centerline.shape[0] < 1:
        return 0.0
    z = centerline[:, 2]
    return float(z.max() - z.min())


def classify_sweep(
    centerline: Optional[np.ndarray],
    sed_obs_m: float,
    log_length_threshold_m: float = 6.1,
) -> Optional[str]:
    """Classify a stem's sweep severity into an Interpine HQP code.

    Algorithm (from ``memory/reference_interpine_hqp_codes.md``):

    1. ``amplitude`` = max perpendicular distance from any centerline
       node to the base-top straight line (3D).
    2. ``ratio = amplitude / sed_obs_m``.
    3. Map to code:

       =====================  =================
       ratio                  code
       =====================  =================
       ≤ 1/8                  ``"8"`` (Gun barrel)
       1/8 < r ≤ 1/5,
       length ≥ 6.1 m         ``"L"`` (Gentle, long-log OK)
       1/8 < r ≤ 1/5,
       length < 6.1 m         ``"S"`` (Gentle, short-log only)
       1/5 < r ≤ 1/3          ``"3"`` (Moderate)
       1/3 < r ≤ 1            ``"1"`` (Excessive)
       > 1                    ``"X"`` (Severe)
       =====================  =================

    Wobble (``W``) and Kink (``K``) require shape-pattern detection on
    the polyline (direction reversals, sharp angle changes); deferred
    to a future module.

    Parameters
    ----------
    centerline : ndarray of shape (M, 3) or None
        Per-tree polyline of section centres
        (``build_centerline_from_sections`` output). ``None`` or a
        polyline with fewer than 3 nodes returns ``None``.
    sed_obs_m : float
        Topmost-extracted-section diameter in metres (i.e.
        ``sed_obs_cm / 100``). Must be > 0 to avoid divide-by-zero;
        an invalid SED returns ``None``.
    log_length_threshold_m : float, default 6.1
        Switching point between ``L`` and ``S`` codes — the standard
        log length above which a stem qualifies for the long-log
        "OK for logs > 6.1 m" classification per the Interpine
        quickcard.

    Returns
    -------
    str or None
        One of ``"8"``, ``"L"``, ``"S"``, ``"3"``, ``"1"``, ``"X"``, or
        ``None`` if the centerline is too short / missing or
        ``sed_obs_m`` is zero / negative.
    """
    if centerline is None:
        return None
    if sed_obs_m <= 0.0:
        return None
    centerline = np.asarray(centerline, dtype=np.float64)
    if centerline.shape[0] < 3:
        return None

    amplitude = _max_deviation_from_baseline(centerline)
    if amplitude <= 0.0:
        return "8"

    ratio = amplitude / sed_obs_m

    if ratio <= 1.0 / 8.0:
        return "8"
    if ratio <= 1.0 / 5.0:
        length_z = _polyline_length_z(centerline)
        return "L" if length_z >= log_length_threshold_m else "S"
    if ratio <= 1.0 / 3.0:
        return "3"
    if ratio <= 1.0:
        return "1"
    return "X"


# ===========================================================================
# Stem_Description sheet (Interpine PlotSafe CSV schema)
# ===========================================================================

def build_stem_description_rows(
    plot_id: str,
    tree_metrics: Sequence[TreeMetrics],
    coverage_metrics: Sequence[CoverageMetrics],
    section_result: SectionResult,
    tree_centerlines: Sequence[Optional[np.ndarray]],
) -> List[dict]:
    """Build the rows that will land in the ``Stem_Description`` xlsx
    sheet, in the same order as Interpine's ``StemDescription.csv``.

    Columns emitted (per user request, omitting ``PopulationName`` and
    ``StratumName``):

    ``PlotId, TreeNumber, StemNo, Level, Position, Diameter, Br, Sw, F``

    For each tree in ``tree_metrics`` (assumed aligned with
    ``coverage_metrics`` and ``tree_centerlines``), emits:

    - **Base marker row**: ``Position = 0.0``, ``Diameter = NaN``
      (Interpine convention — flags the start of a stem entry).
    - **One position row per valid section**: ``Position = z_section``
      (m), ``Diameter`` = 2·√(a·b)·1000 mm (ellipse fit) or 2·R·1000 mm
      (circle fit), ``Br/Sw/F`` left NaN.
    - **One feature row per tree** with the sweep code from
      :func:`classify_sweep` if a code can be computed (``Position`` =
      NaN, ``Sw`` populated, ``Br/F`` NaN).
    - **One feature row per tree** with ``F = "O1.2+"`` if
      ``is_oval_at_dbh`` is True (Interpine HQP excessive ovality code).

    Branch (``Br``) and most ``F`` codes (``B10+``, ``N10+``, ``S*+``,
    ``D``, ``R``, ``F*+``, ``C``) are reserved for future modules — left
    NaN here.

    Parameters
    ----------
    plot_id : str
        ``PlotId`` value to stamp on every row (parametrised by the
        notebook; typically a plot identifier like ``"T460298B"``).
    tree_metrics : sequence of TreeMetrics
        From :func:`src.core.dendrometry.compute_tree_metrics`.
    coverage_metrics : sequence of CoverageMetrics
        From :func:`compute_coverage_metrics`; provides per-tree
        ``sed_obs_cm`` for the sweep classification denominator.
    section_result : SectionResult
        Per-section fits. Read for ``a, b`` (ellipse mode) or ``R``
        (circle mode) at every valid section per tree.
    tree_centerlines : sequence of ndarray or None
        Per-tree polylines from
        :func:`src.core.trunk_validation.build_centerline_from_sections`,
        aligned 1:1 with ``tree_metrics``.

    Returns
    -------
    list of dict
        One dict per row, keyed by the Interpine column names. Pass
        directly to ``pd.DataFrame(rows)`` for the xlsx writer.
    """
    if len(tree_metrics) != len(coverage_metrics):
        raise ValueError(
            "tree_metrics and coverage_metrics must have the same length; "
            f"got {len(tree_metrics)} and {len(coverage_metrics)}"
        )
    if len(tree_metrics) != len(tree_centerlines):
        raise ValueError(
            "tree_metrics and tree_centerlines must have the same length; "
            f"got {len(tree_metrics)} and {len(tree_centerlines)}"
        )

    has_ellipse = (
        section_result.a is not None
        and section_result.b is not None
    )
    sections = np.asarray(section_result.sections, dtype=np.float64)

    rows: List[dict] = []

    def empty_row(tree_number: int) -> dict:
        return {
            "PlotId": plot_id,
            "TreeNumber": tree_number,
            "StemNo": 0,
            "Level": 0,
            "Position": float("nan"),
            "Diameter": float("nan"),
            "Br": float("nan"),
            "Sw": float("nan"),
            "F": float("nan"),
        }

    for i, (tm, cov) in enumerate(zip(tree_metrics, coverage_metrics)):
        tree_number = int(tm.tree_id)

        # 1. Base marker row (Interpine convention): Position=0.0, Diameter=NaN
        base_row = empty_row(tree_number)
        base_row["Position"] = 0.0
        rows.append(base_row)

        # 2. Position rows: one per valid section
        r_row = section_result.R[i]
        for j in range(sections.size):
            if r_row[j] <= 0.0:
                continue
            if has_ellipse:
                a_val = float(section_result.a[i, j])  # type: ignore[index]
                b_val = float(section_result.b[i, j])  # type: ignore[index]
                diameter_mm = (
                    2.0 * float(np.sqrt(max(a_val * b_val, 0.0))) * 1000.0
                )
            else:
                diameter_mm = 2.0 * float(r_row[j]) * 1000.0

            pos_row = empty_row(tree_number)
            pos_row["Position"] = round(float(sections[j]), 3)
            pos_row["Diameter"] = round(diameter_mm, 1)
            rows.append(pos_row)

        # 3. Feature row: Sw (sweep classification from centerline + SED)
        sweep_code = classify_sweep(
            tree_centerlines[i],
            sed_obs_m=(cov.sed_obs_cm / 100.0) if cov.valid_sed else 0.0,
        )
        if sweep_code is not None:
            sw_row = empty_row(tree_number)
            sw_row["Sw"] = sweep_code
            rows.append(sw_row)

        # 4. Feature row: F = O1.2+ if oval at DBH (HQP excessive ovality)
        if tm.is_oval_at_dbh:
            f_row = empty_row(tree_number)
            f_row["F"] = "O1.2+"
            rows.append(f_row)

    return rows


def stem_description_to_dataframe(rows: Sequence[dict]):
    """Convert a list of stem-description row dicts to a pandas
    DataFrame in the exact Interpine column order.
    """
    import pandas as pd

    columns = [
        "PlotId", "TreeNumber", "StemNo", "Level",
        "Position", "Diameter", "Br", "Sw", "F",
    ]
    return pd.DataFrame(rows, columns=columns)
