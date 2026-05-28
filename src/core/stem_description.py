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


def classify_sweep_zones(
    centerline: Optional[np.ndarray],
    sed_obs_m: float,
    upgrade_min_section_m: float = 3.0,
    log_length_threshold_m: float = 6.1,
) -> List[tuple]:
    """Decompose a centerline polyline into Interpine sweep zones via a
    sliding-window local-amplitude classifier (F1.1 algorithm).

    This is the zone-aware sibling of :func:`classify_sweep`: where the
    single-code helper reduces the whole stem to one Interpine code by
    the polyline's global max amplitude, this function walks the
    polyline with a 4 m window (default) and assigns a code per node,
    then coalesces consecutive same-code nodes into zones — recovering
    cases where a 21 m stem labelled ``"1"`` globally actually breaks
    into something like ``[8 (0-5 m), L (5-12 m), 1 (12-18 m), S (18-21 m)]``.

    Algorithm (see ``memory/reference_interpine_hqp_codes.md`` for the
    code-to-amplitude mapping it consumes):

    1. **Sort nodes by z** to be robust against polylines not stored in
       ascending order.
    2. **Per-node amplitude vs the global base-top chord**: build the
       chord from the bottom-most polyline node to the top-most. For
       every node, compute the perpendicular distance to that chord.
       This signal is naturally localised — nodes outside any bow
       region sit on the chord (amplitude ≈ 0), nodes inside a bow have
       amplitude proportional to the bow height. A windowed local-chord
       variant was tried first but tilted the chord whenever the window
       included a bow as an endpoint, leaking the bow's amplitude into
       nominally-straight neighbouring nodes. The global-chord approach
       is simpler and correctly attributes amplitude to the actual
       deviation node.
    3. **Per-node code mapping**: ``ratio = amplitude / sed_obs_m``,
       then map to ``"8" / "LS" / "3" / "1" / "X"`` by the same severity
       thresholds as :func:`classify_sweep`. ``"LS"`` is a placeholder
       resolved to ``"L"`` or ``"S"`` after coalescing, based on zone
       length (≥ 6.1 m → L, else S).
    4. **Coalesce** consecutive same-code nodes into zones
       ``(z_start, z_end, code)``.
    5. **Upgrade-rule enforcement** (Interpine HQP quickcard): a zone
       with a *better* code that is shorter than ``upgrade_min_section_m``
       and is sandwiched between two *worse* zones is **absorbed into
       the worse neighbour**. The reverse — a short worse zone between
       two better zones — keeps its worse code (kinks, local defects
       don't average out). The rule iterates until stable, then
       same-code zones are merged again.

    Limitations
    -----------
    - Pure amplitude-based: cannot distinguish ``W`` (wobble — multiple
      XY direction reversals) or ``K`` (kink — sharp angle change at a
      single segment). Those require shape-pattern detection on the
      polyline, deferred to a future module.
    - ``L`` vs ``S`` is resolved by zone length only, not by the
      "consistent direction" vs "back and forth" criterion of the
      Interpine quickcard (which also needs shape detection). Length is
      the dominant operative cue for log-merchantability, so the
      length-based split is a useful approximation.

    Parameters
    ----------
    centerline : ndarray of shape (N, 3) or None
        Per-tree polyline of section centres. ``None`` or fewer than
        3 nodes returns an empty list.
    sed_obs_m : float
        Topmost-extracted-section diameter in metres. Zero or negative
        returns an empty list.    upgrade_min_section_m : float, default 3.0
        Minimum length a better-coded zone must have to survive the
        upgrade rule. Interpine HQP quickcard: "Only Upgrade Branch
        or Sweep Class … if > 3 m Section Between Zones of Lower
        Quality".
    log_length_threshold_m : float, default 6.1
        Threshold for splitting the SED/5 amplitude class into ``L``
        (length ≥ 6.1 m → "OK for logs > 6.1 m") vs ``S`` (length <
        6.1 m → short-log only).

    Returns
    -------
    list of (z_start, z_end, code) tuples
        Sorted by ``z_start`` ascending. Each tuple defines a contiguous
        zone of the polyline with its Interpine sweep code.
    """
    if centerline is None or sed_obs_m <= 0.0:
        return []
    cl = np.asarray(centerline, dtype=np.float64)
    if cl.ndim != 2 or cl.shape[1] != 3 or cl.shape[0] < 3:
        return []

    # 1. Sort by z
    order = np.argsort(cl[:, 2])
    cl = cl[order]
    z = cl[:, 2]
    n = cl.shape[0]

    # 2. Per-node perpendicular distance to the global base-top chord.
    # Vectorised: build the chord from cl[0] to cl[-1], project every
    # node onto it, and take the residual distance.
    p0 = cl[0]
    p1 = cl[-1]
    axis = p1 - p0
    axis_len_sq = float(np.dot(axis, axis))
    if axis_len_sq < 1e-18:
        return []
    rel = cl - p0
    t = (rel @ axis) / axis_len_sq
    closest = p0 + t[:, None] * axis
    local_amp = np.linalg.norm(cl - closest, axis=1)

    # 3. Per-node code mapping
    inv_sed = 1.0 / sed_obs_m
    codes_per_node: List[str] = []
    for amp in local_amp:
        ratio = amp * inv_sed
        if ratio <= 1.0 / 8.0:
            codes_per_node.append("8")
        elif ratio <= 1.0 / 5.0:
            codes_per_node.append("LS")
        elif ratio <= 1.0 / 3.0:
            codes_per_node.append("3")
        elif ratio <= 1.0:
            codes_per_node.append("1")
        else:
            codes_per_node.append("X")

    # 4. Coalesce consecutive same-code nodes into zones
    zones: List[tuple] = []
    run_start = 0
    for i in range(1, n):
        if codes_per_node[i] != codes_per_node[run_start]:
            zones.append((
                float(z[run_start]),
                float(z[i - 1]),
                codes_per_node[run_start],
            ))
            run_start = i
    zones.append((
        float(z[run_start]),
        float(z[n - 1]),
        codes_per_node[run_start],
    ))

    # Resolve LS → L or S by zone length
    zones = [
        (
            s, e,
            ("L" if (e - s) >= log_length_threshold_m else "S")
            if c == "LS" else c
        )
        for (s, e, c) in zones
    ]

    # 5. Upgrade-rule enforcement (iterate to fixed point)
    severity = {"8": 0, "L": 1, "S": 1, "3": 2, "1": 3, "X": 4}
    safety = 0
    while safety < 20:
        safety += 1
        new_zones: List[tuple] = []
        any_change = False

        for k, (s, e, c) in enumerate(zones):
            length = e - s
            sev = severity.get(c, 0)

            absorb = False
            if k > 0 and k < len(zones) - 1 and length < upgrade_min_section_m:
                prev_c = zones[k - 1][2]
                next_c = zones[k + 1][2]
                prev_sev = severity.get(prev_c, 0)
                next_sev = severity.get(next_c, 0)
                if sev < prev_sev and sev < next_sev:
                    worse_code = prev_c if prev_sev >= next_sev else next_c
                    if new_zones and new_zones[-1][2] == worse_code:
                        s0, _, _ = new_zones[-1]
                        new_zones[-1] = (s0, e, worse_code)
                    else:
                        new_zones.append((s, e, worse_code))
                    absorb = True
                    any_change = True

            if not absorb:
                if new_zones and new_zones[-1][2] == c:
                    s0, _, _ = new_zones[-1]
                    new_zones[-1] = (s0, e, c)
                else:
                    new_zones.append((s, e, c))

        zones = new_zones
        if not any_change:
            break

    return zones


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

    **Sparse schema** — emits boundary rows only, NOT one per section.
    Per Interpine PlotSafe convention, a stem entry encodes its quality
    zones by their **endpoints**, not by dense diameter sampling. For
    each tree:

    - **Base marker row**: ``Position = 0.0``, ``Diameter = NaN``.
    - **DBH row**: ``Position = 1.30``, ``Diameter`` = the equivalent
      diameter at the DBH section in mm (= ``tm.dbh * 1000``). Skipped
      when ``tm.valid_at_dbh`` is False.
    - **End-of-zone Sw row**: ``Position = z_top`` (topmost extracted
      section), ``Sw`` = sweep code from :func:`classify_sweep`. This
      single row currently encodes the whole stem as one zone. When a
      future shape-pattern classifier detects multiple zones (e.g. L
      from 0-4 m, S from 4-7 m), this function will emit one Sw row per
      zone boundary.
    - **Optional F row** with ``F = "O1.2+"`` if ``is_oval_at_dbh`` is
      True (Interpine HQP excessive ovality code).

    Dense per-section diameter sampling lives on the separate ``Taper``
    sheet (see :func:`build_taper_rows`).

    Branch (``Br``) and most ``F`` codes (``B10+``, ``N10+``, ``S*+``,
    ``D``, ``R``, ``F*+``, ``C``) are reserved for future modules — not
    emitted here.

    Parameters
    ----------
    plot_id : str
        ``PlotId`` value to stamp on every row (parametrised by the
        notebook; typically a plot identifier like ``"T460298B"``).
    tree_metrics : sequence of TreeMetrics
        From :func:`src.core.dendrometry.compute_tree_metrics`.
        Provides ``dbh`` (equivalent diameter in m), ``valid_at_dbh``,
        and ``is_oval_at_dbh``.
    coverage_metrics : sequence of CoverageMetrics
        From :func:`compute_coverage_metrics`; provides per-tree
        ``sed_obs_cm`` (sweep denominator) and ``sed_obs_height_m``
        (position of the end-of-zone Sw row).
    section_result : SectionResult
        Per-section fits. Kept on the API for forward compatibility
        with a multi-zone sweep classifier; not currently consulted by
        the sparse-row path (it uses the topmost-section height from
        ``coverage_metrics`` instead).
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

        # 1. Base marker row (Interpine convention): Position=0, Diameter=NaN
        base_row = empty_row(tree_number)
        base_row["Position"] = 0.0
        rows.append(base_row)

        # 2. DBH row: Position=1.30, Diameter from tm.dbh (m → mm)
        if tm.valid_at_dbh and tm.dbh > 0.0:
            dbh_row = empty_row(tree_number)
            dbh_row["Position"] = round(float(tm.dbh_section_height), 3)
            dbh_row["Diameter"] = round(float(tm.dbh) * 1000.0, 1)
            rows.append(dbh_row)

        # 3. Sweep zones via the sliding-window classifier (F1.1). Emits
        # one row per detected zone, with Position at the zone's z_end.
        # When the classifier returns a single zone covering the whole
        # stem, this collapses to the previous single-row behaviour.
        zones = classify_sweep_zones(
            tree_centerlines[i],
            sed_obs_m=(cov.sed_obs_cm / 100.0) if cov.valid_sed else 0.0,
        )
        for (_z_start, z_end, code) in zones:
            sw_row = empty_row(tree_number)
            sw_row["Position"] = round(float(z_end), 3)
            sw_row["Sw"] = code
            rows.append(sw_row)

        # 4. Feature row: F = O1.2+ if oval at DBH (HQP excessive ovality)
        if tm.is_oval_at_dbh:
            f_row = empty_row(tree_number)
            f_row["F"] = "O1.2+"
            rows.append(f_row)

    return rows


def build_taper_rows(
    tree_metrics: Sequence[TreeMetrics],
    section_result: SectionResult,
    z_start: float = 1.30,
    z_step: float = 1.0,
    z_tolerance: float = 0.15,
) -> List[dict]:
    """Build per-meter taper rows for the ``Taper`` xlsx sheet.

    Emits one row per tree per target height ``z``, where ``z`` walks
    from the DBH height (1.30 m default) upward in ``z_step`` (1 m
    default) increments, aligned to integer-metre values above DBH:

        targets = [1.30, 2.00, 3.00, 4.00, ...] up to the topmost
        valid section's z.

    For each target, finds the closest section within ``z_tolerance``
    and records its diameter (mm). The 1 m default matches the
    operational standard for forestry stem taper sampling — dense
    enough for Smalian / Huber / Newton volume integration and Kozak /
    Max-Burkhart taper-function fitting, without inflating the export
    with per-20 cm noise.

    Output columns: ``Tree_ID, Position_m, Diameter_mm``.

    Parameters
    ----------
    tree_metrics : sequence of TreeMetrics
        Same per-tree order as the rest of the pipeline.
    section_result : SectionResult
        Per-section fits in either circle (``R``) or ellipse
        (``a, b``) mode.
    z_start : float, default 1.30
        First target height in metres (DBH convention).
    z_step : float, default 1.0
        Spacing between subsequent targets in metres. Set to 0.5 to
        densify, 2.0 for log-interval spacing.
    z_tolerance : float, default 0.15
        Half-window for snapping each target to the closest
        ``section_result.sections`` entry. With the default 0.2 m
        sectioning, 0.15 m covers the natural ±0.10 m offset.

    Returns
    -------
    list of dict
        One row per (tree, target) pair where a valid section was
        found. Trees with no valid sections emit no rows.
    """
    import math

    has_ellipse = (
        section_result.a is not None
        and section_result.b is not None
    )
    sections = np.asarray(section_result.sections, dtype=np.float64)

    rows: List[dict] = []
    for i, tm in enumerate(tree_metrics):
        tid = int(tm.tree_id)
        r_row = section_result.R[i]
        valid = np.where(r_row > 0.0)[0]
        if valid.size == 0:
            continue
        z_top = float(sections[valid.max()])

        # Build target list: z_start, then integer-metre values above z_start.
        targets: List[float] = [z_start]
        zt = float(math.ceil(z_start + 1e-9))
        while zt <= z_top + 1e-9:
            targets.append(zt)
            zt += z_step

        for tgt in targets:
            diffs = np.abs(sections - tgt)
            j = int(np.argmin(diffs))
            if diffs[j] > z_tolerance or r_row[j] <= 0.0:
                continue
            if has_ellipse:
                a_val = float(section_result.a[i, j])  # type: ignore[index]
                b_val = float(section_result.b[i, j])  # type: ignore[index]
                d_mm = 2.0 * float(np.sqrt(max(a_val * b_val, 0.0))) * 1000.0
            else:
                d_mm = 2.0 * float(r_row[j]) * 1000.0
            rows.append({
                "Tree_ID": tid,
                "Position_m": round(float(tgt), 2),
                "Diameter_mm": round(d_mm, 1),
            })

    return rows


def taper_to_dataframe(rows: Sequence[dict]):
    """Convert taper rows to a pandas DataFrame in the canonical
    column order ``[Tree_ID, Position_m, Diameter_mm]``.
    """
    import pandas as pd
    return pd.DataFrame(rows, columns=["Tree_ID", "Position_m", "Diameter_mm"])


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
