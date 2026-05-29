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


# ===========================================================================
# F1.5 — Direction-pattern primitives (L/S split + future W/K detection)
# ===========================================================================

def _count_swings(s: np.ndarray, deadband: float) -> int:
    """Count direction reversals in a 1D profile (zigzag swing count)
    with a prominence deadband.

    The intuition: walk the profile tracking the running extreme of the
    current direction. A reversal is registered only when the value
    has retraced from the running extreme by at least ``deadband``.
    Tiny jitter (< ``deadband``) is absorbed without spurious swings.

    Used by :func:`_polyline_direction_metrics` to produce ``n_bows`` —
    the number of bows / direction reversals in a polyline's lateral
    offset profile. Consistent (single bow) sweeps yield ``n_bows ≤ 1``;
    back-and-forth (S-curve, camel-back, wave) yields ``n_bows ≥ 2``.

    Parameters
    ----------
    s : array_like
        1D profile of signed values (lateral offset along the dominant
        sweep direction in our use).
    deadband : float
        Minimum retracement required to register a direction reversal,
        in the same units as ``s`` (typically metres). Set to the
        polyline noise floor (~1 cm for our centroid jitter).

    Returns
    -------
    int
        Number of direction reversals (swings) detected.
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    if s.size < 3:
        return 0
    swings = 0
    direction = 0  # +1 rising, -1 falling, 0 undetermined
    pivot = float(s[0])
    for v_arr in s[1:]:
        v = float(v_arr)
        if direction == 0:
            if v - pivot >= deadband:
                direction = 1
                pivot = v
            elif pivot - v >= deadband:
                direction = -1
                pivot = v
            # else: within deadband; pivot stays where it is so the next
            # eventual move is measured from the same anchor.
        elif direction == 1:  # rising
            if v > pivot:
                pivot = v  # new high
            elif pivot - v >= deadband:
                swings += 1
                direction = -1
                pivot = v
        else:  # direction == -1, falling
            if v < pivot:
                pivot = v  # new low
            elif v - pivot >= deadband:
                swings += 1
                direction = 1
                pivot = v
    return swings


def _max_turn_angle_deg(centerline: np.ndarray) -> float:
    """Largest turn angle (degrees) between consecutive XY-projected
    polyline segments.

    Used by :func:`_polyline_direction_metrics` to expose ``max_turn_deg``
    for future ``K`` (kink) detection in F1.5b.
    """
    if centerline.ndim != 2 or centerline.shape[0] < 3:
        return 0.0
    xy = centerline[:, :2]
    segs = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    max_ang = 0.0
    for i in range(len(segs) - 1):
        la, lb = float(seg_len[i]), float(seg_len[i + 1])
        if la < 1e-9 or lb < 1e-9:
            continue
        cos_ang = float(np.clip(np.dot(segs[i], segs[i + 1]) / (la * lb), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_ang)))
        if ang > max_ang:
            max_ang = ang
    return max_ang


def _polyline_direction_metrics(
    centerline: np.ndarray,
    deadband_m: float = 0.01,
) -> dict:
    """Direction-pattern metrics for a centerline polyline (F1.5).

    Used by :func:`classify_sweep_zones` to separate `L` (consistent)
    from `S` (back-and-forth) within SED/5-amplitude zones, and (in
    F1.5b) to detect `W` and `K` defect flags.

    Algorithm:

    1. Sort by z (defensive).
    2. Compute each node's perpendicular offset vector from the
       base-top chord (3D, lies in the plane ⊥ to axis).
    3. Reduce to a signed scalar profile by projecting offsets onto
       the **dominant sweep direction** (first right-singular vector
       of the offsets matrix — i.e. the PCA principal axis of the
       2-D-ish offset cloud).
    4. ``n_bows`` = number of swings in that signed profile via
       :func:`_count_swings` with a prominence ``deadband_m``.
    5. ``max_abs_offset_m`` = max |perpendicular offset| (m), the
       absolute amplitude — used by F1.5b W detection (`> 5 cm`).
    6. ``max_turn_deg`` = max turn angle between consecutive XY
       segments — used by F1.5b K detection.

    Returns a dict; safe to call on degenerate polylines (returns
    zero metrics for fewer than 3 nodes or a degenerate base-top axis).

    Parameters
    ----------
    centerline : ndarray of shape (M, 3)
        Polyline nodes ``(x, y, z)``. Sorted internally by z.
    deadband_m : float, default 0.01 (1 cm)
        Prominence floor for the swing count, in metres. Filters out
        per-section centroid jitter (typically a few mm at section
        scale) so only real direction changes count as bows.

    Returns
    -------
    dict
        ``{"n_bows": int, "max_abs_offset_m": float, "max_turn_deg": float}``.
    """
    zero = {"n_bows": 0, "max_abs_offset_m": 0.0, "max_turn_deg": 0.0}
    cl = np.asarray(centerline, dtype=np.float64)
    if cl.ndim != 2 or cl.shape[1] != 3 or cl.shape[0] < 3:
        return zero

    order = np.argsort(cl[:, 2])
    cl = cl[order]

    p0 = cl[0]
    p1 = cl[-1]
    axis = p1 - p0
    axis_len_sq = float(np.dot(axis, axis))
    if axis_len_sq < 1e-18:
        return zero

    rel = cl - p0
    t = (rel @ axis) / axis_len_sq
    closest = p0 + t[:, None] * axis
    offsets = cl - closest  # (M, 3); each row ⊥ to axis

    offset_norms = np.linalg.norm(offsets, axis=1)
    max_abs_offset_m = float(offset_norms.max())

    # Signed scalar via PCA dominant direction. For a near-straight
    # polyline (all offsets ≈ 0) the SVD direction is meaningless but
    # the projected values are tiny → swing counter returns 0.
    if max_abs_offset_m < 1e-12:
        signed = np.zeros(cl.shape[0])
    else:
        _, _, vt = np.linalg.svd(offsets, full_matrices=False)
        u_dir = vt[0]
        signed = offsets @ u_dir

    n_bows = _count_swings(signed, deadband_m)
    max_turn_deg = _max_turn_angle_deg(cl)

    return {
        "n_bows": n_bows,
        "max_abs_offset_m": max_abs_offset_m,
        "max_turn_deg": max_turn_deg,
    }


def classify_sweep_zones(
    centerline: Optional[np.ndarray],
    sed_obs_m: float,
    min_zone_length_by_code: Optional[dict] = None,
    direction_deadband_m: float = 0.01,
    w_amplitude_floor_m: float = 0.05,
    k_angle_threshold_deg: float = 15.0,
    k_zone_half_width_m: float = 0.25,
) -> List[tuple]:
    """Decompose a centerline polyline into Interpine sweep zones via a
    per-node global-chord amplitude classifier (F1.1) with per-code
    minimum-length enforcement (F1.2).

    This is the zone-aware sibling of :func:`classify_sweep`: where the
    single-code helper reduces the whole stem to one Interpine code by
    the polyline's global max amplitude, this function assigns a code
    per node from its distance to the base-top chord, then coalesces
    consecutive same-code nodes into zones — recovering cases where a
    21 m stem labelled ``"1"`` globally actually breaks into something
    like ``[8 (0-5 m), L (5-12 m), 1 (12-18 m), S (18-21 m)]``.

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
       resolved to ``"L"`` or ``"S"`` after coalescing (step 5).
    4. **Coalesce** consecutive same-code nodes into zones
       ``(z_start, z_end, code)``.
    5. **Resolve LS by direction pattern** (F1.5a): the Interpine
       quickcard separates ``L`` from ``S`` by direction, not just
       length — ``L`` is a gentle sweep in a *consistent single
       direction*; ``S`` is a gentle sweep *back and forth*. For each
       SED/5 zone, slice the centerline to that zone's z-range and
       compute :func:`_polyline_direction_metrics`. The zone is
       ``"L"`` when ``n_bows ≤ 1`` (straight or single consistent
       bow) and ``"S"`` when ``n_bows ≥ 2`` (back-and-forth, e.g.
       S-curve, camel-back, wave). Length is no longer the L/S split
       criterion — it was a proxy in F1.2-F1.4. ``S`` has no maximum
       length under the Interpine convention: a long back-and-forth
       sweep stays ``S``. The F1.4 noise-floor minimum still applies
       to absorb sub-operational zones in step 6.
    6. **Minimum-zone-length enforcement** (F1.4, cluster-aware noise
       floor): a zone is *short* if its length is below its code's
       operative minimum (``min_zone_length_by_code``); ``X`` is exempt
       (its 0.3-1 m definition is intrinsic, so it is always *stable*).
       Maximal **runs** of consecutive short zones are absorbed as a
       block into the worst (higher-severity) **stable** anchor on
       either side, skipping over the other short zones in the run.
       Boundary runs (one stable side, or none) absorb into the single
       available anchor — or, if the whole polyline is sub-operational,
       collapse to the worst code present.

       This supersedes the per-zone F1.3 pass, which absorbed each short
       zone into its immediate worst neighbour and **ping-ponged on
       clusters** of adjacent short zones (e.g. ``S/3/S/3/S`` over 2 m
       between two long ``8`` zones): each short zone's worst neighbour
       was another short zone, so the cluster collapsed inward
       (swapping ``S``↔``3``) and never reached the surrounding stable
       ``8``. Anchoring whole runs to the nearest stable zone fixes
       this in one deterministic pass.

       The F1.3/F1.2 rationale still holds: sub-operational zones are
       polyline noise (the polyline samples every ~0.2 m); the
       Interpine upgrade rule's ``> 3 m`` threshold implicitly assumes
       operative lengths. Worked rationale and trade-off in
       ``external_references/interpine/sweep_classification_literature_review.md``
       §6.

    The per-code minimums are **soft / operational, not normative**.
    The amplitude thresholds (SED/8, SED/5, SED/3, SED/1) are anchored
    in MPI's NZ log-grade tolerances; the section lengths come only from
    Interpine's quickcard, which Interpine itself frames as an
    operational segregation tool ("choose the option that best
    segregates"), not a rigid standard.

    Limitations
    -----------
    - Pure amplitude-based: cannot distinguish ``W`` (wobble — multiple
      XY direction reversals) or ``K`` (kink — sharp angle change at a
      single segment). Those require shape-pattern detection on the
      polyline, deferred to a future module.
    - ``L`` vs ``S`` is direction-aware in F1.5a (``n_bows`` from
      :func:`_polyline_direction_metrics`), matching the Interpine
      quickcard's "consistent direction" vs "back and forth" criterion.
    - F1.5b adds ``W`` (back-and-forth with absolute amplitude > 5 cm,
      pulp quality) and ``K`` (sharp XY direction change). Both are
      reported in the ``Sw`` column at the same level as 8/L/S/3/1/X.
      Severity sits between ``3`` and ``1``: per Jorge's quickcard
      mapping, sawmill-quality codes (``8 / L / S / 3``) sit above the
      "Generally Pulp Quality" line and pulp-quality codes
      (``W / K / 1 / X``) below it. ``K`` is intrinsically a point
      defect (~0.5 m, like ``X``) and is exempt from the length floor.

    Parameters
    ----------
    centerline : ndarray of shape (N, 3) or None
        Per-tree polyline of section centres. ``None`` or fewer than
        3 nodes returns an empty list.
    sed_obs_m : float
        Topmost-extracted-section diameter in metres. Zero or negative
        returns an empty list.
    direction_deadband_m : float, default 0.01 (1 cm)
        Prominence floor for :func:`_count_swings` when counting
        ``n_bows`` in step 5. Filters out per-section centroid jitter
        (typical at section scale) so only real direction reversals
        count as bows. 1 cm is well above the centroid noise floor
        and well below the cm-scale of a genuine sweep, giving a
        clean L vs S split.
    w_amplitude_floor_m : float, default 0.05 (5 cm)
        Absolute-amplitude floor (m) for upgrading a back-and-forth
        verdict from ``S`` to ``W`` in step 5. Matches the Interpine
        quickcard's "> 5 cm" rule for wobble (pulp quality). Note this
        is an **absolute** centimetres threshold, distinct from the
        SED-fraction amplitudes used for 8/L/S/3/1/X.
    k_angle_threshold_deg : float, default 15.0
        Per-segment turn-angle threshold (deg) for K detection in step
        3.5. Set ≤ 0 to disable K detection. 15° is a noticeable
        change-of-direction without being extreme; tune up if real
        data shows spurious K firings on natural curvature.
    k_zone_half_width_m : float, default 0.25
        Half-width (m) of the K override window centred on a detected
        kink — resulting K zones are ~ 2 × this value (~ 0.5 m by
        default), matching the quickcard's "Max 0.5 m" reference.
    min_zone_length_by_code : dict, optional
        Per-code operative minimum zone length (m) for the absorption
        pass. Defaults to ``{"8": 4.0, "L": 4.0, "S": 4.0, "3": 3.0,
        "W": 2.0, "K": 0.0, "1": 2.0}``. ``"X"`` is intentionally
        absent (min 0). ``X`` and ``K`` are defect-flag codes and are
        always treated as stable (never absorbed for being short).
        ``W`` carries a short defect-flag minimum (2 m, not the
        quickcard's 4 m observational window) per the user-confirmed
        treatment of W as a pulp-quality flag.

    Returns
    -------
    list of (z_start, z_end, code) tuples
        Sorted by ``z_start`` ascending. Each tuple defines a contiguous
        zone of the polyline with its Interpine sweep code.
    """
    if min_zone_length_by_code is None:
        min_zone_length_by_code = {
            "8": 4.0, "L": 4.0, "S": 4.0, "3": 3.0,
            # W is a defect flag (pulp quality); user spec — short min
            # (2 m), not the 4 m observational window of the quickcard.
            "W": 2.0,
            # K is a point-defect flag (quickcard "Max 0.5 m"); exempt
            # from the length floor like X.
            "K": 0.0,
            "1": 2.0,
        }
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

    # 3.5. K detection (F1.5b): per-segment XY turn-angle scan. For each
    # interior node where the angle between the incoming and outgoing
    # segments exceeds ``k_angle_threshold_deg``, override the codes of
    # nodes within ±``k_zone_half_width_m`` of that z to ``"K"``. The
    # subsequent coalesce will materialise these as ~0.5 m K zones.
    if (
        k_angle_threshold_deg > 0.0
        and k_zone_half_width_m > 0.0
        and n >= 3
    ):
        xy = cl[:, :2]
        segs = np.diff(xy, axis=0)
        seg_len = np.linalg.norm(segs, axis=1)
        for i in range(1, n - 1):
            la = float(seg_len[i - 1])
            lb = float(seg_len[i])
            if la < 1e-9 or lb < 1e-9:
                continue
            cos_ang = float(np.clip(
                np.dot(segs[i - 1], segs[i]) / (la * lb), -1.0, 1.0,
            ))
            ang_deg = float(np.degrees(np.arccos(cos_ang)))
            if ang_deg > k_angle_threshold_deg:
                z_kink = float(z[i])
                for j in range(n):
                    if abs(float(z[j]) - z_kink) <= k_zone_half_width_m:
                        codes_per_node[j] = "K"

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

    # 5. Resolve LS → L (consistent) or S (back-and-forth) by direction.
    # F1.5a — direction dominates over length. Crucially, a back-and-forth
    # sweep manifests in the polyline as SEPARATE SED/5 amplitude zones
    # (one for each bow side) separated by intermediate axis-crossings
    # (the "8" gaps where the centerline is near the chord). Each
    # individual SED/5 zone is monotonic by construction (it's a
    # contiguous run of SED/5 amplitude), so per-zone slicing would
    # always classify it as L. We therefore compute the direction
    # metric **once over the whole centerline** and apply the verdict
    # to all LS zones: a tree's overall sweep character (one consistent
    # bow vs back-and-forth) is the right unit for the Interpine L/S
    # distinction. Length is no longer the L/S criterion (it was a
    # proxy in F1.2-F1.4); the F1.4 noise-floor min still applies in
    # step 6. S has no maximum length under the Interpine convention.
    global_dir = _polyline_direction_metrics(
        cl, deadband_m=direction_deadband_m,
    )
    is_back_and_forth = global_dir["n_bows"] >= 2
    is_high_amp = global_dir["max_abs_offset_m"] > w_amplitude_floor_m
    # F1.5b: W upgrade. A back-and-forth pattern with absolute amplitude
    # above 5 cm (default) is a Wobble (pulp quality), not gentle S.
    # `W` is reported in the Sw column at the same level as 8/L/S/3/1/X
    # per Jorge's quickcard mapping (severity placed between 3 and 1 —
    # see step 6 severity dict). The amplitude threshold is **absolute
    # centimetres**, distinct from the SED-fraction amplitudes used for
    # 8/L/S/3/1/X.
    if is_back_and_forth and is_high_amp:
        ls_target = "W"
    elif is_back_and_forth:
        ls_target = "S"
    else:
        ls_target = "L"
    zones = [
        (s, e, ls_target if c == "LS" else c)
        for (s, e, c) in zones
    ]

    # 6. Minimum-zone-length enforcement (F1.4 cluster-aware noise floor)
    # Any non-X zone shorter than its code's minimum length is "short".
    # Maximal RUNS of consecutive short zones are absorbed as a block
    # into the worst (higher-severity) STABLE anchor on either side,
    # skipping over the other short zones in the run. X is exempt — its
    # 0.3-1 m definition makes it the only code whose existence at short
    # length is intentional, so X acts as a stable anchor / run splitter.
    #
    # F1.4 supersedes the per-zone F1.3 pass, which absorbed each short
    # zone into its immediate worst neighbour. That ping-ponged on
    # CLUSTERS of adjacent short zones (e.g. S/3/S/3/S over 2 m between
    # two long "8" zones): each short zone's worst neighbour was another
    # short zone, so the cluster collapsed inward (swapping S<->3) and
    # never reached the surrounding stable "8" — leaving sub-operational
    # noise in the output. Anchoring runs to the nearest STABLE zone
    # fixes this in a single deterministic pass.
    #
    # F1.3/F1.2 rationale still holds: sub-operational zones are polyline
    # noise (the polyline samples every ~0.2 m); the Interpine upgrade
    # rule's > 3 m threshold implicitly assumes operative lengths. See
    # ``external_references/interpine/sweep_classification_literature_review.md``
    # §6.
    # F1.5b severity order: 8 < L=S < 3 < W=K < 1 < X.
    # Per Jorge's mapping of the quickcard Sw column: good-for-sawmill
    # codes (8, L, S, 3) sit above the "Generally Pulp Quality" line;
    # pulp-quality codes (W, K, 1, X) sit below. Within pulp quality,
    # severity increases W=K < 1 < X.
    severity = {
        "8": 0, "L": 1, "S": 1, "3": 2,
        "W": 3, "K": 3, "1": 4, "X": 5,
    }

    def _is_stable(zone: tuple) -> bool:
        s, e, c = zone
        # X and K are intrinsically short defect flags — always stable.
        if c == "X" or c == "K":
            return True
        return (e - s) >= min_zone_length_by_code.get(c, 0.0)

    stable_flags = [_is_stable(zn) for zn in zones]
    n_zones = len(zones)
    new_zones: List[tuple] = []
    k = 0
    while k < n_zones:
        if stable_flags[k]:
            new_zones.append(zones[k])
            k += 1
            continue
        # Maximal run of consecutive short zones: [k, j)
        j = k
        while j < n_zones and not stable_flags[j]:
            j += 1
        run_start = zones[k][0]
        run_end = zones[j - 1][1]
        # Anchors: nearest stable zone on each side (None at a boundary).
        candidates = []
        if k > 0:
            lc = zones[k - 1][2]
            candidates.append((severity.get(lc, 0), lc))
        if j < n_zones:
            rc = zones[j][2]
            candidates.append((severity.get(rc, 0), rc))
        if candidates:
            target = max(candidates, key=lambda t: t[0])[1]
        else:
            # No stable anchor anywhere (whole polyline sub-operational):
            # keep the worst code present in the run as a single zone.
            target = max(
                (severity.get(c, 0), c) for (_s, _e, c) in zones[k:j]
            )[1]
        new_zones.append((run_start, run_end, target))
        k = j

    # Final coalesce of adjacent same-code zones.
    coalesced: List[tuple] = []
    for (s, e, c) in new_zones:
        if coalesced and coalesced[-1][2] == c:
            s0, _, _ = coalesced[-1]
            coalesced[-1] = (s0, e, c)
        else:
            coalesced.append((s, e, c))

    return coalesced


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
