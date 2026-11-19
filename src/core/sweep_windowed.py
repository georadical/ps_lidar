"""
Window-based sweep-zone classifier (PlotSafe / HQP cruising semantics).

This is the F2 alternative to the global-chord classifier in
:func:`src.core.stem_description.classify_sweep_zones`. It encodes the
sweep-assessment rules of the PlotSafe procedures manual that the HQP
codes derive from:

    *PlotSafe Overlapping Feature Cruising Forest Inventory Procedures*,
    CNI Regional YTGEN User Group / Interpine Forestry, Feb 2007
    (``external_references/interpine/PLOTSAFE Overlapping Feature
    Cruising Forest Inventory Procedures - Feb 2007.pdf``), §4.1.4–4.1.4.6.

Rules taken from the manual
---------------------------
* Sweep is "proportional to the small end diameter (SED) of the length
  over which it is assessed" (p. 22): every test window is judged
  against the diameter at **its own top end**, not a per-tree scalar.
* "The assessment length (e.g. 6 m) must be applicable for the entire
  length of the sweep code and any adjacent higher coded feature
  lengths ... it must be possible to fit up to a 6 m log with SED/4
  anywhere throughout a 'S' shorts code, and any adjacent 'L' or '8'
  ... If not the sweep code will need to be split/downgraded
  appropriately, or a small section of kink or wobble placed in the
  stem to remove this region" (p. 23).
* Footnote 6 (p. 23): a code may apply to a stem length **shorter** than
  its assessment length (e.g. 3 m straight under a fork is still ``8``),
  so there are no per-code minimum zone lengths.
* "Describe each complete tree in sections using the highest value codes
  possible for the maximum length to which they can be applied" (p. 21);
  suggested steps push ``8`` out first, then ``L``, then ``S`` (p. 24).
* ``L`` vs ``S``: same sweep class, but ``L`` also fits long logs
  (8–12 m in RAD05). The decision is by **log-length feasibility**, not
  by counting direction reversals (p. 23).
* Kink: "a sharp out-of-line deviation ... call a kink for the length
  that kink occupies"; wobble: "two or more deviations ... along the
  length of the stem" (p. 24). Mild sinuosity that still fits short
  logs is *not* wobble (p. 23).
* Ignore stump swelling / butt flare < 0.3 m (p. 23); above the point
  where the stem diameter is < 150 mm the code need not change (p. 21).
* Lower-quality sections are never upgraded (p. 21); the HQP quickcard
  adds "only upgrade ... if > 3 m section between zones of lower
  quality".

Algorithm
---------
1. Sort nodes by z, lightly smooth x/y, and build a non-increasing
   diameter profile (isotonic fit) evaluated at every node.
2. Restrict the assessed domain to ``z ≥ butt_exclusion_height_m`` and
   to nodes whose diameter is ≥ ``min_diameter_m``.
3. Defect flags first: ``W`` where a wobble window shows two or more
   alternating lobes with peak-to-peak above the wobble floor (region
   ≤ ``w_max_length_m``); ``K`` where the turn angle between 1 m arms
   exceeds the threshold *and* the bend is sharp (the angle measured
   with half-length arms is nearly as large). Logs cannot include them,
   so they split the stem into independent segments.
4. Each segment is partitioned by dynamic programming into zones. A
   zone ``[s, t]`` may carry code *c* only if **every** *c* window
   inside it passes (its own chord, ``max sagitta / D(top) ≤ fraction``);
   a zone shorter than the window is judged as one truncated window.
   Between two adjacent log-quality zones, every window of the worse
   code straddling the junction must also pass (the "adjacent higher
   coded lengths" clause). A short ``K`` zone is allowed only at a sharp
   bend; ``X`` is always allowed. The partition minimises
   ``Σ length × severity + zone_penalty`` — the best codes for the
   longest lengths, with no amplitude attributed to individual nodes
   (which avoids the leakage that motivated the global chord in F1.1).
5. HQP upgrade rule, then coalesce to ``(z_start, z_end, code)`` zones
   with boundaries at node midpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ===========================================================================
# Dictionaries
# ===========================================================================

@dataclass(frozen=True)
class WindowTest:
    """One sweep test: max sagitta over ``length_m`` ≤ ``sed_fraction`` × D_top."""
    length_m: float
    sed_fraction: float


@dataclass(frozen=True)
class SweepClassSpec:
    """A sweep code and the window tests it must pass (all of them)."""
    code: str
    tests: Tuple[WindowTest, ...]


@dataclass(frozen=True)
class SweepDictionary:
    """Ordered (best → worst) test-based sweep classes plus defect-flag limits.

    ``K`` (kink), ``W`` (wobble) and ``X`` (severe) are defect flags and
    are produced by detection / segmentation, not by window tests.
    ``w_amplitude_floor_m`` is a peak-to-peak lateral swing, as drawn on
    the quickcard ("wobble tolerance measured" from a straight edge).
    """
    name: str
    classes: Tuple[SweepClassSpec, ...]
    w_amplitude_floor_m: float = 0.05
    w_window_m: float = 4.0
    w_max_length_m: float = 4.0
    k_max_length_m: float = 0.5


#: HQP LiDAR Feature Cruising quickcard (template LIR18-HQP, 2017–2018):
#: 8 = SED/8 over 6 m; L = SED/5 over any 6 m; S = SED/5 over any 4 m;
#: 3 = SED/3 over 4 m; 1 = SED/1; W = > 5 cm over 4 m; K = max 0.5 m.
#: The quickcard gives no window for ``1``; 4 m is assumed.
HQP_QUICKCARD = SweepDictionary(
    name="HQP LiDAR quickcard (LIR18-HQP)",
    classes=(
        SweepClassSpec("8", (WindowTest(6.0, 1.0 / 8.0),)),
        SweepClassSpec("L", (WindowTest(6.0, 1.0 / 5.0),)),
        SweepClassSpec("S", (WindowTest(4.0, 1.0 / 5.0),)),
        SweepClassSpec("3", (WindowTest(4.0, 1.0 / 3.0),)),
        SweepClassSpec("1", (WindowTest(4.0, 1.0),)),
    ),
    w_amplitude_floor_m=0.05,
    w_window_m=4.0,
    w_max_length_m=4.0,
    k_max_length_m=0.5,
)

#: PlotSafe RAD05 / RAD05-A / RAD06 Sw domain (manual pp. 24, 28):
#: 8 = SED/8 over 6.1 m; S = SED/4 over 6.1 m; L = same as S but
#: acceptable for 8–12 m logs ("SED 1/4 over 6 m is about the same as
#: SED 1/2 over 12 m", p. 23); 3 = SED/3 over 4–6.1 m (4 m assumed);
#: 1 = SED/1 (4 m assumed); W = > 5 cm over 6 m; K = max 0.5 m.
PLOTSAFE_RAD05 = SweepDictionary(
    name="PlotSafe RAD05 (Feb 2007)",
    classes=(
        SweepClassSpec("8", (WindowTest(6.1, 1.0 / 8.0),)),
        SweepClassSpec("L", (WindowTest(6.1, 1.0 / 4.0), WindowTest(12.0, 1.0 / 2.0))),
        SweepClassSpec("S", (WindowTest(6.1, 1.0 / 4.0),)),
        SweepClassSpec("3", (WindowTest(4.0, 1.0 / 3.0),)),
        SweepClassSpec("1", (WindowTest(4.0, 1.0),)),
    ),
    w_amplitude_floor_m=0.05,
    w_window_m=6.0,
    w_max_length_m=6.0,
    k_max_length_m=0.5,
)

#: Severity order from the quickcard: sawmill quality 8 < L < S < 3,
#: then "generally pulp quality" W = K < 1 < X. Also used as the
#: per-metre cost of a zone in the segmentation.
SEVERITY: Dict[str, int] = {
    "8": 0, "L": 1, "S": 2, "3": 3, "W": 4, "K": 4, "1": 5, "X": 6,
}

#: Defect flags: bounded-extent codes that split runs of log-quality codes.
DEFECT_CODES = frozenset({"K", "W", "X"})


@dataclass
class WindowedSweepConfig:
    """Parameters for :func:`classify_sweep_zones_windowed`.

    Attributes
    ----------
    dictionary : SweepDictionary
        Class definitions. Default :data:`HQP_QUICKCARD`.
    butt_exclusion_height_m : float, default 0.3
        Nodes below this height (above ground) are not assessed; the
        first assessed code is extended down to the base (manual p. 23).
    min_diameter_m : float, default 0.15
        Above the last node whose diameter is ≥ this value the code is
        not changed (manual p. 21). Ignored when fewer than 3 nodes
        qualify, so small stems are still classified.
    bark_thickness_m : float, default 0.0
        Single-bark thickness subtracted (× 2) from diameters before
        computing ``sagitta / D``. The manual does not state whether the
        SED is over or under bark; keep 0 until HQP confirms.
    upgrade_min_length_m : float, default 3.0
        HQP upgrade rule: a better zone between two worse zones is kept
        only if it is longer than this.
    zone_penalty_m : float, default 0.3
        Fixed cost per zone in the segmentation (in metre × severity
        units); discourages needless fragmentation.
    smooth_window_nodes : int, default 3
        Odd moving-average width applied to x/y before scoring
        (Gorges et al. 2021: 50–200 mm smoothing changes bow ≤ 4 %).
        ``1`` disables it. Kink angles use the unsmoothed centerline.
    k_angle_threshold_deg : float, default 15.0
        Sharp-bend angle (1 m arms) flagged ``K`` without further
        evidence. ``<= 0`` disables it.
    k_remedy_min_angle_deg : float, default 4.0
        Minimum sharp-bend angle at which the segmentation may place a
        short ``K`` zone to split an otherwise feasible stretch.
    k_sharpness_ratio : float, default 0.7
        A bend is *sharp* when the angle measured with half-length arms
        is at least this fraction of the angle with full arms (≈ 1 for a
        kink, ≈ 0.5 for a smooth constant-curvature bend).
    k_arm_m : float, default 1.0
        Arm length for the turn-angle test.
    k_min_arm_nodes : int, default 3
        Minimum nodes per arm (including the pivot).
    k_max_gap_factor : float, default 3.0
        An arm with a gap larger than this × median node spacing is
        treated as interpolated and cannot support a kink.
    w_lobe_min_m : float, default 0.02
        A wobble lobe counts only if its peak offset reaches this.
    w_lobe_bulge_m : float, default 0.01
        …and only if the stem inside it bulges at least this much from
        the line joining the lobe's ends (real curvature, not a straight
        stretch beside a tilted chord).
    w_end_margin_m : float, default 1.0
        Calm stem required on each side of the wobble lobes inside the
        detection window (which is ``w_window_m`` + 2 × this).
    w_end_max_angle_deg : float, default 2.0
        Maximum angle between the stem in each margin and the window
        chord for the window to count as anchored on calm stem.
    swing_deadband_m : float, default 0.01
        Offsets below this are ignored when delimiting lobes.
    """
    dictionary: SweepDictionary = field(default_factory=lambda: HQP_QUICKCARD)
    butt_exclusion_height_m: float = 0.3
    min_diameter_m: float = 0.15
    bark_thickness_m: float = 0.0
    upgrade_min_length_m: float = 3.0
    zone_penalty_m: float = 0.3
    smooth_window_nodes: int = 3
    k_angle_threshold_deg: float = 15.0
    k_remedy_min_angle_deg: float = 4.0
    k_sharpness_ratio: float = 0.7
    k_arm_m: float = 1.0
    k_min_arm_nodes: int = 3
    k_max_gap_factor: float = 3.0
    w_lobe_min_m: float = 0.02
    w_lobe_bulge_m: float = 0.01
    w_end_margin_m: float = 1.0
    w_end_max_angle_deg: float = 2.0
    swing_deadband_m: float = 0.01


# ===========================================================================
# Geometry helpers
# ===========================================================================

def _isotonic_nonincreasing(y: np.ndarray) -> np.ndarray:
    """Least-squares non-increasing fit (pool-adjacent-violators)."""
    means: List[float] = []
    weights: List[float] = []
    counts: List[int] = []
    for v in -np.asarray(y, dtype=np.float64):
        means.append(float(v))
        weights.append(1.0)
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, c2 = means.pop(), weights.pop(), counts.pop()
            m1, w1, c1 = means.pop(), weights.pop(), counts.pop()
            w = w1 + w2
            means.append((m1 * w1 + m2 * w2) / w)
            weights.append(w)
            counts.append(c1 + c2)
    return -np.repeat(np.asarray(means), np.asarray(counts))


def _node_diameters(
    z: np.ndarray,
    diameter_profile: Optional[Tuple[np.ndarray, np.ndarray]],
    sed_m: Optional[float],
) -> Optional[np.ndarray]:
    """Smoothed, non-increasing diameter (m) at every node, or ``None``."""
    if diameter_profile is not None:
        zp = np.asarray(diameter_profile[0], dtype=np.float64).ravel()
        dp = np.asarray(diameter_profile[1], dtype=np.float64).ravel()
        ok = np.isfinite(zp) & np.isfinite(dp) & (dp > 0.0)
        zp, dp = zp[ok], dp[ok]
        if zp.size >= 1:
            order = np.argsort(zp)
            zp, dp = zp[order], dp[order]
            dp = _isotonic_nonincreasing(dp)
            return np.interp(z, zp, dp)
    if sed_m is not None and sed_m > 0.0:
        return np.full(z.shape, float(sed_m))
    return None


def _smooth_xy(cl: np.ndarray, window: int) -> np.ndarray:
    """Moving average of x and y (edge-replicated); z untouched."""
    if window <= 1 or cl.shape[0] < window:
        return cl
    pad = window // 2
    out = cl.copy()
    kernel = np.ones(window) / window
    for col in (0, 1):
        padded = np.pad(cl[:, col], pad, mode="edge")
        out[:, col] = np.convolve(padded, kernel, mode="valid")
    return out


def _chord_offsets(cl: np.ndarray, i: int, e: int) -> np.ndarray:
    """Perpendicular offset vectors of nodes ``i..e`` from chord ``cl[i]→cl[e]``."""
    seg = cl[i:e + 1]
    p0 = cl[i]
    axis = cl[e] - p0
    l2 = float(axis @ axis)
    if l2 < 1e-18:
        return np.zeros_like(seg)
    t = ((seg - p0) @ axis) / l2
    return seg - (p0 + t[:, None] * axis)


def _signed_profile(off: np.ndarray) -> np.ndarray:
    """Project offset vectors on their dominant direction (signed 1D)."""
    if off.shape[0] == 0 or not np.any(off):
        return np.zeros(off.shape[0])
    _, _, vt = np.linalg.svd(off, full_matrices=False)
    return off @ vt[0]


def _lobes(s: np.ndarray, deadband: float, lobe_min: float) -> List[Tuple[int, int, int]]:
    """Alternating lobes ``(start, end, sign)`` of ``s`` with peak ≥ ``lobe_min``.

    A lobe is a maximal run with ``|s| > deadband`` and constant sign;
    lobes below ``lobe_min`` are dropped and consecutive same-sign lobes
    are merged.
    """
    raw: List[Tuple[int, int, int, float]] = []
    for k, v in enumerate(s):
        sg = 1 if v > deadband else (-1 if v < -deadband else 0)
        if sg == 0:
            continue
        if raw and raw[-1][2] == sg and raw[-1][1] == k - 1:
            a, _, _, pk = raw[-1]
            raw[-1] = (a, k, sg, max(pk, abs(v)))
        else:
            raw.append((k, k, sg, abs(v)))
    out: List[Tuple[int, int, int]] = []
    for a, b, sg, pk in raw:
        if pk < lobe_min:
            continue
        if out and out[-1][2] == sg:
            out[-1] = (out[-1][0], b, sg)
        else:
            out.append((a, b, sg))
    return out


def _enumerate_windows(z: np.ndarray, length_m: float, tol: float) -> List[Tuple[int, int]]:
    """Windows ``(i, e)`` of ≈ ``length_m`` starting at every node.

    ``e`` is the first node with ``z[e] ≥ z[i] + length_m − tol``.
    Returns an empty list when the stretch is shorter than the window.
    """
    n = z.size
    out: List[Tuple[int, int]] = []
    for i in range(n):
        e = int(np.searchsorted(z, z[i] + length_m - tol, side="left"))
        if e >= n:
            break
        out.append((i, e))
    return out


def _arm_direction(pts: np.ndarray) -> Optional[np.ndarray]:
    """Unit direction (oriented +z) of the best-fit line through ``pts``."""
    if pts.shape[0] < 2:
        return None
    c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    v = vt[0]
    return -v if v[2] < 0 else v


def _turn_angles_deg(cl: np.ndarray, arm_m: float, min_nodes: int, max_gap: float) -> np.ndarray:
    """Angle between line fits on ``[z−arm, z]`` and ``[z, z+arm]`` per node.

    ``NaN`` where an arm has too few nodes or spans an interpolated gap.
    """
    z = cl[:, 2]
    n = z.size
    ang = np.full(n, np.nan)
    for j in range(n):
        lo = int(np.searchsorted(z, z[j] - arm_m - 1e-9, side="left"))
        hi = int(np.searchsorted(z, z[j] + arm_m + 1e-9, side="right"))
        below, above = cl[lo:j + 1], cl[j:hi]
        if below.shape[0] < min_nodes or above.shape[0] < min_nodes:
            continue
        if (np.diff(below[:, 2]).max() > max_gap) or (np.diff(above[:, 2]).max() > max_gap):
            continue
        u, v = _arm_direction(below), _arm_direction(above)
        if u is None or v is None:
            continue
        ang[j] = float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))
    return ang


def _max_gap(z: np.ndarray, cfg: "WindowedSweepConfig") -> float:
    dz = np.diff(z)
    med = float(np.median(dz)) if dz.size else 0.2
    return cfg.k_max_gap_factor * max(med, 1e-6)


def _sharp_bend_angles(cl: np.ndarray, cfg: "WindowedSweepConfig") -> np.ndarray:
    """Turn angle (full arms) where the bend is sharp, else 0.

    Sharpness: for a kink the angle barely changes when the arms are
    halved; for a smooth bend of constant curvature it halves.
    """
    gap = _max_gap(cl[:, 2], cfg)
    full = _turn_angles_deg(cl, cfg.k_arm_m, cfg.k_min_arm_nodes, gap)
    half = _turn_angles_deg(cl, 0.5 * cfg.k_arm_m, cfg.k_min_arm_nodes, gap)
    with np.errstate(invalid="ignore"):
        sharp = np.isfinite(full) & np.isfinite(half) & (half >= cfg.k_sharpness_ratio * full)
    return np.where(sharp, full, 0.0)


def _kink_span(z: np.ndarray, pivot: int, k_max_length_m: float) -> np.ndarray:
    """Boolean mask of a ``K`` section centred on ``pivot``, ≤ ``k_max_length_m``.

    Zone boundaries sit at node midpoints, so a node is included only if
    its whole cell (± half spacing) fits in the allowed half-length. The
    pivot itself is always included.
    """
    dz = np.diff(z)
    half_cell = 0.5 * float(np.median(dz)) if dz.size else 0.0
    reach = max(0.5 * k_max_length_m - half_cell, 0.0) + 1e-9
    mask = np.abs(z - z[pivot]) <= reach
    mask[pivot] = True
    return mask


def _cell_bounds(z: np.ndarray) -> np.ndarray:
    """Zone boundaries: ``z[0]``, node midpoints, ``z[-1]`` (length n + 1)."""
    b = np.empty(z.size + 1)
    b[0], b[-1] = z[0], z[-1]
    b[1:-1] = 0.5 * (z[:-1] + z[1:])
    return b


# ===========================================================================
# Defect detection
# ===========================================================================

def _curved_lobes(
    cl: np.ndarray, i: int, lobes: List[Tuple[int, int, int]], bulge_min: float,
) -> List[Tuple[int, int, int]]:
    """Keep lobes that actually bend, then re-merge same-sign neighbours.

    A lobe is real curvature only if the stem inside it bulges at least
    ``bulge_min`` away from the line joining the lobe's own ends; a
    straight stretch that merely sits on one side of a tilted chord
    does not.
    """
    out: List[Tuple[int, int, int]] = []
    for a, b, sg in lobes:
        if b - a < 2:
            continue
        bulge = float(np.linalg.norm(_chord_offsets(cl, i + a, i + b), axis=1).max())
        if bulge < bulge_min:
            continue
        if out and out[-1][2] == sg:
            out[-1] = (out[-1][0], b, sg)
        else:
            out.append((a, b, sg))
    return out


def _detect_wobble(cl: np.ndarray, cfg: "WindowedSweepConfig", tol: float) -> np.ndarray:
    """``W`` mask: ≥ 2 alternating lobes, peak-to-peak ≥ floor, region ≤ max.

    The detection window is the wobble window plus ``w_end_margin_m`` on
    each side. A genuine wobble sits inside calm stem that returns to
    its axis, so (i) the stem in both margins must run parallel to the
    window chord (within ``w_end_max_angle_deg``) and (ii) the lobes must
    stay clear of both margins. A window that starts or ends on a single
    bow fails (i): its chord is tilted and makes bow + straight look
    like two alternating lobes.
    """
    dic = cfg.dictionary
    z = cl[:, 2]
    n = z.size
    margin = cfg.w_end_margin_m
    cos_max = np.cos(np.radians(cfg.w_end_max_angle_deg))
    cands: List[Tuple[int, int, float]] = []   # (region_start, region_end, p2p)
    for i, e in _enumerate_windows(z, dic.w_window_m + 2.0 * margin, tol):
        if e - i < 4:
            continue
        chord = cl[e] - cl[i]
        chord = chord / np.linalg.norm(chord)
        i_m = int(np.searchsorted(z, z[i] + margin + 1e-9, side="right"))
        e_m = int(np.searchsorted(z, z[e] - margin - 1e-9, side="left"))
        ends = (_arm_direction(cl[i:i_m]), _arm_direction(cl[e_m:e + 1]))
        if any(u is None or abs(float(u @ chord)) < cos_max for u in ends):
            continue
        s = _signed_profile(_chord_offsets(cl, i, e))
        lobes = _curved_lobes(cl, i, _lobes(s, cfg.swing_deadband_m, cfg.w_lobe_min_m),
                              cfg.w_lobe_bulge_m)
        if len(lobes) < 2:
            continue
        a, b = i + lobes[0][0], i + lobes[-1][1]
        if z[a] < z[i] + margin - tol or z[b] > z[e] - margin + tol:
            continue
        p2p = float(s.max() - s.min())
        if p2p >= dic.w_amplitude_floor_m:
            cands.append((a, b, p2p))
    mask = np.zeros(n, dtype=bool)
    # cluster overlapping candidates; keep the strongest one's lobe span
    cands.sort()
    k = 0
    while k < len(cands):
        cluster = [cands[k]]
        end = cands[k][1]
        k += 1
        while k < len(cands) and cands[k][0] <= end:
            cluster.append(cands[k])
            end = max(end, cands[k][1])
            k += 1
        a, b, _ = max(cluster, key=lambda c: c[2])
        if z[b] - z[a] <= dic.w_max_length_m:
            mask[a:b + 1] = True
    return mask


def _detect_sharp_kinks(z: np.ndarray, cfg: "WindowedSweepConfig", sharp: np.ndarray) -> np.ndarray:
    """``K`` mask: sharp bends above ``k_angle_threshold_deg`` (NMS over one arm)."""
    mask = np.zeros(z.size, dtype=bool)
    if cfg.k_angle_threshold_deg <= 0.0:
        return mask
    taken: List[int] = []
    for j in sorted(np.where(sharp >= cfg.k_angle_threshold_deg)[0], key=lambda k: -sharp[k]):
        if any(abs(z[j] - z[t]) <= cfg.k_arm_m for t in taken):
            continue
        taken.append(int(j))
        mask |= _kink_span(z, int(j), cfg.dictionary.k_max_length_m)
    return mask


# ===========================================================================
# Segmentation (dynamic programming over feasible zones)
# ===========================================================================

class _Ratio:
    """Cached ``max sagitta / D(top)`` for node spans of one segment."""

    def __init__(self, cl: np.ndarray, d: np.ndarray):
        self.cl = cl
        self.d = d
        self.cache: Dict[Tuple[int, int], float] = {}

    def __call__(self, s: int, t: int) -> float:
        key = (s, t)
        r = self.cache.get(key)
        if r is None:
            if t - s < 2:
                r = 0.0
            else:
                dist = np.linalg.norm(_chord_offsets(self.cl, s, t), axis=1)
                r = float(dist.max()) / max(float(self.d[t]), 1e-6)
            self.cache[key] = r
        return r


@dataclass
class _TestTable:
    length_m: float
    fraction: float
    short_segment: bool                       # segment shorter than the window
    segment_ok: bool                          # truncated whole-segment window passes
    fail_end: List[int]                       # min e of failing windows with i ≥ s
    fail_reach: List[int]                     # max e of failing windows with i ≤ s
    straddle: List[List[Tuple[int, int]]]     # failing windows with i < s ≤ e


def _test_table(z: np.ndarray, ratio: _Ratio, test: WindowTest, tol: float) -> _TestTable:
    m = z.size
    fail_end = [m] * (m + 1)
    fail_reach = [-1] * m
    straddle: List[List[Tuple[int, int]]] = [[] for _ in range(m + 1)]
    windows = _enumerate_windows(z, test.length_m, tol)
    for i, e in windows:
        if ratio(i, e) > test.sed_fraction:
            fail_end[i] = min(fail_end[i], e)
            fail_reach[i] = max(fail_reach[i], e)
            for s in range(i + 1, e + 1):
                straddle[s].append((i, e))
    for s in range(m - 1, -1, -1):
        fail_end[s] = min(fail_end[s], fail_end[s + 1])
    for s in range(1, m):
        fail_reach[s] = max(fail_reach[s], fail_reach[s - 1])
    short = not windows
    return _TestTable(
        test.length_m, test.sed_fraction, short,
        (ratio(0, m - 1) <= test.sed_fraction) if short else True,
        fail_end, fail_reach, straddle,
    )


def _segment_dp(
    cl: np.ndarray,
    d: np.ndarray,
    bounds: np.ndarray,
    sharp: np.ndarray,
    cfg: WindowedSweepConfig,
    tol: float,
) -> List[str]:
    """Optimal partition of one defect-free stretch into feasible zones."""
    z = cl[:, 2]
    m = z.size
    ratio = _Ratio(cl, d)
    specs = {spec.code: spec for spec in cfg.dictionary.classes}
    tables = {
        code: [_test_table(z, ratio, t, tol) for t in spec.tests]
        for code, spec in specs.items()
    }
    codes = list(specs) + ["K", "X"]
    k_max = cfg.dictionary.k_max_length_m + 1e-9

    def length(s: int, t: int) -> float:
        return float(bounds[t + 1] - bounds[s])

    def class_status(code: str, s: int, t: int) -> int:
        """1 feasible, 0 infeasible, -1 infeasible for every smaller s too.

        * Segment shorter than the window (stem ends at a fork, the top,
          the base or a defect): the truncated whole-segment window must
          pass — "up to" the assessment length (manual footnote 6).
        * Zone at least one window long: every window inside it passes.
        * Shorter zone inside a longer segment: every full window that
          contains it passes ("anywhere"), so a stem cannot be chopped
          into sub-window pieces to dodge the test.
        """
        for tab in tables[code]:
            if tab.short_segment:
                if not tab.segment_ok:
                    return -1
            elif z[t] - z[s] >= tab.length_m - tol:
                if t >= tab.fail_end[s]:
                    return -1
            elif tab.fail_reach[s] >= t:
                return 0
        return 1

    def junction_ok(prev: str, code: str, s_prev: int, s: int, t: int) -> bool:
        if prev in DEFECT_CODES or code in DEFECT_CODES:
            return True
        worse = prev if SEVERITY[prev] > SEVERITY[code] else code
        for tab in tables[worse]:
            for i, e in tab.straddle[s]:
                if i >= s_prev and e <= t:
                    return False
        return True

    inf = float("inf")
    best = [[inf] * len(codes) for _ in range(m)]
    back: List[List[Optional[Tuple[int, int]]]] = [[None] * len(codes) for _ in range(m)]

    for t in range(m):
        for ci, code in enumerate(codes):
            for s in range(t, -1, -1):
                if code == "K":
                    if length(s, t) > k_max:
                        break
                    if float(np.max(sharp[s:t + 1])) < cfg.k_remedy_min_angle_deg:
                        continue
                elif code != "X":
                    st = class_status(code, s, t)
                    if st < 0:
                        break
                    if st == 0:
                        continue
                cost = length(s, t) * SEVERITY[code] + cfg.zone_penalty_m
                if s == 0:
                    if cost < best[t][ci]:
                        best[t][ci] = cost
                        back[t][ci] = (0, -1)
                    continue
                for pi, prev in enumerate(codes):
                    if prev == code or best[s - 1][pi] == inf:
                        continue
                    s_prev = back[s - 1][pi][0]  # type: ignore[index]
                    if not junction_ok(prev, code, s_prev, s, t):
                        continue
                    total = best[s - 1][pi] + cost
                    if total < best[t][ci]:
                        best[t][ci] = total
                        back[t][ci] = (s, pi)

    labels = [""] * m
    t = m - 1
    ci = int(np.argmin(best[t]))
    while t >= 0:
        s, pi = back[t][ci]  # type: ignore[misc]
        for k in range(s, t + 1):
            labels[k] = codes[ci]
        t, ci = s - 1, pi
    return labels


# ===========================================================================
# Post-processing
# ===========================================================================

def _label_runs(labels: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Inclusive ``(start, end, code)`` runs of identical labels."""
    out: List[Tuple[int, int, str]] = []
    for k, c in enumerate(labels):
        if out and out[-1][2] == c:
            out[-1] = (out[-1][0], k, c)
        else:
            out.append((k, k, c))
    return out


def _apply_upgrade_rule(z: np.ndarray, labels: List[str], min_len: float) -> List[str]:
    """HQP rule: a better zone ≤ ``min_len`` between two worse zones is downgraded.

    Defect flags (K, W, X) are looked past when finding the neighbours,
    so a point defect does not by itself force a downgrade, and the
    replacement is the less severe of the two worse neighbours.
    """
    labels = list(labels)
    bounds = _cell_bounds(z)
    for _ in range(len(labels) + 1):
        runs = _label_runs(labels)
        changed = False
        for r, (s, t, code) in enumerate(runs):
            if code in DEFECT_CODES or bounds[t + 1] - bounds[s] > min_len:
                continue
            left = next((c for (_, _, c) in reversed(runs[:r]) if c not in DEFECT_CODES), None)
            right = next((c for (_, _, c) in runs[r + 1:] if c not in DEFECT_CODES), None)
            if left is None or right is None:
                continue
            if SEVERITY[left] > SEVERITY[code] and SEVERITY[right] > SEVERITY[code]:
                new = left if SEVERITY[left] <= SEVERITY[right] else right
                for k in range(s, t + 1):
                    labels[k] = new
                changed = True
                break
        if not changed:
            break
    return labels


def _labels_to_zones(z: np.ndarray, labels: Sequence[str]) -> List[Tuple[float, float, str]]:
    bounds = _cell_bounds(z)
    return [(float(bounds[s]), float(bounds[t + 1]), code) for s, t, code in _label_runs(labels)]


# ===========================================================================
# Public API
# ===========================================================================

def classify_sweep_zones_windowed(
    centerline: Optional[np.ndarray],
    diameter_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    sed_m: Optional[float] = None,
    config: Optional[WindowedSweepConfig] = None,
) -> List[Tuple[float, float, str]]:
    """Decompose a stem centerline into sweep zones with window tests.

    Parameters
    ----------
    centerline : ndarray (N, 3) or None
        Section centres ``(x, y, z)`` with ``z`` = height above ground.
    diameter_profile : (z, d) arrays, optional
        Per-section diameters in metres (e.g. the valid sections of the
        tree). Smoothed to a non-increasing profile and evaluated at the
        top node of every window.
    sed_m : float, optional
        Fallback constant diameter when no profile is available.
    config : WindowedSweepConfig, optional

    Returns
    -------
    list of (z_start, z_end, code)
        Contiguous zones sorted by height, same format as
        :func:`src.core.stem_description.classify_sweep_zones`.
        In PlotSafe, ``Position`` is the zone's ``z_end`` (top).
    """
    cfg = config or WindowedSweepConfig()
    if centerline is None:
        return []
    cl_raw = np.asarray(centerline, dtype=np.float64)
    if cl_raw.ndim != 2 or cl_raw.shape[1] != 3 or cl_raw.shape[0] < 3:
        return []
    cl_raw = cl_raw[np.argsort(cl_raw[:, 2], kind="stable")]
    z_all = cl_raw[:, 2]
    if z_all[-1] - z_all[0] <= 0.0:
        return []
    d_all = _node_diameters(z_all, diameter_profile, sed_m)
    if d_all is None:
        return []

    # --- assessed domain ---
    # Butt exclusion: nodes below the stump-swelling height are ignored.
    # 150 mm top: the code is frozen above the last node with D ≥
    # min_diameter_m ("need not be changed above this point"), but the
    # stem above still exists, so windows keep using it — with D clamped
    # at the cut so thin tops do not make the windows below artificially
    # strict.
    n_all = z_all.size
    lo = int(np.searchsorted(z_all, cfg.butt_exclusion_height_m, side="left"))
    if lo > n_all - 3:
        lo = 0
    big = np.where(d_all >= cfg.min_diameter_m)[0]
    hi = int(big.max()) if big.size and big.max() - lo >= 2 else n_all - 1
    # smooth only inside the domain so excluded butt flare cannot leak in
    cl = _smooth_xy(cl_raw[lo:], cfg.smooth_window_nodes)
    z = cl[:, 2]
    d_eval = d_all[lo:].copy()
    d_eval[hi - lo + 1:] = d_eval[hi - lo]
    d_eval = np.maximum(d_eval - 2.0 * cfg.bark_thickness_m, 1e-6)
    dz = np.diff(z)
    tol = 0.5 * float(np.median(dz)) if dz.size else 0.0
    bounds = _cell_bounds(z)

    # --- defect flags first (a log cannot include them) ---
    sharp = _sharp_bend_angles(cl_raw[lo:], cfg)
    w_mask = _detect_wobble(cl, cfg, tol)
    k_mask = _detect_sharp_kinks(z, cfg, sharp)
    labels: List[str] = ["K" if k else ("W" if w else "") for k, w in zip(k_mask, w_mask)]

    # --- optimal feasible partition of each defect-free stretch ---
    k = 0
    while k < z.size:
        if labels[k]:
            k += 1
            continue
        e = k
        while e + 1 < z.size and not labels[e + 1]:
            e += 1
        seg = _segment_dp(cl[k:e + 1], d_eval[k:e + 1], bounds[k:e + 2], sharp[k:e + 1], cfg, tol)
        labels[k:e + 1] = seg
        k = e + 1

    top = hi - lo
    labels[top + 1:] = [labels[top]] * (len(labels) - top - 1)
    labels = _apply_upgrade_rule(z, labels, cfg.upgrade_min_length_m)

    full = [labels[0]] * lo + labels
    return _labels_to_zones(z_all, full)
