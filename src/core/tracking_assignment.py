"""
Tracking-based tree_id assignment (Mejora 1, Phase 1B real — GS block).

This module replaces the straight-cylinder propagation in
``src.core.trunk_extraction`` with a fundamentally different mechanism
for assigning ``tree_id`` to LiDAR points:

  1. Filter the whole cloud by verticality (GS.1, this file) so the
     subsequent steps see only stem-like material — no ground, no
     understory, no foliage.
  2. Slice the filtered cloud horizontally at coarse spacing (GS.2)
     — only fine enough to capture inclination and curvature for
     assignment purposes, not for final dendrometric precision.
  3. DBSCAN per slice (GS.3) → cluster candidates that may represent
     stem cross-sections at that height.
  4. RANSAC-ellipse fit per cluster (GS.4) using the ``ellipse_fitting``
     primitives. Bad fits get the cluster discarded.
  5. Vertical tracking: greedy matching of cluster centres between
     adjacent slices by XY proximity (GS.5).
  6. Bootstrap tracks from the basal stripe (GS.6) so spurious
     mid-canopy clusters cannot start their own tree.
  7. Assign ``tree_id`` and ``stem_id`` from the tracks (GS.7) —
     the **principal visual gate** of Phase 1B. The bifurcation rule
     follows the PlotSafe field-data schema: a stem fork promotes the
     larger child to ``stem_id=0`` of the same ``tree_id``, and the
     smaller child starts ``stem_id=1`` whose DBH is measured at
     1.30 m from the bifurcation point.
  8. Orchestrator + config flag (GS.8) so downstream callers can
     choose between the legacy straight-cylinder path and this one.
  9. Integration with ``clean_stems`` and ``compute_stem_sections``
     downstream (GS.9 / GS.9a).

Each sub-fase is committed atomically; visual validation in CloudCompare
on the ``trunks_validated`` checkpoint is the final acceptance gate
(retrieves flanks that the straight cylinder loses on inclined / curved
stems).

This file currently implements only GS.1. The rest land sub-fase by
sub-fase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.core.ellipse_fitting import EllipseFitConfig, _fit_ellipse_check
from src.core.features import compute_verticality_mask_early_exit


# ===========================================================================
# GS.1 — Filter the cloud by verticality (pre-clustering)
# ===========================================================================

def filter_by_verticality(
    xyz: np.ndarray,
    threshold: float = 0.7,
    scale: float = 0.10,
    voxel_resolution_xy: float = 0.05,
    voxel_resolution_z: float = 0.05,
    coarse_resolution: float = 0.10,
    margin: float = 0.10,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return only points whose voxel is sufficiently vertical.

    Thin convenience wrapper around
    :func:`src.core.features.compute_verticality_mask_early_exit`,
    tailored to the Phase 1B context. The default thresholds
    (``threshold=0.7``, ``voxel_resolution=0.05 m``) mirror the
    "primary verticality pass" defaults of ``TrunkExtractionConfig``
    so behaviour is consistent with the legacy pipeline.

    The two-tier early-exit screening is fast enough to run on the
    **full** cloud (not just a basal stripe), which is precisely what
    the tracking-based assignment needs: the upper portion of inclined
    or curved stems cannot be reached by a stripe-based filter.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        Height-normalised vegetation point cloud (typically the same
        ``veg_normalized`` array fed into ``extract_trunks``).
    threshold : float, default 0.7
        Verticality threshold in ``[0, 1]``. Higher → stricter
        rejection of non-vertical material (typical stem material has
        verticality > 0.7).
    scale : float, default 0.10
        Radius (m) used for PCA inside ``pgeof``. Matches
        ``TrunkExtractionConfig.verticality_scale`` default.
    voxel_resolution_xy, voxel_resolution_z : float, default 0.05
        Fine-tier voxel size used in the second pass.
    coarse_resolution : float, default 0.10
        Coarse-tier voxel size used in the first pass (auto-keep).
    margin : float, default 0.10
        Auto-keep margin above ``threshold`` for the coarse pass.
    verbose : bool, default False
        Forwarded to ``pgeof`` so progress prints to stdout when on.

    Returns
    -------
    (filtered_xyz, keep_mask) : tuple
        ``filtered_xyz`` is the subset of points that passed the filter
        (shape ``(K, 3)`` with ``K = keep_mask.sum()``). ``keep_mask``
        is a boolean array of shape ``(N,)`` aligned with ``xyz``;
        ``filtered_xyz == xyz[keep_mask]``.

    Raises
    ------
    ValueError
        If ``xyz`` is not 2D with three columns, or ``threshold`` is
        outside ``[0, 1]``.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3); got {xyz.shape}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be in [0, 1]; got {threshold}"
        )

    keep_mask, _stats = compute_verticality_mask_early_exit(
        xyz,
        threshold=threshold,
        scale=scale,
        voxel_resolution_xy=voxel_resolution_xy,
        voxel_resolution_z=voxel_resolution_z,
        coarse_resolution=coarse_resolution,
        margin=margin,
    )

    return xyz[keep_mask], keep_mask


# ===========================================================================
# GS.2 — Horizontal slicing on coarse spacing
# ===========================================================================

@dataclass(frozen=True)
class HorizontalSlice:
    """A single horizontal slice produced by :func:`slice_horizontal_global`.

    The ``indices`` array points into the **input** point cloud passed
    to ``slice_horizontal_global``. It is convenient to keep these as
    indices (not as a sub-array of XYZ) because downstream stages
    (clustering, ellipse fitting) only need positions, while the final
    ``tree_id`` assignment in GS.7 must scatter results back to the
    full cloud — and indices let us do that in O(1) per point.
    """
    z_centre: float        # nominal centre height of the slice (m)
    z_low: float           # inclusive lower bound (m)
    z_high: float          # exclusive upper bound (m)
    indices: np.ndarray    # (K,) int — positions in the input cloud


def slice_horizontal_global(
    xyz: np.ndarray,
    z_min: float = 0.3,
    z_max: float = 40.0,
    slab_step: float = 1.0,
    slab_half_thickness: float = 0.5,
) -> List[HorizontalSlice]:
    """Slice a point cloud into horizontal slabs at coarse spacing.

    Each slab is a band of height ``2 · slab_half_thickness`` centred at
    a target z. Slab centres are generated by ``np.arange(z_min, z_max,
    slab_step)``. Slabs may overlap or leave gaps depending on the
    relative magnitudes of ``slab_step`` and ``slab_half_thickness``:

      - ``half == step / 2`` → slabs touch at boundaries (no gap, no
        overlap; each point lies in exactly one slab unless it sits
        exactly on a boundary).
      - ``half > step / 2`` → slabs overlap; a point may appear in
        multiple slabs.
      - ``half < step / 2`` → slabs are disjoint with gaps between
        them; some points fall outside every slab.

    The defaults (``step=1.0``, ``half=0.5``) implement the canonical
    no-gap-no-overlap configuration recommended for the Phase 1B
    tracking pipeline: coarse enough for the assignment job, fine
    enough that tracking between adjacent slabs is well-defined.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        Point cloud (typically the verticality-filtered output of
        :func:`filter_by_verticality`).
    z_min : float, default 0.3
        Bottom of the useful stem range (m). Slab centres start at or
        just above this value.
    z_max : float, default 40.0
        Top of the useful stem range (m). Slab centres stop strictly
        below this value (``np.arange`` convention).
    slab_step : float, default 1.0
        Vertical spacing between adjacent slab centres (m). Must be
        positive.
    slab_half_thickness : float, default 0.5
        Half-thickness of each slab (m). Must be positive.

    Returns
    -------
    list of HorizontalSlice
        One entry per slab centre in ``np.arange(z_min, z_max, slab_step)``.
        Slabs whose membership is empty are **still returned** (with
        ``indices.size == 0``) so the caller can reason about coverage
        gaps. Downstream stages should skip empty slabs explicitly.

    Raises
    ------
    ValueError
        If ``xyz`` is not 2D with three columns, ``z_min >= z_max``,
        ``slab_step <= 0``, or ``slab_half_thickness <= 0``.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3); got {xyz.shape}")
    if not z_min < z_max:
        raise ValueError(f"z_min must be < z_max; got {z_min}, {z_max}")
    if slab_step <= 0.0:
        raise ValueError(f"slab_step must be positive; got {slab_step}")
    if slab_half_thickness <= 0.0:
        raise ValueError(
            f"slab_half_thickness must be positive; got {slab_half_thickness}"
        )

    z = xyz[:, 2]
    centres = np.arange(z_min, z_max, slab_step, dtype=np.float64)

    slices: List[HorizontalSlice] = []
    for z_centre in centres:
        z_low = float(z_centre - slab_half_thickness)
        z_high = float(z_centre + slab_half_thickness)
        # Inclusive low, exclusive high — matches np.arange semantics
        # and keeps points strictly on a boundary in at most one slab
        # when half == step / 2.
        mask = (z >= z_low) & (z < z_high)
        idx = np.where(mask)[0]
        slices.append(HorizontalSlice(
            z_centre=float(z_centre),
            z_low=z_low,
            z_high=z_high,
            indices=idx,
        ))

    return slices


# ===========================================================================
# GS.3 — DBSCAN per slice
# ===========================================================================

@dataclass(frozen=True)
class Cluster2D:
    """A 2D cluster within a single horizontal slice.

    The ``indices`` array points back into the **original** input cloud
    passed all the way down from :func:`filter_by_verticality` /
    :func:`slice_horizontal_global` — not into a slice-local subset.
    This is the same convention as :class:`HorizontalSlice` and lets
    the GS.7 assignment scatter results to the full cloud in one pass.
    """
    indices: np.ndarray        # (K,) int — positions in the input cloud
    centroid_xy: np.ndarray    # (2,) float — XY centroid of the cluster
    n_points: int              # convenience (== indices.size)


def cluster_slice(
    xyz: np.ndarray,
    slice_obj: HorizontalSlice,
    eps: float = 0.10,
    min_samples: int = 10,
) -> List[Cluster2D]:
    """Run DBSCAN on the XY positions of points belonging to a slice.

    The horizontal slice from :func:`slice_horizontal_global` carries
    indices into the full cloud; this function reads the XY positions
    of those points (ignoring z, since the slice already constrains it)
    and groups them into 2D clusters with DBSCAN.

    Noise points (label ``-1``) are dropped. Each returned cluster
    carries an index array pointing back into the original cloud, so
    no slice-local renumbering ever leaks downstream.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        The full input point cloud (typically the verticality-filtered
        output of :func:`filter_by_verticality`).
    slice_obj : HorizontalSlice
        One slice from :func:`slice_horizontal_global`.
    eps : float, default 0.10
        DBSCAN ``eps`` in metres. 10 cm separates stems in a typical
        plantation; tighten on dense plots, loosen on sparse ones.
        HDBSCAN as an alternative is parked for the future per the
        Phase 1B plan.
    min_samples : int, default 10
        DBSCAN ``min_samples``. Below this the point becomes noise.

    Returns
    -------
    list of Cluster2D
        One entry per non-noise cluster, in DBSCAN label order
        (no implicit sort by size). Empty if the slice is empty or
        the data contains only noise.

    Raises
    ------
    ValueError
        If ``xyz`` is not 2D with three columns, or ``eps`` /
        ``min_samples`` are not positive.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3); got {xyz.shape}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1; got {min_samples}")

    if slice_obj.indices.size == 0:
        return []

    pts_xy = xyz[slice_obj.indices, :2]

    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_xy)

    clusters: List[Cluster2D] = []
    for label in np.unique(labels):
        if label == -1:
            continue
        mask = labels == label
        cluster_indices = slice_obj.indices[mask]
        centroid = pts_xy[mask].mean(axis=0)
        clusters.append(Cluster2D(
            indices=cluster_indices,
            centroid_xy=centroid.astype(np.float64),
            n_points=int(mask.sum()),
        ))

    return clusters


# ===========================================================================
# GS.4 — Fit RANSAC ellipse per cluster
# ===========================================================================

@dataclass(frozen=True)
class ClusterEllipse:
    """A 2D cluster that has been successfully fitted with an ellipse.

    Produced by :func:`fit_ellipses_in_slice` from a :class:`Cluster2D`
    plus the EL.5 ``_fit_ellipse_check`` output. The ``indices`` field
    still points into the **original** input cloud — convention shared
    with :class:`HorizontalSlice` and :class:`Cluster2D` throughout the
    GS block.

    Convention: ``a ≥ b`` (semi-major ≥ semi-minor); ``theta`` is the
    rotation in radians from +x to the semi-major axis. All length-like
    fields are in the same units as the input XYZ (typically metres).
    """
    indices: np.ndarray        # (K,) int — full-cloud indices
    xc: float                  # ellipse centre X
    yc: float                  # ellipse centre Y
    a: float                   # semi-major axis
    b: float                   # semi-minor axis
    theta: float               # orientation (radians)
    sector_pct: float          # quality: % of sectors occupied around the curve
    check_status: int          # EL.5 status: 0 = first-attempt fit, 1 = retried
    n_points: int              # cluster size fed to RANSAC


def fit_ellipses_in_slice(
    xyz: np.ndarray,
    clusters: List[Cluster2D],
    config: EllipseFitConfig,
    rng: Optional[np.random.Generator] = None,
) -> List[ClusterEllipse]:
    """Fit a RANSAC ellipse to each cluster; keep only successful fits.

    Calls :func:`_fit_ellipse_check` (EL.5) on the XY positions of each
    cluster's points. The wrapper handles RANSAC + geometric refit +
    quality checks (radius range, inner-empty, sector occupancy, aspect
    ratio, inlier fraction) and signals success via positive ``(a, b)``.

    A cluster is **kept** in the output when its fit returned strictly
    positive semi-axes — this captures both ``check_status == 0``
    (clean first-attempt fit) and ``check_status == 1`` with a
    successful retry on the largest DBSCAN sub-cluster. Failed retries
    surface as ``a == b == 0`` and are dropped here, as are slices with
    too few points (``check_status == 2``).

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        The full input point cloud (same one fed to GS.1–GS.3).
    clusters : list of Cluster2D
        Output of :func:`cluster_slice` for a single slab.
    config : EllipseFitConfig
        Configuration forwarded to ``_fit_ellipse_check``. The
        equivalent-radius window ``[r_min, r_max]`` filters out
        cross-sections that are too thin (top of canopy) or too thick
        (artefacts from merged clusters in dense plots).
    rng : np.random.Generator, optional
        Forwarded to the RANSAC loop. Pin a seeded generator for
        reproducible runs (recommended for the GS gate comparisons).

    Returns
    -------
    list of ClusterEllipse
        One entry per cluster whose fit passed the quality checks,
        in the same order as the input ``clusters``. Empty if every
        cluster failed (e.g. all sparse noise) or the input list is
        empty.
    """
    results: List[ClusterEllipse] = []
    for cluster in clusters:
        # _fit_ellipse_check already guards its inputs (length checks
        # etc.), so we don't pre-validate here beyond pulling the XY view.
        cluster_xy = xyz[cluster.indices, :2]
        xc, yc, a, b, theta, status, sector_pct = _fit_ellipse_check(
            cluster_xy[:, 0], cluster_xy[:, 1], config, rng=rng,
        )
        if a > 0.0 and b > 0.0:
            results.append(ClusterEllipse(
                indices=cluster.indices,
                xc=float(xc),
                yc=float(yc),
                a=float(a),
                b=float(b),
                theta=float(theta),
                sector_pct=float(sector_pct),
                check_status=int(status),
                n_points=cluster.n_points,
            ))
    return results


# ===========================================================================
# GS.5 — Vertical tracking (greedy adjacent matching)
# ===========================================================================

@dataclass(frozen=True)
class TrackNode:
    """A single node of a vertical track: an ellipse stamped with its
    slab z. The ellipse remains slab-agnostic in :class:`ClusterEllipse`;
    z lives on the node."""
    z: float
    ellipse: "ClusterEllipse"


@dataclass(frozen=True)
class Track:
    """A vertical sequence of ellipses presumed to belong to one tree.

    ``nodes`` are ordered by ascending z (bottom of the stem first).
    A track may have a single node (a basal ellipse with no upward
    continuation) or many. Bifurcations are NOT resolved here — they
    are handled in GS.6 / GS.7 once basal seeding is layered on top.
    """
    nodes: List[TrackNode]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def z_bottom(self) -> float:
        return self.nodes[0].z

    @property
    def z_top(self) -> float:
        return self.nodes[-1].z


def track_clusters_vertical(
    slab_centres: Sequence[float],
    slab_ellipses: List[List[ClusterEllipse]],
    max_xy_jump: float = 0.30,
    max_gap_slabs: int = 1,
) -> List[Track]:
    """Build vertical tracks by greedy XY matching across adjacent slabs.

    Walks the slabs bottom-to-top. At each new slab, each *open* track
    (one whose most recent node sits within ``max_gap_slabs`` slabs of
    the current one) tries to claim an ellipse in the current slab by
    minimum XY distance from its last node. Matches with distance above
    ``max_xy_jump`` are rejected. Ellipses left unclaimed start a new
    track each.

    The matching strategy is **greedy on a global cost sort** rather
    than per-track: we compute all (eligible track, current ellipse)
    XY distances, sort ascending, and consume in order — each side at
    most once per slab. This avoids the pathological case where one
    track greedily grabs a "good enough" match while another track had
    a much better option for the same ellipse. It is not globally
    optimal (Hungarian would be) but is fast and good enough on the
    well-separated-stems case Jorge's plots present; we can upgrade
    later if it becomes a bottleneck.

    Parameters
    ----------
    slab_centres : sequence of float
        Z-centres of the slabs, in ascending order. Length must match
        ``slab_ellipses``.
    slab_ellipses : list of list of ClusterEllipse
        Per-slab outputs of :func:`fit_ellipses_in_slice`, indexed the
        same as ``slab_centres``. Inner lists may be empty (slab with
        no successful fits) — those slabs are skipped, but they still
        count toward the gap budget for adjacent matching.
    max_xy_jump : float, default 0.30
        Maximum XY distance between a track's last node and a candidate
        ellipse in the current slab to consider them the same stem
        (metres). At 1 m slab spacing this allows ~17° of stem
        inclination per metre — comfortably above what we see in plots.
    max_gap_slabs : int, default 1
        Allow extending a track over a gap of up to this many empty
        slabs (e.g. ``1`` = adjacent-only; ``2`` = may skip one slab).
        Higher values tolerate occlusion at the cost of stitching
        unrelated stems together when they happen to be vertically
        aligned.

    Returns
    -------
    list of Track
        One ``Track`` per detected stem candidate. Tracks of length 1
        (singletons) are returned too — GS.6 will filter or seed them
        based on whether they originate at the basal stripe. Order in
        the returned list is the chronological creation order (oldest
        track first), not sorted by size.

    Raises
    ------
    ValueError
        If ``slab_centres`` and ``slab_ellipses`` have different lengths,
        or ``max_xy_jump`` / ``max_gap_slabs`` are non-positive.
    """
    if len(slab_centres) != len(slab_ellipses):
        raise ValueError(
            f"slab_centres and slab_ellipses must have the same length; "
            f"got {len(slab_centres)} and {len(slab_ellipses)}"
        )
    if max_xy_jump <= 0.0:
        raise ValueError(f"max_xy_jump must be positive; got {max_xy_jump}")
    if max_gap_slabs < 1:
        raise ValueError(f"max_gap_slabs must be >= 1; got {max_gap_slabs}")

    # `tracks[i]` is a list of TrackNode; `track_last_slab[i]` is the
    # slab index where its last node lives. Parallel lists keep the
    # access pattern simple.
    tracks: List[List[TrackNode]] = []
    track_last_slab: List[int] = []

    for slab_idx, (z_centre, ellipses) in enumerate(
        zip(slab_centres, slab_ellipses)
    ):
        if not ellipses:
            continue  # nothing to match or start

        # Eligible tracks: their last node is in a slab within the gap
        # budget. A gap of 0 means same slab (we never match within),
        # so the minimum eligible gap is 1.
        eligible = [
            ti for ti, ls in enumerate(track_last_slab)
            if 1 <= (slab_idx - ls) <= max_gap_slabs
        ]

        used_tracks: set = set()
        used_ellipses: set = set()

        if eligible:
            # Build (distance, track_index, ellipse_index) candidate list.
            costs: List[Tuple[float, int, int]] = []
            for ti in eligible:
                last_ell = tracks[ti][-1].ellipse
                lx, ly = last_ell.xc, last_ell.yc
                for ei, ell in enumerate(ellipses):
                    d = float(np.hypot(ell.xc - lx, ell.yc - ly))
                    if d <= max_xy_jump:
                        costs.append((d, ti, ei))
            # Sort by distance ascending; consume greedily.
            costs.sort(key=lambda c: c[0])
            for d, ti, ei in costs:
                if ti in used_tracks or ei in used_ellipses:
                    continue
                # Extend track ti with ellipse ei.
                tracks[ti].append(TrackNode(z=z_centre, ellipse=ellipses[ei]))
                track_last_slab[ti] = slab_idx
                used_tracks.add(ti)
                used_ellipses.add(ei)

        # Unmatched ellipses → new tracks.
        for ei, ell in enumerate(ellipses):
            if ei in used_ellipses:
                continue
            tracks.append([TrackNode(z=z_centre, ellipse=ell)])
            track_last_slab.append(slab_idx)

    return [Track(nodes=list(t)) for t in tracks]
