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

import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

from src.core.ellipse_fitting import EllipseFitConfig, _fit_ellipse_check
from src.core.features import (
    compute_verticality_mask_early_exit,
    voxelize_cloud,
)
from src.core.trunk_extraction import TrunkExtractionConfig, TrunkExtractionResult


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
    voxel_resolution: float = 0.0,
) -> List[Cluster2D]:
    """Run DBSCAN on the XY positions of points belonging to a slice.

    The horizontal slice from :func:`slice_horizontal_global` carries
    indices into the full cloud; this function reads the XY positions
    of those points (ignoring z, since the slice already constrains it)
    and groups them into 2D clusters with DBSCAN.

    Noise points (label ``-1``) are dropped. Each returned cluster
    carries an index array pointing back into the original cloud, so
    no slice-local renumbering ever leaks downstream.

    Voxel pre-aggregation (performance)
    -----------------------------------
    When ``voxel_resolution > 0``, the slab is voxelised at that
    resolution and DBSCAN runs on the voxel centroids' XY (typically
    20–50× fewer points than the raw slab). After clustering, each
    original-cloud point inherits the cluster label of its voxel.
    This is the canonical optimisation for dense basal slabs where
    raw-point DBSCAN scales pathologically (sklearn's neighbour
    expansion is ~O(N²) inside dense regions).

    With voxel pre-aggregation enabled, ``min_samples`` applies to
    **voxels**, not raw points. A typical regime for 5 cm voxels and
    ~15 cm stems is ``min_samples ≈ 5`` (clusters need at least 5
    voxels = ~125 cm² of stem perimeter coverage). Raise this if
    DBSCAN over-merges; lower if real stems get fragmented.

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
    min_samples : int, default 10
        DBSCAN ``min_samples``. Below this the point becomes noise.
        Meaning depends on ``voxel_resolution`` — see notes above.
    voxel_resolution : float, default 0.0
        If positive, voxelise the slab at this resolution (m) and run
        DBSCAN on voxel centroids instead of raw points. 0.0 disables
        voxelisation (legacy behaviour).

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
        ``min_samples`` / ``voxel_resolution`` are out of range.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3); got {xyz.shape}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1; got {min_samples}")
    if voxel_resolution < 0.0:
        raise ValueError(
            f"voxel_resolution must be >= 0; got {voxel_resolution}"
        )

    if slice_obj.indices.size == 0:
        return []

    slab_xyz = xyz[slice_obj.indices, :]
    pts_xy_full = slab_xyz[:, :2]

    from sklearn.cluster import DBSCAN

    if voxel_resolution > 0.0:
        # Voxelise the slab (XY + Z); DBSCAN on voxel centroids' XY.
        centroids, point_to_voxel, _ = voxelize_cloud(
            slab_xyz,
            resolution_xy=voxel_resolution,
            resolution_z=voxel_resolution,
        )
        voxel_labels = DBSCAN(
            eps=eps, min_samples=min_samples,
        ).fit_predict(centroids[:, :2])
        # Expand voxel-level labels to original-cloud point labels.
        labels = voxel_labels[point_to_voxel]
    else:
        labels = DBSCAN(
            eps=eps, min_samples=min_samples,
        ).fit_predict(pts_xy_full)

    clusters: List[Cluster2D] = []
    for label in np.unique(labels):
        if label == -1:
            continue
        mask = labels == label
        cluster_indices = slice_obj.indices[mask]
        centroid = pts_xy_full[mask].mean(axis=0)
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


# ===========================================================================
# GS.6 — Bootstrap tracks from a basal stripe
# ===========================================================================

def bootstrap_tracks_from_basal_stripe(
    tracks: List[Track],
    stripe_z_low: float,
    stripe_z_high: float,
    min_track_length: int = 1,
) -> List[Track]:
    """Keep only tracks rooted in the basal stripe; discard floating ones.

    A track is **rooted in the basal stripe** when its lowest node
    (``track.z_bottom``) lies in ``[stripe_z_low, stripe_z_high]``.
    Tracks whose lowest node sits above the stripe are presumed to be
    artefacts — clusters that aligned vertically through several slabs
    of foliage / canopy but have no actual basal anchor. They are
    discarded.

    This step is the **bootstrap** that turns the noisy candidate-track
    set from GS.5 (``track_clusters_vertical``) into a set of trustworthy
    per-tree polylines. After GS.6 the surviving tracks define the
    tree IDs that GS.7 will stamp onto the point cloud.

    Parameters
    ----------
    tracks : list of Track
        Candidate tracks from :func:`track_clusters_vertical`.
    stripe_z_low : float
        Lower bound of the basal stripe in metres (inclusive).
    stripe_z_high : float
        Upper bound of the basal stripe in metres (inclusive).
    min_track_length : int, default 1
        Optional length filter: also drop tracks with fewer than
        ``min_track_length`` nodes. ``1`` keeps singleton tracks
        (any track with a basal anchor counts, even if only one
        ellipse passed quality checks). ``2`` requires at least one
        vertical continuation, useful when working with sparse plots.

    Returns
    -------
    list of Track
        Surviving tracks, in the same order as the input list.
        Empty if every track was filtered out (e.g. plot with no
        basal coverage).

    Raises
    ------
    ValueError
        If ``stripe_z_low > stripe_z_high`` or
        ``min_track_length < 1``.
    """
    if stripe_z_low > stripe_z_high:
        raise ValueError(
            f"stripe_z_low must be <= stripe_z_high; "
            f"got {stripe_z_low} and {stripe_z_high}"
        )
    if min_track_length < 1:
        raise ValueError(
            f"min_track_length must be >= 1; got {min_track_length}"
        )

    survivors: List[Track] = []
    for track in tracks:
        if track.n_nodes < min_track_length:
            continue
        z0 = track.z_bottom
        if stripe_z_low <= z0 <= stripe_z_high:
            survivors.append(track)

    return survivors


# ===========================================================================
# GS.7 — Assign tree_id to original-cloud points from surviving tracks
# ===========================================================================

@dataclass
class TrackingAssignmentResult:
    """Per-point ``tree_id`` / ``stem_id`` assignment, produced by
    :func:`assign_tree_ids_from_tracks` (GS.7).

    Mirrors the shape of :class:`TrunkExtractionResult.tree_ids` so it
    can replace the straight-cylinder assignment downstream without
    touching `clean_stems` or `compute_stem_sections`.

    Fields
    ------
    tree_ids : ndarray (N,) int32
        Per-point tree index in ``range(n_trees)``, or ``-1`` for
        unassigned points (everything outside any successful track:
        understory, foliage, canopy, ground noise, sparse fits that
        failed quality checks).
    stem_ids : ndarray (N,) int32
        Per-point stem index within the tree. ``0`` is the main stem;
        values ``>= 1`` mark bifurcated secondary stems whose DBH is
        measured at 1.30 m above the bifurcation point (PlotSafe
        schema). GS.7 leaves all assigned points at ``stem_id == 0``;
        bifurcation handling is a later sub-fase.
    n_trees : int
        Number of distinct tree IDs assigned (== number of surviving
        tracks fed in).
    tracks : list of Track
        The exact tracks used, in the same order as the assigned IDs
        (``tree_id == k`` corresponds to ``tracks[k]``). Kept on the
        result so downstream code can read the per-tree centerline
        directly without re-deriving it.
    """
    tree_ids: np.ndarray
    stem_ids: np.ndarray
    n_trees: int
    tracks: List[Track] = field(default_factory=list)


def assign_tree_ids_from_tracks(
    tracks: List[Track],
    n_points: int,
) -> TrackingAssignmentResult:
    """Stamp ``tree_id`` on every point that belongs to a surviving track.

    For each track in input order, collects the original-cloud indices
    from every node's ellipse (``node.ellipse.indices``) and assigns
    ``tree_id = track_index`` to those points. Points belonging to no
    track remain at ``-1``.

    **First-write-wins** when two tracks share an index (which only
    happens with overlapping slabs, ``slab_half_thickness > slab_step
    / 2``): the earlier track keeps the point. This is deterministic
    and idempotent; in the default no-overlap configuration the
    conflict path never fires.

    Bifurcations are **not** handled here. Every assigned point gets
    ``stem_id == 0``. A later sub-fase will detect orphan tracks
    near a parent track and reassign them to ``stem_id >= 1`` of the
    parent's tree, following the PlotSafe field-data schema.

    Parameters
    ----------
    tracks : list of Track
        Surviving tracks from :func:`bootstrap_tracks_from_basal_stripe`
        (GS.6). The function trusts these are the trees to keep;
        filtering is a GS.6 responsibility.
    n_points : int
        Length of the original input cloud. The returned arrays are
        allocated to this length so downstream code can use them as
        per-point labels without any re-indexing.

    Returns
    -------
    TrackingAssignmentResult
        With ``tree_ids`` / ``stem_ids`` arrays of length ``n_points``
        and ``n_trees == len(tracks)``.

    Raises
    ------
    ValueError
        If ``n_points`` is negative, or any track contains an index
        out of range for ``n_points``.
    """
    if n_points < 0:
        raise ValueError(f"n_points must be >= 0; got {n_points}")

    tree_ids = np.full(n_points, -1, dtype=np.int32)
    stem_ids = np.zeros(n_points, dtype=np.int32)

    for track_idx, track in enumerate(tracks):
        for node in track.nodes:
            indices = node.ellipse.indices
            if indices.size == 0:
                continue
            # Defensive bounds check — catches mocks or upstream bugs.
            if int(indices.max()) >= n_points or int(indices.min()) < 0:
                raise ValueError(
                    f"track {track_idx} has indices out of range "
                    f"[0, {n_points}); got min={int(indices.min())}, "
                    f"max={int(indices.max())}"
                )
            # First-write-wins: only stamp tree_id where still unassigned.
            unassigned = tree_ids[indices] == -1
            tree_ids[indices[unassigned]] = track_idx

    return TrackingAssignmentResult(
        tree_ids=tree_ids,
        stem_ids=stem_ids,
        n_trees=len(tracks),
        tracks=list(tracks),
    )


# ===========================================================================
# GS.7b — Curved-cylinder assignment with adaptive radius
# ===========================================================================
#
# Replaces the GS.7 DBSCAN-membership approach for assignment. Where GS.7
# stamps ``tree_id`` only on the points that survived RANSAC fits inside
# the slabs the track touched, this approach builds a 3D **tube** around
# each track's polyline and assigns every point that falls inside the
# tube — recovering points DBSCAN fragmented, slabs where the ellipse
# fit failed, and the regions between successful slabs.
#
# This is the operational retirement of the straight cylinder: the
# legacy `extract_trunks` cylinder is straight per-tree; this one is a
# chained sequence of small cylindrical segments along the curved track,
# with the radius interpolating between adjacent nodes' fitted semi-major
# axes.

def assign_trees_by_curved_cylinder(
    xyz: np.ndarray,
    tracks: List[Track],
    radius_factor: float = 1.5,
    radius_min: float = 0.05,
    radius_max: float = 0.50,
    z_margin: float = 0.5,
    extend_below: Optional[float] = None,
    extend_above: Optional[float] = None,
) -> TrackingAssignmentResult:
    """Assign tree_id by a curved-cylinder buffer around each track.

    For each track, the polyline is the sequence of node centres
    ``(xc, yc, z)`` sorted ascending in z. Per node the radius is
    ``ellipse.a * radius_factor`` clipped to
    ``[radius_min, radius_max]``. Each polyline segment (pair of
    consecutive nodes) defines a frustum-like tube whose radius
    interpolates linearly between the two endpoints. The polyline is
    virtually extended above and below by ``z_margin`` so points
    slightly beyond the topmost / bottommost node are still captured.

    For every cloud point inside any tube, the assignment goes to the
    track with the smallest **normalised perpendicular distance**
    (``perp_distance / local_radius``). This handles dense plots where
    adjacent stems' tubes overlap — the point lands on the actually
    closer stem in stem-radius units, not the arbitrarily nearer one
    in absolute metres.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        Original input cloud (same as fed to `assign_trees_by_tracking`).
    tracks : list of Track
        Surviving tracks from GS.6.
    radius_factor : float, default 1.5
        Multiplier applied to ``node.ellipse.a`` to size each segment.
        1.5× the semi-major axis is a forgiving default — the tube
        captures the outer perimeter of the stem plus a small margin
        for sensor noise.
    radius_min, radius_max : float, defaults 0.05 / 0.50 (m)
        Hard bounds on the per-node radius. Stops degenerate fits
        (a → 0 from a bad RANSAC outcome) from producing a zero-radius
        tube, and prevents merged-cluster artefacts from creating an
        oversized tube that would steal points from neighbouring stems.
    z_margin : float, default 0.5 (m)
        Backward-compat default for ``extend_below`` and ``extend_above``
        when those are left as ``None``. Has no other effect.
    extend_below : float, optional
        Arc-length distance to extrapolate the polyline below its bottom
        node, along the **tangent of the first segment** (``-(n[1]-n[0])``
        direction). Set to ~3 m to reach ground from a track that starts
        at z=3 m. If ``None``, defaults to ``z_margin``. Set to 0 to
        disable the bottom extension entirely.
    extend_above : float, optional
        Arc-length distance to extrapolate the polyline above its top
        node, along the **tangent of the last segment** (``n[-1]-n[-2]``
        direction). Useful when point density / occlusion above the
        canopy break prevents new ellipse fits but the trunk visibly
        continues upward — the tube follows the lean the centerline was
        already taking. If ``None``, defaults to ``z_margin``. Set to 0
        to disable.

    Returns
    -------
    TrackingAssignmentResult
        ``tree_ids`` array of length ``N`` with assigned IDs in
        ``range(n_trees)`` and ``-1`` outside every tube;
        ``stem_ids`` is all zeros (bifurcations not handled yet).
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3); got {xyz.shape}")
    if radius_factor <= 0.0:
        raise ValueError(f"radius_factor must be positive; got {radius_factor}")
    if not (0.0 < radius_min <= radius_max):
        raise ValueError(
            f"radius bounds invalid; need 0 < min <= max, "
            f"got [{radius_min}, {radius_max}]"
        )
    if z_margin < 0.0:
        raise ValueError(f"z_margin must be >= 0; got {z_margin}")
    if extend_below is None:
        extend_below = z_margin
    if extend_above is None:
        extend_above = z_margin
    if extend_below < 0.0:
        raise ValueError(f"extend_below must be >= 0; got {extend_below}")
    if extend_above < 0.0:
        raise ValueError(f"extend_above must be >= 0; got {extend_above}")

    n_points = xyz.shape[0]
    tree_ids = np.full(n_points, -1, dtype=np.int32)
    stem_ids = np.zeros(n_points, dtype=np.int32)
    # Per-point tie-breaker: normalised distance to the closest tube
    # so far. Lower wins.
    best_norm_dist = np.full(n_points, np.inf, dtype=np.float64)

    x_all = xyz[:, 0]
    y_all = xyz[:, 1]
    z_all = xyz[:, 2]

    for track_idx, track in enumerate(tracks):
        # Polyline + per-node radius
        nodes_xyz = np.array(
            [(node.ellipse.xc, node.ellipse.yc, node.z) for node in track.nodes],
            dtype=np.float64,
        )
        radii = np.clip(
            np.array(
                [node.ellipse.a * radius_factor for node in track.nodes],
                dtype=np.float64,
            ),
            radius_min,
            radius_max,
        )

        # Extend the polyline by phantom nodes at both endpoints along
        # the **local tangent** of the first / last segment. This lets
        # the tube follow the lean the centerline was already taking
        # (rather than snapping to pure-vertical), and the extension
        # arc-length is configurable independently for each end.
        # Singletons are handled below with a spherical buffer instead,
        # so we skip the extension for them.
        if len(nodes_xyz) >= 2:
            bottom_ghost = None
            top_ghost = None

            if extend_below > 0.0:
                first_seg = nodes_xyz[1] - nodes_xyz[0]
                seg_len = float(np.linalg.norm(first_seg))
                if seg_len > 1e-9:
                    # Direction pointing OUT of the polyline at the bottom
                    # (i.e., from n[1] toward n[0] and beyond).
                    outward = -first_seg / seg_len
                    bottom_ghost = nodes_xyz[0] + outward * extend_below

            if extend_above > 0.0:
                last_seg = nodes_xyz[-1] - nodes_xyz[-2]
                seg_len = float(np.linalg.norm(last_seg))
                if seg_len > 1e-9:
                    # Direction pointing OUT of the polyline at the top.
                    outward = last_seg / seg_len
                    top_ghost = nodes_xyz[-1] + outward * extend_above

            # Apply both ghosts at once to keep index arithmetic simple.
            prepend = (
                [bottom_ghost[np.newaxis, :]] if bottom_ghost is not None
                else []
            )
            append = (
                [top_ghost[np.newaxis, :]] if top_ghost is not None
                else []
            )
            if prepend or append:
                nodes_xyz = np.vstack(prepend + [nodes_xyz] + append)
                pre_r = [radii[:1]] if bottom_ghost is not None else []
                post_r = [radii[-1:]] if top_ghost is not None else []
                radii = np.concatenate(pre_r + [radii] + post_r)

        # Derive the z-AABB from the (possibly extended) polyline. With
        # tangent-based extensions, the bottom/top ghost z's are not
        # simply ``z_bottom - extend_below`` — they depend on the local
        # tangent's vertical component.
        z_lo = float(nodes_xyz[:, 2].min())
        z_hi = float(nodes_xyz[:, 2].max())

        # Candidate point indices in 3D AABB (cheap pre-filter).
        # Per-track XY bounding box: spans the polyline's xc/yc range with
        # an extra radius_max margin on each side, which is the largest
        # cross-segment distance any inlier point could ever have. Any
        # point outside this AABB cannot fall inside the tube for ANY
        # segment, so we can safely prune it before the per-segment loop.
        # On a 20m-radius plot with ~57 tracks, this drops cand_xyz from
        # ~21M points (z-only filter) to ~10s-100s of k per track
        # (~2m × 2m × Δz footprint) — typically 100-1000× speedup on the
        # per-segment work that dominates GS.7b runtime.
        x_lo = float(nodes_xyz[:, 0].min() - radius_max)
        x_hi = float(nodes_xyz[:, 0].max() + radius_max)
        y_lo = float(nodes_xyz[:, 1].min() - radius_max)
        y_hi = float(nodes_xyz[:, 1].max() + radius_max)
        in_z = np.where(
            (z_all >= z_lo) & (z_all <= z_hi)
            & (x_all >= x_lo) & (x_all <= x_hi)
            & (y_all >= y_lo) & (y_all <= y_hi)
        )[0]
        if in_z.size == 0:
            continue

        cand_xyz = xyz[in_z]

        if len(nodes_xyz) == 1:
            # Singleton track: spherical assignment centred at the node.
            p = nodes_xyz[0]
            r = radii[0]
            diff = cand_xyz - p
            norm_dist = np.linalg.norm(diff, axis=1) / r
        else:
            # Track minimum normalised distance across all segments.
            norm_dist = np.full(in_z.size, np.inf, dtype=np.float64)
            for i in range(len(nodes_xyz) - 1):
                p0 = nodes_xyz[i]
                p1 = nodes_xyz[i + 1]
                r0 = radii[i]
                r1 = radii[i + 1]
                seg = p1 - p0
                seg_len_sq = float(np.dot(seg, seg))
                if seg_len_sq < 1e-12:
                    # Coincident nodes (shouldn't happen post-GS.5); skip.
                    continue
                # Projection parameter onto the segment.
                t = ((cand_xyz - p0) @ seg) / seg_len_sq
                t_clipped = np.clip(t, 0.0, 1.0)
                # Closest point on the segment for each candidate.
                closest = p0 + t_clipped[:, None] * seg
                diff = cand_xyz - closest
                dist = np.linalg.norm(diff, axis=1)
                # Linearly interpolated radius along the segment.
                r_interp = r0 + t_clipped * (r1 - r0)
                seg_norm_dist = dist / r_interp
                np.minimum(norm_dist, seg_norm_dist, out=norm_dist)

        # A point is "inside" this track's tube iff norm_dist < 1.
        inside = norm_dist < 1.0
        # Competitive update against tracks processed earlier.
        better = inside & (norm_dist < best_norm_dist[in_z])
        global_idx = in_z[better]
        tree_ids[global_idx] = track_idx
        best_norm_dist[global_idx] = norm_dist[better]

    return TrackingAssignmentResult(
        tree_ids=tree_ids,
        stem_ids=stem_ids,
        n_trees=len(tracks),
        tracks=list(tracks),
    )


# ===========================================================================
# GS.8 — Orchestrator + compatibility shim
# ===========================================================================

@dataclass
class TrackingAssignmentConfig:
    """End-to-end configuration for :func:`assign_trees_by_tracking`.

    Groups the per-sub-fase knobs into one object so the notebook (and
    other top-level callers) can tune the pipeline without importing
    seven different functions.

    Defaults follow the Phase 1B plan and have been smoke-tested on
    synthetic clouds; calibration on a real plot may need to widen
    ``max_xy_jump`` (for plots with strong inclination), tighten
    ``dbscan_eps`` (in dense plantations), or shift
    ``basal_stripe_z_*`` to wherever the basal cross-section is clean
    on the user's data.
    """
    # --- GS.1 verticality filter ---
    verticality_threshold: float = 0.7
    verticality_scale: float = 0.10
    voxel_resolution_xy: float = 0.05
    voxel_resolution_z: float = 0.05
    coarse_resolution: float = 0.10
    coarse_margin: float = 0.10

    # --- GS.2 horizontal slicing ---
    z_min: float = 0.3
    z_max: float = 40.0
    slab_step: float = 1.0
    slab_half_thickness: float = 0.5

    # --- GS.3 DBSCAN per slice ---
    dbscan_eps: float = 0.10
    dbscan_min_samples: int = 10
    # When > 0, voxelise each slab at this resolution (m) before
    # running DBSCAN on the voxel centroids. Trades a marginal loss
    # of geometric resolution (~ voxel_resolution) for a 20–50×
    # speedup on dense basal slabs where raw-point DBSCAN scales
    # pathologically. 0.05 m matches the stripe of stems at typical
    # MLS/TLS sampling densities. 0.0 disables (legacy behaviour).
    # Note: ``dbscan_min_samples`` applies to whatever DBSCAN sees;
    # when voxelised, set it to the per-voxel count (≈ 5 for 5 cm
    # voxels and ~15 cm stems), NOT the per-point count.
    dbscan_voxel_resolution: float = 0.05

    # --- GS.4 ellipse fit (delegated to EllipseFitConfig) ---
    ellipse: EllipseFitConfig = field(default_factory=EllipseFitConfig)

    # --- GS.5 vertical tracking ---
    max_xy_jump: float = 0.30
    max_gap_slabs: int = 1

    # --- GS.6 basal-stripe bootstrap ---
    basal_stripe_z_low: float = 0.5
    basal_stripe_z_high: float = 2.0
    min_track_length: int = 1

    # --- Diagnostic dump (GS.8c, debug-only) ---
    # When True, the orchestrator prints per-cluster geometry + fit
    # outcome for basal slabs (z_centre <= basal_stripe_z_high).
    # Useful when GS.4 reports near-zero valid ellipses despite many
    # clusters — exposes whether the rejection is due to fragmentation
    # (n_pts < min_points_section), elongation (aspect > 3, not a stem
    # cross-section), or quality-check failure (sectors / inlier).
    # Costs ~10 s per basal slab via re-running the fit; only enable
    # for debug runs.
    diagnostic_basal_slabs: bool = False

    # Per-track diagnostic dump after GS.6 (debug-only). Shows
    # (n_nodes, z_bottom, z_top, Δz, xc_bot, yc_bot) and the basal-
    # stripe verdict per candidate track, with a category histogram
    # of why each was accepted/rejected. Useful when GS.6 keeps only
    # a handful of tracks out of many — exposes whether the bottleneck
    # is the stripe lower bound, the upper bound, the minimum length
    # filter, or fragmentation. Costs ~0 s.
    diagnostic_candidate_tracks: bool = False

    # --- GS.7 / GS.7b assignment mode ---
    # "curved_cylinder" is the default and what jubilates the straight
    # cylinder: an adaptive-radius buffer around the track's polyline
    # (cilindro curvo). "dbscan_membership" is the GS.7 legacy mode
    # that assigns only the points that survived the DBSCAN clusters
    # of the track's nodes — kept for diagnostic A/B comparisons.
    assignment_method: Literal["curved_cylinder", "dbscan_membership"] = (
        "curved_cylinder"
    )
    # Cylinder radius is `node.ellipse.a * cylinder_radius_factor`, clipped
    # to ``[cylinder_min_radius, cylinder_max_radius]`` to absorb
    # degenerate fits at the extremes.
    cylinder_radius_factor: float = 1.5
    cylinder_min_radius: float = 0.05
    cylinder_max_radius: float = 0.50
    # Polyline ends are extended by this margin above/below to capture
    # points that fall slightly beyond the topmost / bottommost node
    # (e.g. roots flare, sub-canopy points whose slab had no ellipse).
    # ``cylinder_extend_below`` and ``cylinder_extend_above`` default to
    # this when left None; set them explicitly for asymmetric extensions
    # (e.g. reach ground from a track that starts at z=3 m).
    cylinder_z_margin: float = 0.5
    # Arc-length distance to extrapolate the polyline below/above its
    # bottom/top node along the **local tangent** of the first/last
    # segment. When None, both default to ``cylinder_z_margin``. Set
    # explicitly when the GS pipeline's z_min cuts off the basal section
    # (extend_below ≈ z_min - ground) or when point density drops at the
    # top of the canopy (extend_above follows the lean of the last
    # segment, recovering the upper trunk that has no ellipse fits).
    cylinder_extend_below: Optional[float] = None
    cylinder_extend_above: Optional[float] = None


def diagnose_slab_clusters(
    xyz: np.ndarray,
    slab: "HorizontalSlice",
    config: "TrackingAssignmentConfig",
    rng: Optional[np.random.Generator] = None,
    max_clusters: int = 25,
) -> None:
    """Print per-cluster geometry and fit-rejection diagnostics.

    Re-runs :func:`cluster_slice` + :func:`_fit_ellipse_check` for the
    given slab and prints one row per cluster: point count, XY
    bounding box, PCA aspect ratio, fit status, sector_pct, and the
    rejection ``reason`` returned by the EL.5 wrapper
    (``return_reason=True``). Used to diagnose the **why** behind a
    near-zero GS.4 ellipse-acceptance rate — the ``outcome`` column
    names the first failing stage (``ransac_none``, ``sectors_low``,
    ``inlier_low``, ``r_min``/``r_max``, ``aspect_low``,
    ``inner_full``, ``conic_degenerate``, ``too_few_pts``, or
    ``retry_failed(<prev>)``).

    Intended for debug-only invocation. Costs ~10 s per slab via the
    re-fitting. The orchestrator calls this for every basal slab when
    ``config.diagnostic_basal_slabs`` is True.
    """
    clusters = cluster_slice(
        xyz, slab,
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
        voxel_resolution=config.dbscan_voxel_resolution,
    )
    if not clusters:
        print(f"\n[diagnose] slab z={slab.z_centre:.2f}m: 0 clusters — skipped")
        return

    # Header
    print(
        f"\n[diagnose] slab z={slab.z_centre:.2f}m: {len(clusters)} clusters "
        f"(showing first {min(max_clusters, len(clusters))})"
    )
    print(
        "    idx  n_pts   bbox_x  bbox_y  aspect   r_eq   status  "
        "sector_pct  outcome"
    )

    # Sort by descending size so the most interesting (largest) clusters
    # come first. Helps spot whether the biggest cluster is actually a
    # stem or something else.
    clusters_sorted = sorted(clusters, key=lambda c: c.n_points, reverse=True)
    for i, cluster in enumerate(clusters_sorted[:max_clusters]):
        cluster_xy = xyz[cluster.indices, :2]
        n_pts = cluster.n_points
        bbox_x = float(cluster_xy[:, 0].ptp())
        bbox_y = float(cluster_xy[:, 1].ptp())
        # PCA aspect ratio: stems are ~1.0, elongated foliage > 3.0.
        if n_pts > 1:
            centred = cluster_xy - cluster_xy.mean(axis=0)
            cov = np.cov(centred.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(np.abs(eigvals))
            aspect = float(eigvals[-1] / max(eigvals[0], 1e-12))
        else:
            aspect = float("inf")

        xc, yc, a, b, _theta, status, sector_pct, reason = _fit_ellipse_check(
            cluster_xy[:, 0], cluster_xy[:, 1], config.ellipse, rng=rng,
            return_reason=True,
        )
        r_eq = float(np.sqrt(a * b)) if (a > 0 and b > 0) else 0.0

        # Outcome = the actual rejection stage reported by _fit_ellipse_check.
        # bbox / aspect / r_eq are already columns of the dump above, so the
        # outcome just names *why* the fit failed (or "VALID" on success).
        if a > 0 and b > 0:
            outcome = "VALID" if status == 0 else "VALID (retry)"
        else:
            outcome = reason

        print(
            f"    {i:3d}  {n_pts:6d}  {bbox_x:5.3f}   {bbox_y:5.3f}   "
            f"{aspect:5.2f}  {r_eq:5.3f}    {int(status)}     {sector_pct:6.1f}    "
            f"{outcome}"
        )


def diagnose_candidate_tracks(
    tracks: List["Track"],
    config: "TrackingAssignmentConfig",
) -> None:
    """Print per-track diagnostics for GS.5 candidates + GS.6 verdicts.

    Mirrors the spirit of :func:`diagnose_slab_clusters` one stage
    upstream: when GS.6 keeps only a handful of tracks out of many,
    this dump shows *why* — for each candidate, prints
    ``(n_nodes, z_bottom, z_top, Δz, xc_bot, yc_bot)`` plus the basal-
    stripe verdict (``ACCEPT`` / ``below_stripe`` / ``above_stripe`` /
    ``too_short``) and a category histogram so a single run reveals
    whether the bottleneck is the stripe lower bound, the upper bound,
    or the minimum length filter.

    Intended for debug-only invocation. Costs ~0 s (tracks are already
    computed; just printed). The orchestrator calls this after GS.6
    when ``config.diagnostic_candidate_tracks`` is True.
    """
    z_low = config.basal_stripe_z_low
    z_high = config.basal_stripe_z_high
    min_len = config.min_track_length

    # Classify each track
    verdicts: List[str] = []
    cat_counts = {
        "ACCEPT": 0,
        "below_stripe": 0,
        "above_stripe": 0,
        "too_short": 0,
    }
    nnodes_buckets = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, ">10": 0}
    for track in tracks:
        n = track.n_nodes
        z0 = track.z_bottom
        if n < min_len:
            v = "too_short"
        elif z0 < z_low:
            v = "below_stripe"
        elif z0 > z_high:
            v = "above_stripe"
        else:
            v = "ACCEPT"
        verdicts.append(v)
        cat_counts[v] += 1

        if n == 1:
            nnodes_buckets["1"] += 1
        elif n == 2:
            nnodes_buckets["2"] += 1
        elif n <= 5:
            nnodes_buckets["3-5"] += 1
        elif n <= 10:
            nnodes_buckets["6-10"] += 1
        else:
            nnodes_buckets[">10"] += 1

    # Header + summary
    n_total = len(tracks)
    print(f"\n  === Candidate-track diagnostics (GS.5 → GS.6) ===")
    print(
        f"  basal_stripe = [{z_low:.2f}, {z_high:.2f}]m, "
        f"min_track_length = {min_len}"
    )
    print(
        f"  {n_total} candidate tracks → "
        f"{cat_counts['ACCEPT']} accepted, "
        f"{n_total - cat_counts['ACCEPT']} rejected"
    )
    print(f"    rejected by category:")
    print(f"      z_bottom < {z_low:.2f} (below stripe): {cat_counts['below_stripe']}")
    print(f"      z_bottom > {z_high:.2f} (above stripe): {cat_counts['above_stripe']}")
    print(f"      n_nodes < {min_len} (too short):       {cat_counts['too_short']}")
    print(f"    n_nodes histogram (all {n_total} tracks):")
    for k, v in nnodes_buckets.items():
        print(f"      {k:>5}: {v}")

    # Per-track table — sort by verdict (ACCEPT first), then by Δz desc
    # so the most "tree-like" candidates appear at the top.
    order = sorted(
        range(n_total),
        key=lambda i: (
            0 if verdicts[i] == "ACCEPT" else 1,
            -(tracks[i].z_top - tracks[i].z_bottom),
        ),
    )
    print(
        f"\n    idx  n_nodes  z_bot   z_top    Δz    xc_bot   yc_bot   verdict"
    )
    for i in order:
        t = tracks[i]
        z0 = t.z_bottom
        z1 = t.z_top
        dz = z1 - z0
        node0 = t.nodes[0]
        xc = float(node0.ellipse.xc)
        yc = float(node0.ellipse.yc)
        print(
            f"    {i:3d}   {t.n_nodes:5d}   {z0:5.2f}  {z1:6.2f}  {dz:5.2f}  "
            f"{xc:7.3f}  {yc:7.3f}   {verdicts[i]}"
        )
    print("  === End candidate-track diagnostics ===\n")


def assign_trees_by_tracking(
    xyz: np.ndarray,
    config: Optional[TrackingAssignmentConfig] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> TrackingAssignmentResult:
    """Run the full GS.1 → GS.7 chain and return per-point assignment.

    The orchestrator preserves the **original-cloud index space**
    throughout: GS.1 returns a mask into the input cloud; we record
    that mapping and remap GS.2's slice indices into the original
    coordinate system before passing them downstream. Every
    ``ClusterEllipse.indices``, ``TrackNode.ellipse.indices``, and
    the final ``tree_ids`` array therefore points to the SAME cloud
    the caller fed in — no per-stage re-indexing needed.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        Height-normalised vegetation point cloud.
    config : TrackingAssignmentConfig, optional
        Pipeline configuration. Defaults to a fresh
        ``TrackingAssignmentConfig()``.
    rng : np.random.Generator, optional
        Forwarded to the RANSAC ellipse fits. Pin a seed for
        reproducible runs (recommended for gate comparisons).
    verbose : bool, default False
        Print per-stage progress to stdout.

    Returns
    -------
    TrackingAssignmentResult
        With ``tree_ids`` / ``stem_ids`` of length ``N``,
        ``n_trees`` and the list of surviving Tracks.
    """
    if config is None:
        config = TrackingAssignmentConfig()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3); got {xyz.shape}")
    n_orig = xyz.shape[0]

    if verbose:
        print("\n=== GS.8: tracking-based tree_id assignment ===")
        print(f"  Input: {n_orig:,} points")

    t_pipeline_start = time.perf_counter()

    # GS.1 — verticality filter on the FULL cloud.
    t0 = time.perf_counter()
    filtered_xyz, keep_mask = filter_by_verticality(
        xyz,
        threshold=config.verticality_threshold,
        scale=config.verticality_scale,
        voxel_resolution_xy=config.voxel_resolution_xy,
        voxel_resolution_z=config.voxel_resolution_z,
        coarse_resolution=config.coarse_resolution,
        margin=config.coarse_margin,
    )
    original_indices = np.where(keep_mask)[0]  # filtered → original mapping
    t_gs1 = time.perf_counter() - t0
    if verbose:
        print(
            f"  GS.1 verticality filter: {filtered_xyz.shape[0]:,} / "
            f"{n_orig:,} kept ({100.0 * keep_mask.mean():.1f}%)  "
            f"[{t_gs1:.1f}s]"
        )

    # GS.2 — slice the filtered cloud horizontally.
    t0 = time.perf_counter()
    slices_filtered = slice_horizontal_global(
        filtered_xyz,
        z_min=config.z_min,
        z_max=config.z_max,
        slab_step=config.slab_step,
        slab_half_thickness=config.slab_half_thickness,
    )
    # Remap each slab's indices into the ORIGINAL cloud's coordinate
    # system. Downstream stages then carry original-cloud indices end
    # to end — no surprise re-indexing at GS.7.
    slices = [
        HorizontalSlice(
            z_centre=s.z_centre,
            z_low=s.z_low,
            z_high=s.z_high,
            indices=(
                original_indices[s.indices]
                if s.indices.size else s.indices
            ),
        )
        for s in slices_filtered
    ]
    t_gs2 = time.perf_counter() - t0
    if verbose:
        n_non_empty = sum(1 for s in slices if s.indices.size > 0)
        print(
            f"  GS.2 slicing: {len(slices)} slabs total, "
            f"{n_non_empty} non-empty  [{t_gs2:.1f}s]"
        )

    # GS.3 + GS.4 per slab — passing the ORIGINAL `xyz` since the
    # remapped slice indices point there.
    slab_ellipses: List[List[ClusterEllipse]] = []
    n_clusters_total = 0
    n_ellipses_total = 0
    t_gs3 = 0.0
    t_gs4 = 0.0
    # Per-slab profiling buckets for hot-slab diagnosis.
    per_slab_times: List[tuple] = []  # (slab_idx, z_centre, n_pts, t_gs3_slab, t_gs4_slab, n_clusters, n_ellipses)
    for slab_idx, s in enumerate(slices):
        t_slab3_start = time.perf_counter()
        clusters = cluster_slice(
            xyz, s,
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples,
            voxel_resolution=config.dbscan_voxel_resolution,
        )
        t_slab3 = time.perf_counter() - t_slab3_start
        t_gs3 += t_slab3
        n_clusters_total += len(clusters)

        t_slab4_start = time.perf_counter()
        ellipses = fit_ellipses_in_slice(
            xyz, clusters, config.ellipse, rng=rng,
        )
        t_slab4 = time.perf_counter() - t_slab4_start
        t_gs4 += t_slab4
        n_ellipses_total += len(ellipses)
        slab_ellipses.append(ellipses)
        per_slab_times.append((
            slab_idx, float(s.z_centre), int(s.indices.size),
            t_slab3, t_slab4, len(clusters), len(ellipses),
        ))
    if verbose:
        print(
            f"  GS.3 DBSCAN total: {n_clusters_total} clusters  [{t_gs3:.1f}s]"
        )
        print(
            f"  GS.4 ellipse fit:  {n_ellipses_total} valid ellipses  "
            f"[{t_gs4:.1f}s]"
        )
        # Top-5 hot slabs by total time, so Jorge can see where the
        # cost is concentrated (typically a couple of dense slabs).
        hot = sorted(
            per_slab_times,
            key=lambda r: r[3] + r[4],
            reverse=True,
        )[:5]
        print("    top-5 hot slabs (GS.3 + GS.4):")
        print("      idx   z(m)    pts   GS.3 s  GS.4 s  clusters  ellipses")
        for idx, z, npts, t3, t4, nc, ne in hot:
            print(
                f"      {idx:3d}  {z:5.2f}  {npts:7,}  "
                f"{t3:6.2f}  {t4:6.2f}    {nc:5d}    {ne:5d}"
            )

    # GS.5 — vertical tracking.
    t0 = time.perf_counter()
    slab_centres = [s.z_centre for s in slices]
    tracks = track_clusters_vertical(
        slab_centres, slab_ellipses,
        max_xy_jump=config.max_xy_jump,
        max_gap_slabs=config.max_gap_slabs,
    )
    t_gs5 = time.perf_counter() - t0
    if verbose:
        print(
            f"  GS.5 tracking: {len(tracks)} candidate tracks  [{t_gs5:.2f}s]"
        )

    # GS.6 — basal-stripe bootstrap.
    t0 = time.perf_counter()
    survivors = bootstrap_tracks_from_basal_stripe(
        tracks,
        config.basal_stripe_z_low,
        config.basal_stripe_z_high,
        min_track_length=config.min_track_length,
    )
    t_gs6 = time.perf_counter() - t0
    if verbose:
        print(
            f"  GS.6 basal bootstrap: {len(survivors)} surviving tracks "
            f"(= n_trees)  [{t_gs6:.2f}s]"
        )

    # Per-track diagnostic dump (palanca 0b, debug-only).
    if config.diagnostic_candidate_tracks:
        diagnose_candidate_tracks(tracks, config)

    # Diagnostic dump on basal slabs (GS.8c, debug-only).
    if config.diagnostic_basal_slabs:
        print("\n  === Basal-slab cluster diagnostics ===")
        for slab in slices:
            if (
                slab.z_centre <= config.basal_stripe_z_high
                and slab.indices.size > 0
            ):
                diagnose_slab_clusters(xyz, slab, config, rng=rng)
        print("  === End diagnostics ===\n")

    # GS.7 / GS.7b — assignment to the full-size cloud.
    t0 = time.perf_counter()
    if config.assignment_method == "curved_cylinder":
        result = assign_trees_by_curved_cylinder(
            xyz, survivors,
            radius_factor=config.cylinder_radius_factor,
            radius_min=config.cylinder_min_radius,
            radius_max=config.cylinder_max_radius,
            z_margin=config.cylinder_z_margin,
            extend_below=config.cylinder_extend_below,
            extend_above=config.cylinder_extend_above,
        )
        method_label = "curved cylinder (GS.7b)"
    elif config.assignment_method == "dbscan_membership":
        result = assign_tree_ids_from_tracks(survivors, n_points=n_orig)
        method_label = "DBSCAN membership (GS.7 legacy)"
    else:
        raise ValueError(
            f"unknown assignment_method: {config.assignment_method!r}"
        )
    t_gs7 = time.perf_counter() - t0

    t_total = time.perf_counter() - t_pipeline_start
    if verbose:
        n_assigned = int((result.tree_ids >= 0).sum())
        print(
            f"  {method_label}: {n_assigned:,} / {n_orig:,} points "
            f"labelled ({100.0 * n_assigned / max(n_orig, 1):.1f}%)  "
            f"[{t_gs7:.1f}s]"
        )
        print(
            f"=== Done. n_trees = {result.n_trees}  "
            f"(total {t_total:.1f}s) ==="
        )
        # Compact roll-up so Jorge can paste it back if needed.
        print(
            "  breakdown: "
            f"GS.1={t_gs1:.1f}s  GS.2={t_gs2:.2f}s  "
            f"GS.3={t_gs3:.1f}s  GS.4={t_gs4:.1f}s  "
            f"GS.5={t_gs5:.2f}s  GS.6={t_gs6:.2f}s  GS.7={t_gs7:.1f}s"
        )

    return result


def build_tree_axes_from_tracks(
    xyz: np.ndarray,
    tracking_result: TrackingAssignmentResult,
) -> List[dict]:
    """Build a list of ``tree_axes`` dicts compatible with
    :class:`TrunkExtractionResult.tree_axes`.

    Downstream stages (``clean_stems`` suspicious scoring, the audit
    table, the inventory export) read a handful of fields from each
    axis: ``tree_id``, ``centroid``, ``direction``, ``line_point``,
    ``z_min``, ``z_max``. This shim populates them from the track
    geometry:

    - ``centroid``: the basal node's ellipse centre in 3D (x, y, z).
    - ``direction``: dominant PCA direction over the track's node
      positions, sign-flipped to keep ``z > 0``. Singletons fall back
      to ``[0, 0, 1]``.
    - ``line_point``: same as ``centroid`` (fallback for code that
      reads ``ax.get("line_point", ax["centroid"])``).
    - ``z_min`` / ``z_max``: actual z-extent of all points labelled
      with this ``tree_id``.

    The straight-cylinder-only fields (``basal_anchor_*``) are NOT
    populated; consumers must use ``.get(..., default)`` to read them.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        The original cloud (same as fed to ``assign_trees_by_tracking``).
    tracking_result : TrackingAssignmentResult
        Output of ``assign_trees_by_tracking``.

    Returns
    -------
    list of dict
        One dict per tree, in the same order as
        ``tracking_result.tracks``.
    """
    axes: List[dict] = []
    for i, track in enumerate(tracking_result.tracks):
        # z-extent from the actual labelled points (not from the track
        # nodes — the points usually reach above the topmost node).
        tree_mask = tracking_result.tree_ids == i
        if int(tree_mask.sum()) > 0:
            tree_z = xyz[tree_mask, 2]
            z_min = float(tree_z.min())
            z_max = float(tree_z.max())
        else:
            z_min = float(track.z_bottom)
            z_max = float(track.z_top)

        basal = track.nodes[0]
        centroid = np.array(
            [basal.ellipse.xc, basal.ellipse.yc, float(basal.z)],
            dtype=np.float64,
        )

        if track.n_nodes >= 2:
            nodes_xyz = np.array(
                [(n.ellipse.xc, n.ellipse.yc, n.z) for n in track.nodes],
                dtype=np.float64,
            )
            centred = nodes_xyz - nodes_xyz.mean(axis=0)
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            direction = vt[0]
            if direction[2] < 0.0:
                direction = -direction
        else:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        axes.append({
            "tree_id": i,
            "centroid": centroid,
            "direction": direction,
            "line_point": centroid,
            "z_min": z_min,
            "z_max": z_max,
        })

    return axes


def tracking_result_to_trunk_extraction_result(
    xyz: np.ndarray,
    tracking_result: TrackingAssignmentResult,
) -> TrunkExtractionResult:
    """Wrap a :class:`TrackingAssignmentResult` as a drop-in
    :class:`TrunkExtractionResult` so existing downstream consumers
    (``clean_stems``, ``compute_stem_sections``, the audit table, the
    inventory export) keep working without modification.

    ``trunk_mask`` becomes ``tree_ids >= 0`` (any point with an assigned
    tree is considered a trunk point).
    """
    trunk_mask = tracking_result.tree_ids >= 0
    tree_axes = build_tree_axes_from_tracks(xyz, tracking_result)
    # `cluster_points` is the stripe-clusters output of the legacy
    # pipeline. Tracking has no equivalent (its "stripe" is the whole
    # basal-mask intersected with valid tracks), so we surface an
    # empty array. No downstream consumer in the active notebook
    # cells reads this field.
    return TrunkExtractionResult(
        trunk_mask=trunk_mask,
        tree_ids=tracking_result.tree_ids,
        n_trees=tracking_result.n_trees,
        tree_axes=tree_axes,
        cluster_points=np.empty((0, 3), dtype=np.float64),
        config=TrunkExtractionConfig(),
    )


# ===========================================================================
# GS.8b — Tracking audit table (false-positive diagnostic)
# ===========================================================================

def build_tracking_audit_table(
    xyz: np.ndarray,
    tracking_result: TrackingAssignmentResult,
    slab_step: float,
):
    """Build a per-track diagnostic CSV analogous to ``trunk_audit.csv``.

    Surfaces the metrics needed to spot false-positive tracks (small
    isolated clusters that happened to fit an ellipse but don't look
    like real trees):

      * ``n_nodes``: how many slabs the track spans. Real trees usually
        cover ≥ 5 nodes; FPs often have 1–2.
      * ``z_extent``: total vertical extent in metres. FPs are typically
        < 3 m.
      * ``gap_fraction``: ``1 − n_nodes / expected_nodes`` where
        ``expected_nodes`` is the count of slab positions that fit
        inside the track's z-range. Continuous real trees → ~ 0.0;
        FPs with gaps → ≥ 0.3.
      * ``n_points``: total points labelled with the tree_id (membership
        of every ellipse cluster across all the track's nodes).
      * ``mean_radius``, ``radius_cv``: consistency of √(a·b) across
        nodes. Real trees taper smoothly; FPs jump around.
      * ``mean_aspect_ratio``: average ``a / b``. Real stems ≈ 1.0;
        elongated FPs (rows of leaves, ground texture) > 1.5.
      * ``mean_sector_pct``: average sector-occupancy quality. Real
        stems often > 60 %; FPs hover at the rejection floor.
      * ``density_pts_per_m3``: ``n_points / (π·r²·h)`` — order-of-
        magnitude FP separator on the right sensor.

    The function does NOT filter anything: it only reports. Use the CSV
    to pick thresholds, then either re-run with a stricter config or
    write a downstream filter (GS.8c, optional).

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        The original cloud passed to ``assign_trees_by_tracking``. Only
        used for ``n_points`` lookup (we count by ``tree_ids == k``).
    tracking_result : TrackingAssignmentResult
        Output of the GS pipeline.
    slab_step : float
        Vertical spacing of the slabs (from
        ``TrackingAssignmentConfig.slab_step``). Needed to compute
        ``gap_fraction``.

    Returns
    -------
    pandas.DataFrame
        One row per track (= per tree_id). Empty if
        ``tracking_result.tracks`` is empty.
    """
    import pandas as pd

    if slab_step <= 0.0:
        raise ValueError(f"slab_step must be positive; got {slab_step}")

    rows = []
    for i, track in enumerate(tracking_result.tracks):
        n_nodes = track.n_nodes
        z_bottom = float(track.z_bottom)
        z_top = float(track.z_top)
        z_extent = z_top - z_bottom

        # Expected number of nodes if the track had no gaps:
        # one node per slab centre in the track's z range, inclusive.
        if z_extent > 0:
            expected_nodes = int(round(z_extent / slab_step)) + 1
        else:
            expected_nodes = 1
        gap_fraction = max(0.0, 1.0 - n_nodes / max(expected_nodes, 1))

        # Per-node geometric stats.
        ellipses = [node.ellipse for node in track.nodes]
        radii = np.array(
            [float(np.sqrt(e.a * e.b)) for e in ellipses if e.a > 0 and e.b > 0],
            dtype=np.float64,
        )
        aspects = np.array(
            [e.a / e.b for e in ellipses if e.b > 0],
            dtype=np.float64,
        )
        sector_pcts = np.array(
            [e.sector_pct for e in ellipses],
            dtype=np.float64,
        )

        mean_radius = float(radii.mean()) if radii.size else 0.0
        radius_cv = (
            float(radii.std() / mean_radius) if mean_radius > 0 else 0.0
        )
        mean_aspect_ratio = float(aspects.mean()) if aspects.size else 0.0
        mean_sector_pct = (
            float(sector_pcts.mean()) if sector_pcts.size else 0.0
        )

        n_points = int((tracking_result.tree_ids == i).sum())

        if mean_radius > 0.0 and z_extent > 0.0:
            volume = np.pi * mean_radius * mean_radius * z_extent
            density = float(n_points / volume)
        else:
            density = 0.0

        rows.append({
            "tree_id": i,
            "n_nodes": n_nodes,
            "z_bottom": round(z_bottom, 3),
            "z_top": round(z_top, 3),
            "z_extent": round(z_extent, 3),
            "gap_fraction": round(gap_fraction, 3),
            "n_points": n_points,
            "mean_radius": round(mean_radius, 4),
            "radius_cv": round(radius_cv, 3),
            "mean_aspect_ratio": round(mean_aspect_ratio, 3),
            "mean_sector_pct": round(mean_sector_pct, 1),
            "density_pts_per_m3": round(density, 1),
        })

    return pd.DataFrame(rows)


def export_tracking_audit_table(df, path):
    """Write the tracking-audit DataFrame to a CSV.

    Wrapper kept for symmetry with ``export_trunk_audit_table`` in
    ``src.core.trunk_audit`` so the notebook cells can use the same
    idiom for both audits.
    """
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p
