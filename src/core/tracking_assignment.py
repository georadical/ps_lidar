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

from typing import Tuple

import numpy as np

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
