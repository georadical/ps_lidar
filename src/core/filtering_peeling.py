"""
Compatibility wrapper for iterative peeling.

The canonical implementation lives in `src.core.peeling`.
This module is kept only to avoid breaking older imports.
"""

import numpy as np
import warnings

from .peeling import (
    IterativePeelingResult,
    iterative_peeling_understory as _iterative_peeling_understory_impl,
)


def iterative_peeling_understory(
    xyz: np.ndarray,
    verticality: np.ndarray,
    linearity: np.ndarray,
    sphericity: np.ndarray,
    dist_to_ground: np.ndarray,
    seed_verticality: float = 0.9,
    seed_linearity: float = 0.6,
    seed_height_min: float = 1.0,
    seed_height_max: float = 2.5,
    expansion_verticality: float = 0.5,
    expansion_radius: float = 0.3,
    max_iterations: int = 50,
    verbose: bool = False,
) -> IterativePeelingResult:
    """
    Backward-compatible wrapper.

    Notes
    -----
    - `expansion_radius` is accepted for API compatibility but ignored.
    - Uses the voxel-based implementation from `src.core.peeling`.
    """
    warnings.warn(
        "src.core.filtering_peeling.iterative_peeling_understory is deprecated; "
        "use src.core.peeling.iterative_peeling_understory instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    _ = expansion_radius
    return _iterative_peeling_understory_impl(
        xyz=xyz,
        verticality=verticality,
        linearity=linearity,
        sphericity=sphericity,
        dist_to_ground=dist_to_ground,
        seed_verticality=seed_verticality,
        seed_linearity=seed_linearity,
        seed_height_min=seed_height_min,
        seed_height_max=seed_height_max,
        expansion_verticality=expansion_verticality,
        max_iterations=max_iterations,
        verbose=verbose,
    )
