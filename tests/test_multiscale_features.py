"""Tests for Brick 7.1 multiscale geometric feature extraction."""

from pathlib import Path
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import compute_multiscale_geometric_features


def _synthetic_cloud(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Build synthetic cloud with two structures:
    - trunk-like vertical cylinder (label 1)
    - understory-like spherical cluster (label 0)
    """
    rng = np.random.default_rng(seed)

    n_trunk = 1200
    angles = rng.uniform(0.0, 2.0 * np.pi, n_trunk)
    radii = rng.normal(0.08, 0.01, n_trunk)
    z = rng.uniform(0.0, 10.0, n_trunk)
    trunk = np.column_stack(
        [
            radii * np.cos(angles),
            radii * np.sin(angles),
            z,
        ]
    )

    n_understory = 1200
    center = np.array([2.0, 2.0, 1.0])
    understory = center + rng.normal(0.0, [0.35, 0.35, 0.35], size=(n_understory, 3))
    understory[:, 2] = np.clip(understory[:, 2], 0.0, 2.0)

    xyz = np.vstack([trunk, understory]).astype(np.float64)
    labels = np.concatenate(
        [
            np.ones(n_trunk, dtype=np.int32),
            np.zeros(n_understory, dtype=np.int32),
        ]
    )
    return xyz, labels


def test_multiscale_feature_shapes_and_finite_values():
    xyz, _ = _synthetic_cloud()

    result = compute_multiscale_geometric_features(
        xyz,
        scales=(0.10, 0.20, 0.35),
        voxel_size=0.08,
        min_neighbors=8,
        return_per_scale=False,
        verbose=False,
    )

    n = len(xyz)
    assert result.scales == (0.1, 0.2, 0.35)

    for name in (
        "verticality",
        "linearity",
        "planarity",
        "sphericity",
        "roughness",
        "mean_curvature",
        "gaussian_curvature",
        "neighbor_count",
        "surface_density",
        "volume_density",
    ):
        values = getattr(result, name)
        assert values.shape == (n,)
        assert np.all(np.isfinite(values))

    assert np.all(result.neighbor_count > 0)
    assert np.all(result.surface_density > 0)
    assert np.all(result.volume_density > 0)


def test_multiscale_features_separate_trunk_and_understory_signal():
    xyz, labels = _synthetic_cloud(seed=7)

    result = compute_multiscale_geometric_features(
        xyz,
        scales=(0.12, 0.25),
        voxel_size=0.08,
        min_neighbors=10,
        return_per_scale=False,
        verbose=False,
    )

    trunk_mask = labels == 1
    understory_mask = labels == 0

    trunk_verticality = float(np.mean(result.verticality[trunk_mask]))
    understory_verticality = float(np.mean(result.verticality[understory_mask]))
    trunk_linearity = float(np.mean(result.linearity[trunk_mask]))
    understory_linearity = float(np.mean(result.linearity[understory_mask]))
    trunk_sphericity = float(np.mean(result.sphericity[trunk_mask]))
    understory_sphericity = float(np.mean(result.sphericity[understory_mask]))

    assert trunk_verticality > understory_verticality
    assert trunk_linearity > understory_linearity
    assert understory_sphericity > trunk_sphericity


def test_multiscale_feature_return_per_scale_shapes():
    xyz, _ = _synthetic_cloud(seed=99)
    scales = (0.1, 0.2, 0.4)

    result = compute_multiscale_geometric_features(
        xyz,
        scales=scales,
        voxel_size=0.1,
        min_neighbors=8,
        return_per_scale=True,
        verbose=False,
    )

    assert result.per_scale is not None
    assert set(result.per_scale.keys()) == {
        "verticality",
        "linearity",
        "planarity",
        "sphericity",
        "roughness",
        "mean_curvature",
        "gaussian_curvature",
        "neighbor_count",
        "surface_density",
        "volume_density",
    }

    n = len(xyz)
    s = len(scales)
    for values in result.per_scale.values():
        assert values.shape == (n, s)
