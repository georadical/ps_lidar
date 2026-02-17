"""Tests for optimized feature backend selection and benchmarking."""

from pathlib import Path
import sys

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.features import compute_all_features_fast, benchmark_feature_backends


def _synthetic_cloud(n: int = 4000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mixed vertical + scattered structure
    trunk = np.column_stack(
        [
            rng.normal(0.0, 0.12, n // 2),
            rng.normal(0.0, 0.12, n // 2),
            rng.uniform(0.0, 12.0, n // 2),
        ]
    )
    canopy = np.column_stack(
        [
            rng.normal(0.0, 1.5, n // 2),
            rng.normal(0.0, 1.5, n // 2),
            rng.uniform(8.0, 16.0, n // 2),
        ]
    )
    return np.vstack([trunk, canopy]).astype(np.float64)


def test_compute_all_features_fast_default_backend():
    xyz = _synthetic_cloud()
    features, dist_to_ground, dist_to_top = compute_all_features_fast(xyz, verbose=False)

    assert len(features.verticality) == len(xyz)
    assert len(dist_to_ground) == len(xyz)
    assert len(dist_to_top) == len(xyz)


def test_compute_all_features_fast_invalid_backend():
    xyz = _synthetic_cloud()
    with pytest.raises(ValueError):
        compute_all_features_fast(xyz, backend="invalid", verbose=False)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pgeof") is None,
    reason="pgeof not installed",
)
def test_compute_all_features_fast_pgeof_backend():
    xyz = _synthetic_cloud()
    features, dist_to_ground, dist_to_top = compute_all_features_fast(
        xyz,
        backend="pgeof",
        verbose=False,
    )

    assert len(features.verticality) == len(xyz)
    assert features.eigenvalues.shape == (len(xyz), 3)
    assert features.normals.shape == (len(xyz), 3)
    assert len(dist_to_ground) == len(xyz)
    assert len(dist_to_top) == len(xyz)


def test_benchmark_feature_backends_voxel_only():
    xyz = _synthetic_cloud(n=2000)
    report = benchmark_feature_backends(
        xyz,
        backends=("voxel",),
        repeats=1,
        sample_size=1000,
        verbose=False,
    )

    assert "metadata" in report
    assert "backends" in report
    assert report["backends"]["voxel"]["ok"] is True
