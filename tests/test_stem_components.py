"""Tests for Brick 7.2 stem seeds and connected components."""

from pathlib import Path
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    MultiScaleGeometricFeatures,
    extract_stem_seed_mask,
    label_stem_seed_components,
)


def _make_cloud_and_features(seed: int = 42):
    rng = np.random.default_rng(seed)

    # Two trunk-like components inside stripe
    n_a = 220
    n_b = 210

    angles_a = rng.uniform(0.0, 2.0 * np.pi, n_a)
    angles_b = rng.uniform(0.0, 2.0 * np.pi, n_b)
    radii_a = rng.normal(0.08, 0.01, n_a)
    radii_b = rng.normal(0.09, 0.01, n_b)

    trunk_a = np.column_stack(
        [
            -1.5 + radii_a * np.cos(angles_a),
            -0.5 + radii_a * np.sin(angles_a),
            rng.uniform(1.6, 4.2, n_a),
        ]
    )
    trunk_b = np.column_stack(
        [
            1.5 + radii_b * np.cos(angles_b),
            0.8 + radii_b * np.sin(angles_b),
            rng.uniform(1.7, 4.3, n_b),
        ]
    )

    # Shrub-like points in stripe but poor trunk geometry
    n_shrub = 250
    shrub = np.column_stack(
        [
            rng.normal(0.0, 0.45, n_shrub),
            rng.normal(0.0, 0.45, n_shrub),
            rng.uniform(1.7, 3.9, n_shrub),
        ]
    )

    xyz = np.vstack([trunk_a, trunk_b, shrub]).astype(np.float64)
    labels_true = np.concatenate(
        [
            np.zeros(n_a, dtype=np.int32),
            np.ones(n_b, dtype=np.int32),
            np.full(n_shrub, 2, dtype=np.int32),
        ]
    )

    verticality = np.concatenate(
        [
            np.full(n_a, 0.80, dtype=np.float32),
            np.full(n_b, 0.78, dtype=np.float32),
            np.full(n_shrub, 0.22, dtype=np.float32),
        ]
    )
    linearity = np.concatenate(
        [
            np.full(n_a, 0.60, dtype=np.float32),
            np.full(n_b, 0.58, dtype=np.float32),
            np.full(n_shrub, 0.15, dtype=np.float32),
        ]
    )
    sphericity = np.concatenate(
        [
            np.full(n_a, 0.20, dtype=np.float32),
            np.full(n_b, 0.22, dtype=np.float32),
            np.full(n_shrub, 0.78, dtype=np.float32),
        ]
    )
    roughness = np.concatenate(
        [
            np.full(n_a, 0.08, dtype=np.float32),
            np.full(n_b, 0.09, dtype=np.float32),
            np.full(n_shrub, 0.32, dtype=np.float32),
        ]
    )
    surface_density = np.concatenate(
        [
            np.full(n_a, 40.0, dtype=np.float32),
            np.full(n_b, 38.0, dtype=np.float32),
            np.full(n_shrub, 6.0, dtype=np.float32),
        ]
    )
    volume_density = np.concatenate(
        [
            np.full(n_a, 85.0, dtype=np.float32),
            np.full(n_b, 82.0, dtype=np.float32),
            np.full(n_shrub, 12.0, dtype=np.float32),
        ]
    )

    features = MultiScaleGeometricFeatures(
        scales=(0.1, 0.2),
        verticality=verticality,
        linearity=linearity,
        planarity=np.zeros(len(xyz), dtype=np.float32),
        sphericity=sphericity,
        roughness=roughness,
        mean_curvature=np.zeros(len(xyz), dtype=np.float32),
        gaussian_curvature=np.zeros(len(xyz), dtype=np.float32),
        neighbor_count=np.zeros(len(xyz), dtype=np.float32),
        surface_density=surface_density,
        volume_density=volume_density,
        per_scale=None,
    )
    return xyz, labels_true, features


def test_extract_stem_seed_mask_rejects_shrub_like_points():
    xyz, labels_true, features = _make_cloud_and_features(seed=11)

    result = extract_stem_seed_mask(
        xyz=xyz,
        features=features,
        z_min_stem=1.5,
        z_max_stem=4.5,
        min_verticality=0.70,
        min_linearity=0.30,
        max_sphericity=0.55,
        max_roughness=0.22,
        min_surface_density=10.0,
        min_volume_density=20.0,
    )

    assert result.n_points == len(xyz)
    assert result.n_stripe == len(xyz)  # all synthetic points are in stripe
    assert result.n_seeds > 0

    shrub_seed_fraction = float(np.mean(result.seed_mask[labels_true == 2]))
    trunk_seed_fraction = float(np.mean(result.seed_mask[labels_true <= 1]))
    assert shrub_seed_fraction < 0.05
    assert trunk_seed_fraction > 0.90


def test_label_stem_seed_components_finds_two_components():
    xyz, labels_true, features = _make_cloud_and_features(seed=19)
    seed_result = extract_stem_seed_mask(xyz, features)

    cc_result = label_stem_seed_components(
        xyz=xyz,
        seed_mask=seed_result.seed_mask,
        voxel_size=0.12,
        min_component_points=80,
    )

    assert cc_result.n_components_raw >= 2
    assert cc_result.n_components_kept == 2
    assert len(cc_result.component_point_counts) == 2

    labels = cc_result.component_labels
    assert labels.shape == (len(xyz),)
    assert np.all(labels[~seed_result.seed_mask] == -1)
    assert set(np.unique(labels[labels >= 0]).tolist()) == {0, 1}


def test_label_stem_seed_components_handles_no_seeds():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    seed_mask = np.array([False, False], dtype=bool)

    cc_result = label_stem_seed_components(
        xyz=xyz,
        seed_mask=seed_mask,
        voxel_size=0.1,
        min_component_points=5,
    )

    assert cc_result.n_components_raw == 0
    assert cc_result.n_components_kept == 0
    assert np.all(cc_result.component_labels == -1)
