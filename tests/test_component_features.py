"""Tests for Brick 7.2 component-level geometric aggregation and filtering."""

from pathlib import Path
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    MultiScaleGeometricFeatures,
    ComponentGeometricRules,
    component_ids_to_point_mask,
    compute_component_feature_table,
    filter_components_by_geometric_rules,
)


def _build_synthetic_components():
    """
    Build three labelled components:
    - component 0: trunk-like (should pass)
    - component 1: shrub-like (should fail geometric thresholds)
    - component 2: tiny fragment (should fail min_points/min_height)
    """
    # Component 0
    c0_n = 240
    c0_xyz = np.column_stack(
        [
            np.random.default_rng(1).normal(0.0, 0.07, c0_n),
            np.random.default_rng(2).normal(0.0, 0.07, c0_n),
            np.linspace(0.0, 9.5, c0_n),
        ]
    )

    # Component 1
    c1_n = 220
    c1_xyz = np.column_stack(
        [
            np.random.default_rng(3).normal(3.0, 0.45, c1_n),
            np.random.default_rng(4).normal(3.0, 0.45, c1_n),
            np.random.default_rng(5).uniform(0.2, 2.0, c1_n),
        ]
    )

    # Component 2
    c2_n = 20
    c2_xyz = np.column_stack(
        [
            np.random.default_rng(6).normal(-2.0, 0.10, c2_n),
            np.random.default_rng(7).normal(-2.0, 0.10, c2_n),
            np.random.default_rng(8).uniform(0.0, 0.6, c2_n),
        ]
    )

    xyz = np.vstack([c0_xyz, c1_xyz, c2_xyz]).astype(np.float64)
    labels = np.concatenate(
        [
            np.zeros(c0_n, dtype=np.int32),
            np.ones(c1_n, dtype=np.int32),
            np.full(c2_n, 2, dtype=np.int32),
        ]
    )

    # Synthetic feature values with clear separation.
    verticality = np.concatenate(
        [
            np.full(c0_n, 0.75, dtype=np.float32),
            np.full(c1_n, 0.18, dtype=np.float32),
            np.full(c2_n, 0.40, dtype=np.float32),
        ]
    )
    linearity = np.concatenate(
        [
            np.full(c0_n, 0.62, dtype=np.float32),
            np.full(c1_n, 0.12, dtype=np.float32),
            np.full(c2_n, 0.22, dtype=np.float32),
        ]
    )
    sphericity = np.concatenate(
        [
            np.full(c0_n, 0.18, dtype=np.float32),
            np.full(c1_n, 0.82, dtype=np.float32),
            np.full(c2_n, 0.50, dtype=np.float32),
        ]
    )
    roughness = np.concatenate(
        [
            np.full(c0_n, 0.06, dtype=np.float32),
            np.full(c1_n, 0.36, dtype=np.float32),
            np.full(c2_n, 0.10, dtype=np.float32),
        ]
    )
    mean_curvature = np.concatenate(
        [
            np.full(c0_n, 0.35, dtype=np.float32),
            np.full(c1_n, 2.40, dtype=np.float32),
            np.full(c2_n, 0.90, dtype=np.float32),
        ]
    )
    gaussian_curvature = np.concatenate(
        [
            np.full(c0_n, 0.80, dtype=np.float32),
            np.full(c1_n, 4.60, dtype=np.float32),
            np.full(c2_n, 1.50, dtype=np.float32),
        ]
    )
    neighbour_count = np.concatenate(
        [
            np.full(c0_n, 42.0, dtype=np.float32),
            np.full(c1_n, 12.0, dtype=np.float32),
            np.full(c2_n, 8.0, dtype=np.float32),
        ]
    )
    surface_density = np.concatenate(
        [
            np.full(c0_n, 45.0, dtype=np.float32),
            np.full(c1_n, 6.0, dtype=np.float32),
            np.full(c2_n, 9.0, dtype=np.float32),
        ]
    )
    volume_density = np.concatenate(
        [
            np.full(c0_n, 90.0, dtype=np.float32),
            np.full(c1_n, 14.0, dtype=np.float32),
            np.full(c2_n, 12.0, dtype=np.float32),
        ]
    )

    features = MultiScaleGeometricFeatures(
        scales=(0.1, 0.2),
        verticality=verticality,
        linearity=linearity,
        planarity=np.zeros(len(xyz), dtype=np.float32),
        sphericity=sphericity,
        roughness=roughness,
        mean_curvature=mean_curvature,
        gaussian_curvature=gaussian_curvature,
        neighbor_count=neighbour_count,
        surface_density=surface_density,
        volume_density=volume_density,
        per_scale=None,
    )
    return xyz, labels, features


def test_component_feature_table_basic_properties():
    xyz, labels, features = _build_synthetic_components()

    table = compute_component_feature_table(
        xyz=xyz,
        component_labels=labels,
        features=features,
        ignore_label=-1,
    )

    assert np.array_equal(table.component_ids, np.array([0, 1, 2], dtype=np.int64))
    assert np.array_equal(table.point_count, np.array([240, 220, 20], dtype=np.int64))
    assert table.z_extent[0] > table.z_extent[1] > table.z_extent[2]
    assert table.verticality_mean[0] > table.verticality_mean[1]
    assert table.linearity_mean[0] > table.linearity_mean[1]
    assert table.sphericity_mean[1] > table.sphericity_mean[0]


def test_filter_components_by_geometric_rules_keeps_expected_component():
    xyz, labels, features = _build_synthetic_components()

    table = compute_component_feature_table(
        xyz=xyz,
        component_labels=labels,
        features=features,
    )

    rules = ComponentGeometricRules(
        min_points=100,
        min_height=2.0,
        min_verticality_mean=0.30,
        min_linearity_mean=0.20,
        max_sphericity_mean=0.60,
        max_roughness_mean=0.25,
        max_mean_curvature_mean=2.00,
        max_gaussian_curvature_mean=4.00,
        min_surface_density_mean=10.0,
        min_volume_density_mean=20.0,
    )
    result = filter_components_by_geometric_rules(table, rules=rules)

    assert np.array_equal(result.kept_component_ids, np.array([0], dtype=np.int64))
    assert set(result.rejected_component_ids.tolist()) == {1, 2}
    assert 1 in result.rejection_reasons
    assert 2 in result.rejection_reasons

    point_mask = component_ids_to_point_mask(labels, result.kept_component_ids)
    assert int(np.sum(point_mask)) == 240
    assert np.all(labels[point_mask] == 0)


def test_component_ids_to_point_mask_with_empty_selection():
    labels = np.array([0, 0, 1, 2, 2], dtype=np.int32)
    mask = component_ids_to_point_mask(labels, kept_component_ids=[])
    assert mask.shape == labels.shape
    assert not np.any(mask)
