import numpy as np

from src.core.trunk_extraction import (
    TrunkExtractionConfig,
    _detect_basal_tracks,
    _match_basal_tracks_to_axes,
    _refine_axes_with_basal_anchor,
    extract_trunks,
)


def _make_arc(
    center_xy: tuple[float, float],
    radius_x: float,
    radius_y: float,
    z: float,
    angle_start_deg: float = 20.0,
    angle_end_deg: float = 220.0,
    n_points: int = 80,
) -> np.ndarray:
    angles = np.deg2rad(np.linspace(angle_start_deg, angle_end_deg, n_points))
    x = center_xy[0] + radius_x * np.cos(angles)
    y = center_xy[1] + radius_y * np.sin(angles)
    z_arr = np.full_like(x, z, dtype=np.float64)
    return np.column_stack([x, y, z_arr]).astype(np.float64)


def _make_ring(
    center_xy: tuple[float, float],
    radius_x: float,
    radius_y: float,
    z: float,
    n_points: int = 72,
) -> np.ndarray:
    return _make_arc(center_xy, radius_x, radius_y, z, 0.0, 360.0, n_points=n_points)


def _make_stem_with_basal_arcs(
    base_center: tuple[float, float] = (0.0, 0.0),
    stripe_center: tuple[float, float] = (0.02, 0.01),
    radius_x: float = 0.12,
    radius_y: float = 0.08,
    basal_levels: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35),
    with_outliers: bool = False,
) -> np.ndarray:
    points = []
    for z in basal_levels:
        arc = _make_arc(base_center, radius_x, radius_y, z)
        if with_outliers:
            rng = np.random.default_rng(int(z * 1000))
            noise_xy = rng.uniform(-0.05, 0.05, size=(16, 2)) + np.array(base_center)
            noise_z = np.full((16, 1), z, dtype=np.float64)
            arc = np.vstack([arc, np.hstack([noise_xy, noise_z])])
        points.append(arc)

    for z in np.arange(1.05, 1.55, 0.05):
        points.append(_make_ring(stripe_center, radius_x, radius_x, float(z)))

    return np.vstack(points)


def _make_axis(tree_id: int, centroid: tuple[float, float, float], stripe_diameter: float) -> dict:
    return {
        "tree_id": tree_id,
        "centroid": np.array(centroid, dtype=np.float64),
        "direction": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "stripe_diameter": stripe_diameter,
        "n_points": 100,
    }


def _common_config(**overrides) -> TrunkExtractionConfig:
    defaults = dict(
        stripe_lower_limit=1.0,
        stripe_upper_limit=1.5,
        dbh_min=0.20,
        dbh_max=0.80,
        basal_anchor_min_points=20,
        basal_anchor_cluster_eps=0.05,
        basal_anchor_min_support_slices=3,
        basal_anchor_min_arc_coverage=0.18,
        basal_anchor_max_fit_residual_ratio=0.18,
        basal_anchor_max_center_drift=0.08,
        basal_anchor_max_axis_ratio=2.2,
        basal_anchor_match_max_xy_distance=0.75,
        basal_anchor_match_max_tilt_deg=30.0,
    )
    defaults.update(overrides)
    return TrunkExtractionConfig(**defaults)


def test_detect_basal_tracks_finds_multislice_partial_arc_without_pca_guidance():
    xyz = _make_stem_with_basal_arcs()
    tracks, total_candidates, validated_candidates = _detect_basal_tracks(xyz, _common_config())

    assert total_candidates >= 1
    assert validated_candidates >= 1
    assert len(tracks) >= 1
    assert tracks[0]["support_slices"] >= 3
    assert tracks[0]["model"] in {"circle", "ellipse"}
    assert tracks[0]["track_score"] >= 0.0


def test_detect_basal_tracks_rejects_single_slice_arc():
    xyz = _make_stem_with_basal_arcs(basal_levels=(0.25,))
    tracks, total_candidates, validated_candidates = _detect_basal_tracks(xyz, _common_config())

    assert total_candidates >= 1
    assert validated_candidates == 0
    assert tracks == []


def test_detect_basal_tracks_survives_outliers():
    xyz = _make_stem_with_basal_arcs(with_outliers=True)
    tracks, _, validated_candidates = _detect_basal_tracks(
        xyz,
        _common_config(basal_anchor_max_fit_residual_ratio=0.22),
    )

    assert validated_candidates >= 1
    assert len(tracks) >= 1
    assert tracks[0]["fit_residual"] <= 0.22


def test_match_basal_tracks_is_exclusive_when_one_track_competes_for_two_trees():
    xyz = _make_stem_with_basal_arcs()
    tracks, _, validated_candidates = _detect_basal_tracks(xyz, _common_config())
    assert validated_candidates >= 1

    axes = [
        _make_axis(0, (0.02, 0.01, 1.25), 0.24),
        _make_axis(1, (0.28, 0.02, 1.25), 0.24),
    ]
    best_by_tree, assigned_by_tree = _match_basal_tracks_to_axes(tracks, axes, _common_config())

    assert len(best_by_tree) >= 1
    assert len(assigned_by_tree) == 1
    assert 0 in assigned_by_tree
    assert assigned_by_tree[0]["track_id"] == tracks[0]["track_id"]


def test_match_basal_tracks_rejects_track_without_compatible_stripe():
    xyz = _make_stem_with_basal_arcs()
    tracks, _, validated_candidates = _detect_basal_tracks(xyz, _common_config())
    assert validated_candidates >= 1

    axes = [_make_axis(0, (1.50, 1.50, 1.25), 0.24)]
    best_by_tree, assigned_by_tree = _match_basal_tracks_to_axes(tracks, axes, _common_config())

    assert best_by_tree == {}
    assert assigned_by_tree == {}


def test_detect_basal_tracks_prefers_ellipse_for_elliptical_base():
    xyz = _make_stem_with_basal_arcs(radius_x=0.16, radius_y=0.08)
    tracks, _, validated_candidates = _detect_basal_tracks(xyz, _common_config())

    assert validated_candidates >= 1
    assert tracks[0]["model"] == "ellipse"


def test_detect_basal_tracks_prefers_circle_for_nearly_circular_base():
    xyz = _make_stem_with_basal_arcs(radius_x=0.12, radius_y=0.12)
    tracks, _, validated_candidates = _detect_basal_tracks(xyz, _common_config())

    assert validated_candidates >= 1
    assert tracks[0]["model"] == "circle"


def test_refine_axes_with_basal_anchor_sets_line_point_and_direction():
    xyz = _make_stem_with_basal_arcs()
    axes = [_make_axis(0, (0.02, 0.01, 1.25), 0.24)]

    refined = _refine_axes_with_basal_anchor(xyz, axes, _common_config())

    assert len(refined) == 1
    assert refined[0]["axis_source"] == "basal_anchor"
    assert refined[0]["basal_anchor_applied"] is True
    assert refined[0]["basal_anchor_track_id"] >= 0
    assert refined[0]["direction"][2] > 0.0
    assert not np.allclose(refined[0]["line_point"], refined[0]["centroid"])


def test_extract_trunks_basal_anchor_mode_matches_none_when_no_tracks(monkeypatch):
    xyz = _make_stem_with_basal_arcs(radius_x=0.12, radius_y=0.12)

    monkeypatch.setattr(
        "src.core.trunk_extraction.compute_verticality",
        lambda points, scale, voxel_resolution_xy, voxel_resolution_z: np.ones(len(points), dtype=np.float64),
    )
    monkeypatch.setattr(
        "src.core.trunk_extraction._detect_basal_tracks",
        lambda xyz, cfg: ([], 0, 0),
    )

    common = dict(
        stripe_lower_limit=1.0,
        stripe_upper_limit=1.5,
        dbh_min=0.20,
        dbh_max=0.80,
        verticality_threshold=0.5,
        peeling_iterations=1,
        min_cluster_points=10,
        voxel_resolution_xy=0.05,
        voxel_resolution_z=0.05,
        height_range=0.5,
        cluster_circularity_min=0.05,
        cluster_diameter_max_factor=3.0,
        stem_search_radius=0.30,
        max_axis_distance=1.0,
    )

    result_none = extract_trunks(
        xyz,
        TrunkExtractionConfig(axis_refinement_mode="none", **common),
        verbose=False,
    )
    result_anchor = extract_trunks(
        xyz,
        TrunkExtractionConfig(axis_refinement_mode="basal_anchor", **common),
        verbose=False,
    )

    assert result_none.n_trees == result_anchor.n_trees
    assert np.array_equal(result_none.tree_ids, result_anchor.tree_ids)
    assert np.array_equal(result_none.trunk_mask, result_anchor.trunk_mask)
    assert result_anchor.tree_axes[0]["axis_source"] == "pca"
    assert np.allclose(result_anchor.tree_axes[0]["line_point"], result_anchor.tree_axes[0]["centroid"])
