import numpy as np

from src.core.trunk_extraction import TrunkExtractionConfig, TrunkExtractionResult
from src.core.trunk_validation import StemCleaningConfig, clean_stems


def _make_trunk_result() -> TrunkExtractionResult:
    trunk_mask = np.array([True, True, True, True, True, True, True, True], dtype=bool)
    tree_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    tree_axes = [
        {
            "tree_id": 0,
            "centroid": np.array([0.0, 0.0, 1.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "n_points": 4,
            "z_min": 0.5,
            "z_max": 2.0,
            "stripe_points": 4,
            "stripe_circularity": 0.90,
            "stripe_diameter": 0.18,
            "stripe_z_span": 1.5,
        },
        {
            "tree_id": 1,
            "centroid": np.array([10.0, 0.0, 1.0]),
            "direction": np.array([0.35, 0.0, 0.94]),
            "n_points": 4,
            "z_min": 0.5,
            "z_max": 2.0,
            "stripe_points": 4,
            "stripe_circularity": 0.80,
            "stripe_diameter": 0.18,
            "stripe_z_span": 1.5,
        },
    ]
    return TrunkExtractionResult(
        trunk_mask=trunk_mask,
        tree_ids=tree_ids,
        n_trees=2,
        tree_axes=tree_axes,
        cluster_points=np.empty((0, 3), dtype=np.float64),
        config=TrunkExtractionConfig(),
    )


def test_clean_stems_global_keeps_reference_behavior(monkeypatch):
    xyz = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 1.1],
            [0.1, 0.1, 1.4],
            [10.0, 0.0, 0.5],
            [10.0, 0.1, 0.8],
            [10.1, 0.0, 1.1],
            [10.1, 0.1, 1.4],
        ],
        dtype=np.float64,
    )
    trunk_result = _make_trunk_result()

    def fake_compute_verticality(points, scale, voxel_resolution_xy, voxel_resolution_z):
        assert len(points) == 8
        return np.array([0.9, 0.4, 0.8, 0.3, 0.95, 0.2, 0.7, 0.1], dtype=np.float64)

    monkeypatch.setattr("src.core.trunk_validation.compute_verticality", fake_compute_verticality)

    result = clean_stems(
        xyz,
        trunk_result,
        StemCleaningConfig(mode="global", verticality_threshold=0.5),
        verbose=False,
    )

    assert np.array_equal(result.stem_mask, np.array([True, False, True, False, True, False, True, False]))
    assert result.mode_used == "global"
    assert result.n_trees_processed == 2
    assert result.n_trees_skipped == 0
    assert result.n_points_processed_verticality == 8
    assert result.used_global_fallback is False
    assert result.per_tree_stats == [
        {"tree_id": 0, "before": 4, "after": 2, "removed": 2, "pct_removed": 50.0},
        {"tree_id": 1, "before": 4, "after": 2, "removed": 2, "pct_removed": 50.0},
    ]


def test_clean_stems_suspicious_only_skips_safe_trees(monkeypatch):
    xyz = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 1.1],
            [0.1, 0.1, 1.4],
            [10.0, 0.0, 0.5],
            [10.0, 0.1, 0.8],
            [10.1, 0.0, 1.1],
            [10.1, 0.1, 1.4],
        ],
        dtype=np.float64,
    )
    trunk_result = _make_trunk_result()

    calls = []

    def fake_compute_verticality(points, scale, voxel_resolution_xy, voxel_resolution_z):
        calls.append(len(points))
        assert len(points) == 4
        assert float(points[:, 0].mean()) > 5.0
        return np.array([0.9, 0.3, 0.8, 0.2], dtype=np.float64)

    monkeypatch.setattr("src.core.trunk_validation.compute_verticality", fake_compute_verticality)

    result = clean_stems(
        xyz,
        trunk_result,
        StemCleaningConfig(
            mode="suspicious_only",
            verticality_threshold=0.5,
            suspicious_axis_tilt_deg=10.0,
            global_fallback_tree_ratio=0.75,
            global_fallback_point_ratio=0.75,
        ),
        verbose=False,
    )

    assert calls == [4]
    assert np.array_equal(result.stem_mask, np.array([True, True, True, True, True, False, True, False]))
    assert result.mode_used == "suspicious_only"
    assert result.n_trees_processed == 1
    assert result.n_trees_skipped == 1
    assert result.n_points_processed_verticality == 4
    assert result.used_global_fallback is False
    assert result.per_tree_stats == [
        {"tree_id": 0, "before": 4, "after": 4, "removed": 0, "pct_removed": 0.0},
        {"tree_id": 1, "before": 4, "after": 2, "removed": 2, "pct_removed": 50.0},
    ]


def test_clean_stems_suspicious_only_falls_back_to_global(monkeypatch):
    xyz = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 1.1],
            [0.1, 0.1, 1.4],
            [10.0, 0.0, 0.5],
            [10.0, 0.1, 0.8],
            [10.1, 0.0, 1.1],
            [10.1, 0.1, 1.4],
        ],
        dtype=np.float64,
    )
    trunk_result = _make_trunk_result()
    for axis in trunk_result.tree_axes:
        axis["stripe_circularity"] = 0.20

    def fake_compute_verticality(points, scale, voxel_resolution_xy, voxel_resolution_z):
        assert len(points) == 8
        return np.array([0.9, 0.4, 0.8, 0.3, 0.95, 0.2, 0.7, 0.1], dtype=np.float64)

    monkeypatch.setattr("src.core.trunk_validation.compute_verticality", fake_compute_verticality)

    global_result = clean_stems(
        xyz,
        trunk_result,
        StemCleaningConfig(mode="global", verticality_threshold=0.5),
        verbose=False,
    )
    fallback_result = clean_stems(
        xyz,
        trunk_result,
        StemCleaningConfig(
            mode="suspicious_only",
            verticality_threshold=0.5,
            suspicious_stripe_circularity_max=0.45,
            global_fallback_tree_ratio=0.50,
            global_fallback_point_ratio=0.50,
        ),
        verbose=False,
    )

    assert np.array_equal(fallback_result.stem_mask, global_result.stem_mask)
    assert fallback_result.mode_used == "global"
    assert fallback_result.used_global_fallback is True
    assert fallback_result.n_points_processed_verticality == global_result.n_points_processed_verticality
