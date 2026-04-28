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


# ---------------------------------------------------------------------------
# Profiling instrumentation tests
# ---------------------------------------------------------------------------

_GLOBAL_PROFILE_KEYS = {
    "total_seconds",
    "input_preparation_seconds",
    "verticality_computation_seconds",
    "threshold_filtering_seconds",
    "mask_update_seconds",
    "stats_aggregation_seconds",
    "n_points_verticality",
}

_SUSPICIOUS_PROFILE_KEYS = _GLOBAL_PROFILE_KEYS | {
    "suspicious_scoring_seconds",
    "n_trees_suspicious",
    "n_trees_skipped",
    "per_tree_timings",
}


def test_profile_populated_in_global_mode(monkeypatch):
    """Profile dict must be present with correct keys in global mode."""
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
        return np.array([0.9, 0.4, 0.8, 0.3, 0.95, 0.2, 0.7, 0.1], dtype=np.float64)

    monkeypatch.setattr("src.core.trunk_validation.compute_verticality", fake_compute_verticality)

    result = clean_stems(
        xyz,
        trunk_result,
        StemCleaningConfig(mode="global", verticality_threshold=0.5),
        verbose=False,
    )

    assert result.profile is not None
    assert _GLOBAL_PROFILE_KEYS.issubset(result.profile.keys())
    assert result.profile["total_seconds"] > 0.0
    assert result.profile["n_points_verticality"] == 8
    # Confirm outputs unchanged by profiling
    assert np.array_equal(result.stem_mask, np.array([True, False, True, False, True, False, True, False]))


def test_profile_populated_in_suspicious_only_mode(monkeypatch):
    """Profile dict must include per-tree timings in suspicious_only mode."""
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

    assert result.profile is not None
    assert _SUSPICIOUS_PROFILE_KEYS.issubset(result.profile.keys())
    assert result.profile["total_seconds"] > 0.0
    assert result.profile["n_trees_suspicious"] == 1
    assert result.profile["n_trees_skipped"] == 1
    assert len(result.profile["per_tree_timings"]) == 1
    tree_timing = result.profile["per_tree_timings"][0]
    assert tree_timing["tree_id"] == 1
    assert tree_timing["n_points"] == 4
    assert tree_timing["verticality_seconds"] >= 0.0
    # Confirm outputs unchanged
    assert np.array_equal(result.stem_mask, np.array([True, True, True, True, True, False, True, False]))


def test_profile_populated_on_global_fallback(monkeypatch):
    """Fallback path must merge suspicious_scoring_seconds into the global profile."""
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
        return np.array([0.9, 0.4, 0.8, 0.3, 0.95, 0.2, 0.7, 0.1], dtype=np.float64)

    monkeypatch.setattr("src.core.trunk_validation.compute_verticality", fake_compute_verticality)

    result = clean_stems(
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

    assert result.used_global_fallback is True
    assert result.profile is not None
    assert "suspicious_scoring_seconds" in result.profile
    assert result.profile["suspicious_scoring_seconds"] > 0.0


# ---------------------------------------------------------------------------
# Parallel execution tests
# ---------------------------------------------------------------------------


def test_parallel_suspicious_only_matches_serial(monkeypatch):
    """Parallel and serial suspicious_only must produce identical stem_mask."""
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
    # Make both trees suspicious via tilt
    trunk_result.tree_axes[0]["direction"] = np.array([0.35, 0.0, 0.94])
    trunk_result.tree_axes[1]["direction"] = np.array([0.35, 0.0, 0.94])

    def fake_compute_verticality(points, scale, voxel_resolution_xy, voxel_resolution_z):
        n = len(points)
        # Alternating high/low verticality
        return np.array([0.9, 0.3] * (n // 2), dtype=np.float64)[:n]

    monkeypatch.setattr("src.core.trunk_validation.compute_verticality", fake_compute_verticality)

    common = dict(
        mode="suspicious_only",
        verticality_threshold=0.5,
        suspicious_axis_tilt_deg=10.0,
        # Set above 1.0 to never trigger fallback even if 100% are suspicious
        global_fallback_tree_ratio=1.10,
        global_fallback_point_ratio=1.10,
    )

    result_serial = clean_stems(
        xyz, trunk_result,
        StemCleaningConfig(**common, parallel=False),
        verbose=False,
    )
    result_parallel = clean_stems(
        xyz, trunk_result,
        StemCleaningConfig(**common, parallel=True, parallel_max_workers=2),
        verbose=False,
    )

    assert np.array_equal(result_serial.stem_mask, result_parallel.stem_mask)
    assert result_serial.n_points_removed == result_parallel.n_points_removed
    assert result_serial.per_tree_stats == result_parallel.per_tree_stats
    assert result_parallel.profile is not None
    assert result_parallel.profile["parallel"] is True
    assert result_serial.profile["parallel"] is False


def test_parallel_flag_ignored_with_single_suspicious_tree(monkeypatch):
    """When only 1 tree is suspicious, parallel=True falls back to serial."""
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
            parallel=True,
            parallel_max_workers=2,
        ),
        verbose=False,
    )

    # Only tree 1 is suspicious (tilt > 10°), so parallel should be unused
    assert result.n_trees_processed == 1
    assert result.profile is not None
    assert result.profile["parallel"] is False

