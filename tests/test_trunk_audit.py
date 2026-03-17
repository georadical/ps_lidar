import numpy as np
import pandas as pd
from pathlib import Path

from src.core.trunk_audit import build_trunk_audit_table, export_trunk_audit_table
from src.core.trunk_extraction import TrunkExtractionConfig, TrunkExtractionResult
from src.core.trunk_validation import SectionResult, StemCleaningConfig, StemCleaningResult


def test_build_trunk_audit_table_computes_expected_metrics(monkeypatch):
    xyz = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 1.2],
        [0.1, 0.0, 1.4],
        [0.2, 0.1, 1.6],
        [2.0, 0.0, 1.0],
        [2.1, 0.1, 1.2],
        [2.2, 0.0, 1.4],
        [2.3, 0.1, 1.6],
    ], dtype=np.float64)

    trunk_result = TrunkExtractionResult(
        trunk_mask=np.array([True, True, False, False, True, True, False, False]),
        tree_ids=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        n_trees=2,
        tree_axes=[
            {
                "tree_id": 0,
                "centroid": np.array([0.05, 0.05, 1.3]),
                "direction": np.array([0.0, 0.1, 0.995]),
                "n_points": 4,
                "z_min": 1.0,
                "z_max": 1.6,
                "stripe_points": 4,
                "stripe_diameter": 0.22,
                "stripe_circularity": 0.82,
                "stripe_z_span": 0.6,
                "seed_selection_mode": "microband",
                "seed_points": 96,
                "seed_z_min": 1.0,
                "seed_z_max": 1.6,
                "seed_z_span": 0.6,
                "seed_mean_circularity": 0.80,
                "seed_min_circularity": 0.72,
                "seed_radius_cv": 0.08,
                "seed_center_step_mean": 0.02,
                "seed_center_step_max": 0.03,
                "seed_mean_dominant_fraction": 0.95,
                "seed_min_dominant_fraction": 0.90,
                "seed_n_slices": 6,
                "seed_radius_ref": 0.11,
                "growth_slices_up": 3,
                "growth_slices_down": 4,
                "growth_points_kept": 8,
                "growth_stop_reason_up": "cloud_extent",
                "growth_stop_reason_down": "max_empty_slices",
            },
            {
                "tree_id": 1,
                "centroid": np.array([2.15, 0.05, 1.3]),
                "direction": np.array([0.2, 0.0, 0.98]),
                "n_points": 4,
                "z_min": 1.0,
                "z_max": 1.6,
                "stripe_points": 4,
                "stripe_diameter": 0.35,
                "stripe_circularity": 0.41,
                "stripe_z_span": 0.6,
                "seed_selection_mode": "full_stripe_short",
                "seed_points": 84,
                "seed_z_min": 1.0,
                "seed_z_max": 1.6,
                "seed_z_span": 0.6,
                "seed_mean_circularity": 0.39,
                "seed_min_circularity": 0.31,
                "seed_radius_cv": 0.15,
                "seed_center_step_mean": 0.04,
                "seed_center_step_max": 0.06,
                "seed_mean_dominant_fraction": 0.88,
                "seed_min_dominant_fraction": 0.80,
                "seed_n_slices": 6,
                "seed_radius_ref": 0.17,
                "growth_slices_up": 2,
                "growth_slices_down": 1,
                "growth_points_kept": 6,
                "growth_stop_reason_up": "cloud_extent",
                "growth_stop_reason_down": "cloud_extent",
            },
        ],
        cluster_points=np.empty((0, 3)),
        config=TrunkExtractionConfig(),
    )

    cleaning_result = StemCleaningResult(
        stem_mask=np.array([True, False, False, False, True, False, False, False]),
        n_points_before=4,
        n_points_after=2,
        n_points_removed=2,
        per_tree_stats=[
            {"tree_id": 0, "before": 2, "after": 1, "removed": 1, "pct_removed": 50.0},
            {"tree_id": 1, "before": 2, "after": 1, "removed": 1, "pct_removed": 50.0},
        ],
    )

    section_result = SectionResult(
        X_c=np.array([[0.00, 0.02, 0.04], [2.00, 2.08, 0.00]], dtype=np.float64),
        Y_c=np.array([[0.00, 0.01, 0.02], [0.00, 0.03, 0.00]], dtype=np.float64),
        R=np.array([[0.10, 0.11, 0.09], [0.16, 0.18, 0.00]], dtype=np.float64),
        check=np.zeros((2, 3), dtype=np.float64),
        sector_pct=np.full((2, 3), 100.0, dtype=np.float64),
        sections=np.array([1.0, 1.2, 1.4], dtype=np.float64),
        tree_ids=[0, 1],
        config=StemCleaningConfig(),
    )

    df = build_trunk_audit_table(
        xyz,
        trunk_result,
        cleaning_result,
        section_result,
        center_x=0.0,
        center_y=0.0,
    )

    assert list(df["tree_id"]) == [0, 1]
    assert list(df["assigned_points_total"]) == [4, 4]
    assert list(df["trunk_points"]) == [2, 2]
    assert list(df["clean_after"]) == [1, 1]
    assert list(df["valid_sections_total"]) == [3, 2]
    assert list(df["valid_sections_consecutive_max"]) == [3, 2]
    assert list(df["seed_selection_mode"]) == ["microband", "full_stripe_short"]
    assert list(df["growth_slices_down"]) == [4, 1]
    assert list(df["growth_points_kept"]) == [8, 6]
    assert np.isclose(df.loc[0, "mean_radius"], 0.10)
    assert np.isclose(df.loc[1, "seed_radius_ref"], 0.17)
    assert np.isclose(df.loc[0, "seed_vs_half_diam"], 1.0)
    assert np.isclose(df.loc[0, "seed_mean_dominant_fraction"], 0.95)
    assert np.isclose(df.loc[0, "growth_ratio"], 2.0)
    assert df.loc[1, "distance_to_center"] > df.loc[0, "distance_to_center"]
    assert "plo_seed_mode" in df.columns
    assert "plo_growth_voxel_count" in df.columns
    assert df.loc[0, "plo_seed_mode"] == ""

    called = {}

    def fake_to_csv(self, path, index=False):
        called["csv"] = (Path(path).suffix, index)

    def fake_to_excel(self, path, index=False, sheet_name=None):
        called["xlsx"] = (Path(path).suffix, index, sheet_name)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    csv_path = export_trunk_audit_table(df, Path("audit.csv"))
    xlsx_path = export_trunk_audit_table(df, Path("audit.xlsx"))

    assert csv_path.name == "audit.csv"
    assert xlsx_path.name == "audit.xlsx"
    assert called["csv"] == (".csv", False)
    assert called["xlsx"] == (".xlsx", False, "Trunk Audit")
