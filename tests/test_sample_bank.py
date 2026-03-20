from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np
import pandas as pd

from src.core.sample_bank import (
    build_sample_bank,
    reports_to_dataframe,
    save_sample_splits,
    split_sample_bank_by_plot,
)


def _make_las(xyz: np.ndarray, labels: np.ndarray | None = None) -> laspy.LasData:
    header = laspy.LasHeader(point_format=3, version="1.2")
    if labels is not None:
        header.add_extra_dim(laspy.ExtraBytesParams(name="training_label", type=np.int16))

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    if labels is not None:
        las["training_label"] = labels.astype(np.int16)
    return las


def test_build_sample_bank_keeps_clear_voxels_and_discards_mixed() -> None:
    stem_points = np.array([
        [0.01, 0.01, 1.00],
        [0.02, 0.01, 1.01],
        [0.01, 0.02, 1.02],
        [0.02, 0.02, 1.03],
        [0.03, 0.02, 1.04],
    ])
    no_stem_points = np.array([
        [0.21, 0.21, 1.00],
        [0.22, 0.21, 1.01],
        [0.21, 0.22, 1.02],
        [0.22, 0.22, 1.03],
        [0.23, 0.22, 1.04],
    ])
    mixed_points = np.array([
        [0.41, 0.41, 1.00],
        [0.42, 0.41, 1.01],
        [0.41, 0.42, 1.02],
        [0.42, 0.42, 1.03],
        [0.43, 0.42, 1.04],
    ])

    xyz = np.vstack([stem_points, no_stem_points, mixed_points])
    labels = np.array([1] * 5 + [0] * 5 + [1, 1, 1, 0, 0], dtype=np.int16)

    las = _make_las(xyz, labels)
    fake_path = Path("plot_a.las")
    with patch("src.core.sample_bank._expand_inputs", return_value=[fake_path]), patch("laspy.read", return_value=las):
        bank_df, result = build_sample_bank(
            inputs=[str(fake_path)],
            voxel_size=0.05,
        )

        assert result.n_files_scanned == 1
        assert result.n_files_usable == 1
        assert len(bank_df) == 2
        assert set(bank_df["label"].tolist()) == {0, 1}
        assert set(bank_df["label_name"].tolist()) == {"stem", "no_stem"}
        assert set(bank_df["plot_id"].tolist()) == {"plot_a"}
        assert set(bank_df.columns) >= {
            "plot_id",
            "source_file",
            "voxel_key",
            "x",
            "y",
            "z",
            "n_points",
            "dominant_fraction",
            "label",
            "label_name",
            "verticality",
            "linearity",
            "planarity",
            "sphericity",
            "anisotropy",
            "surface_variation",
            "roughness",
            "neighbor_count",
            "volume_density",
        }
        assert np.all(bank_df["dominant_fraction"] >= 0.8)


def test_build_sample_bank_reports_missing_label_field() -> None:
    xyz = np.array([
        [0.00, 0.00, 0.00],
        [0.10, 0.10, 0.10],
        [0.20, 0.20, 0.20],
    ])
    las = _make_las(xyz, labels=None)
    fake_path = Path("plot_missing.las")
    with patch("src.core.sample_bank._expand_inputs", return_value=[fake_path]), patch("laspy.read", return_value=las):
        bank_df, result = build_sample_bank(inputs=[str(fake_path)])
        assert bank_df.empty
        assert result.n_files_scanned == 1
        assert result.n_files_usable == 0
        assert result.reports[0].status == "missing_label_field"

    reports_df = reports_to_dataframe(result.reports)
    assert list(reports_df["status"]) == ["missing_label_field"]


def test_split_sample_bank_by_plot_and_save_csv() -> None:
    bank_df = pd.DataFrame(
        {
            "plot_id": ["p1", "p1", "p2", "p2", "p3", "p4"],
            "source_file": ["a", "a", "b", "b", "c", "d"],
            "voxel_key": ["1", "2", "3", "4", "5", "6"],
            "x": [0, 1, 2, 3, 4, 5],
            "y": [0, 1, 2, 3, 4, 5],
            "z": [0, 1, 2, 3, 4, 5],
            "n_points": [5, 6, 7, 8, 9, 10],
            "dominant_fraction": [1.0] * 6,
            "label": [1, 1, 0, 0, 1, 0],
            "label_name": ["stem", "stem", "no_stem", "no_stem", "stem", "no_stem"],
            "verticality": [0.9] * 6,
            "linearity": [0.8] * 6,
            "planarity": [0.1] * 6,
            "sphericity": [0.1] * 6,
            "anisotropy": [0.9] * 6,
            "surface_variation": [0.05] * 6,
            "roughness": [0.01] * 6,
            "neighbor_count": [5] * 6,
            "volume_density": [10.0] * 6,
        }
    )

    splits = split_sample_bank_by_plot(bank_df, seed=7)
    train_groups = set(splits["train"]["plot_id"].unique())
    val_groups = set(splits["val"]["plot_id"].unique())
    test_groups = set(splits["test"]["plot_id"].unique())

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    saved: list[Path] = []

    def _fake_save(df, output_path):
        saved.append(Path(output_path))
        return Path(output_path)

    with patch("src.core.sample_bank._save_dataframe", side_effect=_fake_save), patch("pathlib.Path.mkdir"):
        paths = save_sample_splits(splits, Path("virtual_splits"), file_format="csv")

    assert paths["train"] == Path("virtual_splits/train.csv")
    assert paths["val"] == Path("virtual_splits/val.csv")
    assert paths["test"] == Path("virtual_splits/test.csv")
    assert saved == [
        Path("virtual_splits/train.csv"),
        Path("virtual_splits/val.csv"),
        Path("virtual_splits/test.csv"),
    ]
