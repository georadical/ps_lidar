"""Tests for training data bank utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.training_bank import (
    inspect_training_file,
    build_training_bank,
    split_training_bank_by_plot,
)


def _create_labeled_las(path: Path, n: int = 1000, seed: int = 42, include_labels: bool = True) -> Path:
    import laspy

    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(version="1.4", point_format=6)
    las = laspy.LasData(header)

    las.x = rng.uniform(0, 10, n)
    las.y = rng.uniform(0, 10, n)
    las.z = rng.uniform(0, 20, n)

    las.add_extra_dim(laspy.ExtraBytesParams(name="verticality", type="float32"))
    las.add_extra_dim(laspy.ExtraBytesParams(name="linearity", type="float32"))
    las.add_extra_dim(laspy.ExtraBytesParams(name="sphericity", type="float32"))

    las.verticality = rng.uniform(0, 1, n).astype(np.float32)
    las.linearity = rng.uniform(0, 1, n).astype(np.float32)
    las.sphericity = rng.uniform(0, 1, n).astype(np.float32)

    if include_labels:
        labels = np.zeros(n, dtype=np.uint8)
        labels[: n // 2] = 1
        rng.shuffle(labels)
        las.add_extra_dim(laspy.ExtraBytesParams(name="training_label", type="uint8"))
        las.training_label = labels

    las.write(str(path))
    return path


def test_inspect_training_file_usable(tmp_path: Path):
    f = _create_labeled_las(tmp_path / "plot_a.las", n=500)
    report = inspect_training_file(f)

    assert report.is_usable is True
    assert report.has_label_field is True
    assert report.reason == "ok"
    assert report.valid_point_count == 500


def test_build_training_bank_skips_invalid(tmp_path: Path):
    valid = _create_labeled_las(tmp_path / "plot_valid.las", n=400, include_labels=True)
    invalid = _create_labeled_las(tmp_path / "plot_invalid.las", n=300, include_labels=False)

    bank_df, result = build_training_bank(inputs=[str(valid), str(invalid)])

    assert result.n_files_scanned == 2
    assert result.n_files_usable == 1
    assert len(bank_df) == 400
    assert set(bank_df["label"].unique()) == {0, 1}
    assert "plot_id" in bank_df.columns


def test_split_training_bank_by_plot_no_leakage(tmp_path: Path):
    f1 = _create_labeled_las(tmp_path / "plot_01.las", n=200)
    f2 = _create_labeled_las(tmp_path / "plot_02.las", n=200)
    f3 = _create_labeled_las(tmp_path / "plot_03.las", n=200)

    bank_df, _ = build_training_bank(inputs=[str(f1), str(f2), str(f3)])
    splits = split_training_bank_by_plot(bank_df, train_ratio=0.67, val_ratio=0.0, test_ratio=0.33, seed=123)

    train_plots = set(splits["train"]["plot_id"].unique())
    test_plots = set(splits["test"]["plot_id"].unique())

    assert len(train_plots) > 0
    assert len(test_plots) > 0
    assert train_plots.isdisjoint(test_plots)
