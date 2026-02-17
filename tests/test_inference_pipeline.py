"""Tests for classifier inference on LAS/LAZ files."""

from __future__ import annotations

from pathlib import Path
import sys

import laspy
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (  # noqa: E402
    apply_understory_classifier_to_las,
    load_training_data_from_dataframe,
    prepare_las_feature_matrix,
    save_classifier_bundle,
    train_and_evaluate_classifier,
)


def _make_training_dataframe(n: int = 1800, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    verticality = rng.uniform(0.0, 1.0, n)
    linearity = rng.uniform(0.0, 1.0, n)
    sphericity = rng.uniform(0.0, 1.0, n)
    dist_to_ground = rng.uniform(0.0, 14.0, n)

    score = 1.6 * verticality + 1.0 * linearity - 1.1 * sphericity + 0.07 * dist_to_ground
    labels = (score > 1.0).astype(np.int32)

    return pd.DataFrame(
        {
            "verticality": verticality,
            "linearity": linearity,
            "sphericity": sphericity,
            "dist_to_ground": dist_to_ground,
            "label": labels,
        }
    )


def _write_input_las(
    path: Path,
    n: int = 1200,
    with_features: bool = True,
    seed: int = 22,
) -> Path:
    rng = np.random.default_rng(seed)

    header = laspy.LasHeader(version="1.4", point_format=6)
    las = laspy.LasData(header)

    x = rng.uniform(0.0, 10.0, n)
    y = rng.uniform(0.0, 10.0, n)
    z = rng.uniform(0.0, 16.0, n)

    las.x = x
    las.y = y
    las.z = z

    if with_features:
        las.add_extra_dim(laspy.ExtraBytesParams(name="verticality", type="float32"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="linearity", type="float32"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="sphericity", type="float32"))

        # Deterministic feature generation for stable tests
        las.verticality = np.clip((z / 16.0) + rng.normal(0.0, 0.05, n), 0.0, 1.0).astype(np.float32)
        las.linearity = np.clip((x / 10.0) + rng.normal(0.0, 0.05, n), 0.0, 1.0).astype(np.float32)
        las.sphericity = np.clip((1.0 - y / 10.0) + rng.normal(0.0, 0.05, n), 0.0, 1.0).astype(np.float32)

    las.write(str(path))
    return path


def _train_model(model_path: Path) -> Path:
    df = _make_training_dataframe()
    train_data = load_training_data_from_dataframe(df)

    classifier, _ = train_and_evaluate_classifier(
        training_data=train_data,
        validation_data=None,
        test_data=None,
        n_estimators=80,
        max_depth=10,
        random_state=21,
        probability_threshold=0.5,
        verbose=False,
    )

    save_classifier_bundle(
        classifier=classifier,
        filepath=model_path,
        feature_names=train_data.feature_names,
        metadata={"threshold": 0.5},
    )
    return model_path


def test_prepare_las_feature_matrix_uses_z_for_dist_auto(tmp_path: Path):
    input_las = _write_input_las(tmp_path / "input_with_features.las", with_features=True)
    las_data = laspy.read(str(input_las))

    prep = prepare_las_feature_matrix(
        las_data=las_data,
        feature_names=["verticality", "linearity", "sphericity", "dist_to_ground"],
        dist_source="auto",
        backend="voxel",
        voxel_size=0.15,
        k_neighbors=16,
        verbose=False,
    )

    assert prep.feature_matrix.shape == (len(las_data.x), 4)
    assert len(prep.dist_to_ground) == len(las_data.x)


def test_apply_understory_classifier_to_las_writes_prediction_fields(tmp_path: Path):
    model_path = _train_model(tmp_path / "rf_model.pkl")
    input_las = _write_input_las(tmp_path / "inference_input.las", with_features=True)
    output_laz = tmp_path / "inference_output.laz"

    result = apply_understory_classifier_to_las(
        input_path=input_las,
        model_path=model_path,
        output_path=output_laz,
        slice_height=3.0,
        dist_source="auto",
        backend="voxel",
        voxel_size=0.12,
        k_neighbors=16,
        verbose=False,
    )

    assert output_laz.exists()
    assert result.n_points > 0
    assert result.n_tree + result.n_understory == result.n_points

    out = laspy.read(str(output_laz))
    dims = {name.lower() for name in out.point_format.dimension_names}

    assert "understory_prob" in dims
    assert "tree_prob" in dims
    assert "is_tree_ml" in dims

    mask = np.asarray(out.is_tree_ml)
    assert set(np.unique(mask)).issubset({0, 1})


def test_apply_understory_classifier_computes_missing_features(tmp_path: Path):
    model_path = _train_model(tmp_path / "rf_model_missing.pkl")
    input_las = _write_input_las(tmp_path / "inference_missing_features.las", with_features=False)
    output_laz = tmp_path / "inference_missing_features_out.laz"

    result = apply_understory_classifier_to_las(
        input_path=input_las,
        model_path=model_path,
        output_path=output_laz,
        slice_height=None,
        dist_source="relative",
        backend="voxel",
        voxel_size=0.2,
        k_neighbors=12,
        write_computed_features=True,
        verbose=False,
    )

    assert output_laz.exists()
    assert "verticality" in result.computed_features
    assert "linearity" in result.computed_features
    assert "sphericity" in result.computed_features

    out = laspy.read(str(output_laz))
    dims = {name.lower() for name in out.point_format.dimension_names}
    assert "verticality" in dims
    assert "linearity" in dims
    assert "sphericity" in dims
