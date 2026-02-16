"""Tests for classifier training/evaluation pipeline utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    load_classifier,
    load_training_data_from_dataframe,
    save_classifier_bundle,
    split_training_bank_by_plot,
    train_and_evaluate_classifier,
)


def _synthetic_bank(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    verticality = rng.uniform(0.0, 1.0, n)
    linearity = rng.uniform(0.0, 1.0, n)
    sphericity = rng.uniform(0.0, 1.0, n)
    dist_to_ground = rng.uniform(0.0, 14.0, n)

    score = (
        1.8 * verticality
        + 1.1 * linearity
        - 1.2 * sphericity
        + 0.08 * dist_to_ground
        + rng.normal(0.0, 0.12, n)
    )
    label = (score > 1.05).astype(np.int32)

    # Ensure both classes in synthetic generation
    if len(np.unique(label)) < 2:
        label[: n // 2] = 0
        label[n // 2 :] = 1

    plot_count = 8
    plot_ids = np.array([f"plot_{i:02d}" for i in range(plot_count)])
    assigned_plot = plot_ids[rng.integers(0, plot_count, size=n)]

    return pd.DataFrame(
        {
            "verticality": verticality,
            "linearity": linearity,
            "sphericity": sphericity,
            "dist_to_ground": dist_to_ground,
            "label": label,
            "plot_id": assigned_plot,
        }
    )


def test_load_training_data_from_dataframe_filters_invalid_labels():
    df = _synthetic_bank(n=500)
    invalid = pd.DataFrame(
        {
            "verticality": [0.5, 0.2],
            "linearity": [0.5, 0.2],
            "sphericity": [0.5, 0.2],
            "dist_to_ground": [2.0, 3.0],
            "label": [3, -1],
            "plot_id": ["plot_x", "plot_y"],
        }
    )
    df_invalid = pd.concat([df, invalid], axis=0, ignore_index=True)

    td = load_training_data_from_dataframe(df_invalid)
    assert len(td.labels) == len(df)
    assert set(np.unique(td.labels)) == {0, 1}

    with pytest.raises(ValueError):
        load_training_data_from_dataframe(df_invalid, drop_invalid_labels=False)


def test_train_and_evaluate_classifier_on_synthetic_bank():
    bank_df = _synthetic_bank(n=2600, seed=123)
    splits = split_training_bank_by_plot(
        bank_df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=123,
        group_col="plot_id",
    )

    train_data = load_training_data_from_dataframe(splits["train"])
    val_data = load_training_data_from_dataframe(splits["val"])
    test_data = load_training_data_from_dataframe(splits["test"])

    _, metrics = train_and_evaluate_classifier(
        training_data=train_data,
        validation_data=val_data,
        test_data=test_data,
        n_estimators=150,
        max_depth=10,
        random_state=123,
        probability_threshold=0.5,
        verbose=False,
    )

    assert set(metrics.keys()) == {"train", "val", "test"}
    assert metrics["test"].f1_tree > 0.80
    assert metrics["test"].f1_understory > 0.80


def test_save_and_load_classifier_bundle_roundtrip(tmp_path: Path):
    bank_df = _synthetic_bank(n=1200, seed=7)
    train_data = load_training_data_from_dataframe(bank_df)

    classifier, _ = train_and_evaluate_classifier(
        training_data=train_data,
        validation_data=None,
        test_data=None,
        n_estimators=80,
        max_depth=8,
        random_state=7,
        verbose=False,
    )

    out_model = tmp_path / "rf_model.pkl"
    metadata = {"phase": "phase3", "threshold": 0.5}
    save_classifier_bundle(
        classifier=classifier,
        filepath=out_model,
        feature_names=train_data.feature_names,
        metadata=metadata,
    )

    loaded_classifier, feature_names, loaded_metadata = load_classifier(
        out_model,
        return_metadata=True,
    )

    sample = train_data.features[:50]
    original_proba = classifier.predict_proba(sample)
    loaded_proba = loaded_classifier.predict_proba(sample)

    assert feature_names == train_data.feature_names
    assert loaded_metadata["phase"] == "phase3"
    assert np.allclose(original_proba, loaded_proba)
