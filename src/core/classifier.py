"""
Machine learning utilities for understory separation.
Uses Random Forest trained on geometric features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClassifierResult:
    """Result of ML classification."""

    is_tree: np.ndarray
    probabilities: np.ndarray
    n_tree: int
    n_understory: int


@dataclass
class TrainingData:
    """Training data for classifier."""

    features: np.ndarray
    labels: np.ndarray
    feature_names: list


@dataclass
class BinaryClassificationMetrics:
    """Binary classification metrics summary."""

    accuracy: float
    precision_tree: float
    recall_tree: float
    f1_tree: float
    precision_understory: float
    recall_understory: float
    f1_understory: float
    balanced_accuracy: float
    support_tree: int
    support_understory: int
    confusion_matrix: np.ndarray
    roc_auc_tree: Optional[float] = None


@dataclass
class SliceClassifierResult:
    """Result of slice-based ML classification."""

    is_tree: np.ndarray
    probabilities: np.ndarray
    n_tree: int
    n_understory: int
    n_protected: int
    n_ml_classified: int


def prepare_features(
    verticality: np.ndarray,
    linearity: np.ndarray,
    sphericity: np.ndarray,
    dist_to_ground: np.ndarray,
    planarity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, list]:
    """Prepare feature matrix for classification."""
    feature_list = [verticality, linearity, sphericity, dist_to_ground]
    feature_names = ["verticality", "linearity", "sphericity", "dist_to_ground"]

    if planarity is not None:
        feature_list.append(planarity)
        feature_names.append("planarity")

    features = np.column_stack(feature_list)
    return features, feature_names


def train_classifier(
    training_data: TrainingData,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    verbose: bool = False,
):
    """Train Random Forest classifier on labeled data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    if verbose:
        n_tree = int(np.sum(training_data.labels == 1))
        n_understory = int(np.sum(training_data.labels == 0))
        print("Training Random Forest classifier...")
        print(f"  Samples: {len(training_data.labels):,}")
        print(f"  Features: {len(training_data.feature_names)}")
        print(f"  Class balance: tree={n_tree:,}, understory={n_understory:,}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    if verbose:
        scores = cross_val_score(clf, training_data.features, training_data.labels, cv=5)
        print(f"  Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    clf.fit(training_data.features, training_data.labels)

    if verbose:
        print("  Training complete")
        print("  Feature importances:")
        for name, importance in sorted(
            zip(training_data.feature_names, clf.feature_importances_),
            key=lambda item: -item[1],
        ):
            print(f"    {name}: {importance:.3f}")

    return clf


def classify_understory_ml(
    features: np.ndarray,
    classifier,
    probability_threshold: float = 0.5,
    verbose: bool = False,
) -> ClassifierResult:
    """Classify points using trained ML model."""
    if verbose:
        print(f"Classifying {len(features):,} points...")

    probabilities = classifier.predict_proba(features)[:, 1]
    is_tree = probabilities >= probability_threshold

    n_tree = int(np.sum(is_tree))
    n_understory = int(len(is_tree) - n_tree)

    if verbose:
        print(f"  Tree: {n_tree:,} ({100 * n_tree / len(is_tree):.1f}%)")
        print(f"  Understory: {n_understory:,} ({100 * n_understory / len(is_tree):.1f}%)")

    return ClassifierResult(
        is_tree=is_tree,
        probabilities=probabilities,
        n_tree=n_tree,
        n_understory=n_understory,
    )


def save_classifier(classifier, filepath: str | Path, feature_names: list):
    """Backward-compatible save helper."""
    return save_classifier_bundle(
        classifier=classifier,
        filepath=filepath,
        feature_names=feature_names,
        metadata=None,
    )


def save_classifier_bundle(
    classifier,
    filepath: str | Path,
    feature_names: list,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save trained classifier bundle to file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "classifier": classifier,
        "feature_names": feature_names,
        "metadata": metadata or {},
    }
    with open(out_path, "wb") as file_obj:
        pickle.dump(data, file_obj)

    print(f"Classifier saved to {out_path}")
    return out_path


def load_classifier(filepath: str | Path, return_metadata: bool = False):
    """Load trained classifier from file."""
    with open(filepath, "rb") as file_obj:
        data = pickle.load(file_obj)

    classifier = data["classifier"]
    feature_names = data["feature_names"]
    metadata = data.get("metadata", {})

    if return_metadata:
        return classifier, feature_names, metadata

    return classifier, feature_names


def load_training_data_from_laz(
    filepath: str,
    label_field: str = "training_label",
    verbose: bool = False,
) -> TrainingData:
    """Load training data from LAS/LAZ file with labels."""
    import laspy

    las = laspy.read(filepath)

    dist_to_ground = np.asarray(las.z, dtype=np.float32)
    verticality = np.asarray(las.verticality, dtype=np.float32)
    linearity = np.asarray(las.linearity, dtype=np.float32)
    sphericity = np.asarray(las.sphericity, dtype=np.float32)

    labels = np.asarray(getattr(las, label_field), dtype=np.int32)

    features, feature_names = prepare_features(
        verticality=verticality,
        linearity=linearity,
        sphericity=sphericity,
        dist_to_ground=dist_to_ground,
    )

    valid_mask = (labels == 0) | (labels == 1)
    features = features[valid_mask]
    labels = labels[valid_mask]

    if verbose:
        print(f"Loaded training data from {filepath}")
        print(f"  Points (valid labels): {len(labels):,}")
        print(f"  Tree (1): {int(np.sum(labels == 1)):,}")
        print(f"  Understory (0): {int(np.sum(labels == 0)):,}")

    if len(np.unique(labels)) < 2:
        raise ValueError("Training data must contain both classes (0 and 1)")

    return TrainingData(features=features, labels=labels, feature_names=feature_names)


def load_training_data_from_dataframe(
    dataframe,
    feature_names: Sequence[str] = ("verticality", "linearity", "sphericity", "dist_to_ground"),
    label_col: str = "label",
    drop_invalid_labels: bool = True,
) -> TrainingData:
    """Build TrainingData from a DataFrame."""
    missing_cols = [name for name in (*feature_names, label_col) if name not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    labels = np.asarray(dataframe[label_col], dtype=np.int32)
    valid_mask = (labels == 0) | (labels == 1)

    if not np.all(valid_mask):
        if drop_invalid_labels:
            dataframe = dataframe.loc[valid_mask]
            labels = labels[valid_mask]
        else:
            raise ValueError("Label column contains values outside {0, 1}")

    features = np.asarray(dataframe.loc[:, feature_names], dtype=np.float32)

    if len(features) == 0:
        raise ValueError("No valid rows available for training")

    if len(np.unique(labels)) < 2:
        raise ValueError("Training data must contain both classes (0 and 1)")

    return TrainingData(
        features=features,
        labels=labels.astype(np.int32),
        feature_names=list(feature_names),
    )


def evaluate_classifier(
    classifier,
    evaluation_data: TrainingData,
    probability_threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    """Evaluate a trained classifier on labeled data."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    labels = evaluation_data.labels.astype(np.int32)
    probabilities = classifier.predict_proba(evaluation_data.features)[:, 1]
    predictions = (probabilities >= probability_threshold).astype(np.int32)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    cm = confusion_matrix(labels, predictions, labels=[0, 1]).astype(np.int32)

    roc_auc_tree = None
    if np.unique(labels).size >= 2:
        roc_auc_tree = float(roc_auc_score(labels, probabilities))

    return BinaryClassificationMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision_tree=float(precision[1]),
        recall_tree=float(recall[1]),
        f1_tree=float(f1[1]),
        precision_understory=float(precision[0]),
        recall_understory=float(recall[0]),
        f1_understory=float(f1[0]),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        support_tree=int(support[1]),
        support_understory=int(support[0]),
        confusion_matrix=cm,
        roc_auc_tree=roc_auc_tree,
    )


def train_and_evaluate_classifier(
    training_data: TrainingData,
    validation_data: Optional[TrainingData] = None,
    test_data: Optional[TrainingData] = None,
    probability_threshold: float = 0.5,
    n_estimators: int = 200,
    max_depth: int = 12,
    random_state: int = 42,
    verbose: bool = False,
):
    """Train classifier and evaluate available splits."""
    classifier = train_classifier(
        training_data=training_data,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose,
    )

    metrics: Dict[str, BinaryClassificationMetrics] = {
        "train": evaluate_classifier(
            classifier=classifier,
            evaluation_data=training_data,
            probability_threshold=probability_threshold,
        )
    }

    if validation_data is not None:
        metrics["val"] = evaluate_classifier(
            classifier=classifier,
            evaluation_data=validation_data,
            probability_threshold=probability_threshold,
        )

    if test_data is not None:
        metrics["test"] = evaluate_classifier(
            classifier=classifier,
            evaluation_data=test_data,
            probability_threshold=probability_threshold,
        )

    return classifier, metrics


def slice_classify_understory(
    features: np.ndarray,
    dist_to_ground: np.ndarray,
    classifier,
    slice_height: float = 3.5,
    probability_threshold: float = 0.5,
    verbose: bool = False,
) -> SliceClassifierResult:
    """Classify understory in lower slice while protecting upper slice."""
    n_points = len(features)

    lower_mask = dist_to_ground < slice_height
    upper_mask = ~lower_mask

    n_lower = int(np.sum(lower_mask))
    n_upper = int(np.sum(upper_mask))

    if verbose:
        print(f"Slice-based ML classification (threshold: {slice_height}m)...")
        print(f"  Upper zone (protected): {n_upper:,} points")
        print(f"  Lower zone (ML): {n_lower:,} points")

    is_tree = np.ones(n_points, dtype=bool)
    probabilities = np.ones(n_points, dtype=np.float32)

    if n_lower > 0:
        lower_features = features[lower_mask]
        tree_proba = classifier.predict_proba(lower_features)[:, 1]
        is_tree_lower = tree_proba >= probability_threshold

        is_tree[lower_mask] = is_tree_lower
        probabilities[lower_mask] = tree_proba

        if verbose:
            n_understory_lower = int(np.sum(~is_tree_lower))
            print(f"  ML removed {n_understory_lower:,} understory points from lower zone")

    n_tree = int(np.sum(is_tree))
    n_understory = int(n_points - n_tree)

    if verbose:
        print(f"  Final: Tree={n_tree:,}, Understory={n_understory:,}")

    return SliceClassifierResult(
        is_tree=is_tree,
        probabilities=probabilities,
        n_tree=n_tree,
        n_understory=n_understory,
        n_protected=n_upper,
        n_ml_classified=n_lower,
    )
