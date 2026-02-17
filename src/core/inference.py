"""Inference utilities for applying trained understory classifiers to LAS/LAZ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import warnings

import laspy
import numpy as np

from .classifier import (
    classify_understory_ml,
    load_classifier,
    slice_classify_understory,
)
from .features import compute_all_features_fast, compute_relative_features_fast


SUPPORTED_GEOMETRIC_FEATURES = {"verticality", "linearity", "sphericity", "planarity"}


@dataclass
class UnderstoryInferenceResult:
    """Summary of classifier inference on one point cloud."""

    input_path: str
    output_path: str
    n_points: int
    n_tree: int
    n_understory: int
    probability_threshold: float
    slice_height: Optional[float]
    feature_names: Sequence[str]
    computed_features: Sequence[str]


@dataclass
class FeaturePreparationResult:
    """Prepared feature matrix plus supporting arrays."""

    feature_matrix: np.ndarray
    dist_to_ground: np.ndarray
    computed_feature_names: Sequence[str]


def _dimension_name_map(las_data: laspy.LasData) -> Dict[str, str]:
    return {name.lower(): name for name in las_data.point_format.dimension_names}


def _upsert_extra_dimension(las_data: laspy.LasData, name: str, data_type: str) -> str:
    dim_map = _dimension_name_map(las_data)
    existing = dim_map.get(name.lower())
    if existing is not None:
        return existing

    las_data.add_extra_dim(laspy.ExtraBytesParams(name=name, type=data_type))
    return name


def prepare_las_feature_matrix(
    las_data: laspy.LasData,
    feature_names: Sequence[str],
    dist_source: str = "auto",
    backend: str = "voxel",
    voxel_size: float = 0.1,
    k_neighbors: int = 20,
    pgeof_scale: float = 0.15,
    pgeof_max_knn: int = 50000,
    verbose: bool = False,
) -> FeaturePreparationResult:
    """Build feature matrix for classifier inference from LAS/LAZ data."""
    if dist_source not in {"auto", "z", "relative"}:
        raise ValueError(f"Unsupported dist_source '{dist_source}'. Use 'auto', 'z', or 'relative'.")

    feature_names = [str(name) for name in feature_names]
    dim_map = _dimension_name_map(las_data)

    xyz = np.column_stack([las_data.x, las_data.y, las_data.z]).astype(np.float64)

    required_geometric = {
        name for name in feature_names if name in SUPPORTED_GEOMETRIC_FEATURES and name not in dim_map
    }
    needs_dist = "dist_to_ground" in feature_names and "dist_to_ground" not in dim_map

    geometric = None
    dist_to_ground_computed = None

    should_compute_all = bool(required_geometric)
    if not should_compute_all and needs_dist and dist_source == "relative":
        should_compute_all = False

    if should_compute_all:
        if verbose:
            print(
                "Computing missing geometric features for inference: "
                f"{sorted(required_geometric)}"
            )
        geometric, dist_to_ground_computed, _ = compute_all_features_fast(
            xyz,
            voxel_size=voxel_size,
            k_neighbors=k_neighbors,
            backend=backend,
            pgeof_scale=pgeof_scale,
            pgeof_max_knn=pgeof_max_knn,
            verbose=verbose,
        )

    if needs_dist and dist_source == "relative" and dist_to_ground_computed is None:
        if verbose:
            print("Computing relative dist_to_ground for inference...")
        dist_to_ground_computed, _ = compute_relative_features_fast(
            xyz,
            voxel_size=voxel_size,
            verbose=verbose,
        )

    matrix_columns = []
    computed_feature_names = []

    for feature_name in feature_names:
        lower_name = feature_name.lower()

        if lower_name in dim_map:
            column = np.asarray(getattr(las_data, dim_map[lower_name]), dtype=np.float32)
            matrix_columns.append(column)
            continue

        if lower_name == "dist_to_ground":
            if dist_source == "z":
                column = np.asarray(las_data.z, dtype=np.float32)
            elif dist_to_ground_computed is not None:
                column = np.asarray(dist_to_ground_computed, dtype=np.float32)
                computed_feature_names.append("dist_to_ground")
            else:
                # auto fallback for normalized clouds
                column = np.asarray(las_data.z, dtype=np.float32)
            matrix_columns.append(column)
            continue

        if lower_name in SUPPORTED_GEOMETRIC_FEATURES and geometric is not None:
            column = np.asarray(getattr(geometric, lower_name), dtype=np.float32)
            matrix_columns.append(column)
            computed_feature_names.append(lower_name)
            continue

        raise ValueError(
            f"Feature '{feature_name}' not found in LAS dimensions and cannot be computed automatically."
        )

    feature_matrix = np.column_stack(matrix_columns).astype(np.float32)

    # Dist to ground needed by slice mode regardless of model feature list.
    if "dist_to_ground" in dim_map:
        dist_to_ground = np.asarray(getattr(las_data, dim_map["dist_to_ground"]), dtype=np.float32)
    elif dist_to_ground_computed is not None:
        dist_to_ground = np.asarray(dist_to_ground_computed, dtype=np.float32)
    else:
        dist_to_ground = np.asarray(las_data.z, dtype=np.float32)

    return FeaturePreparationResult(
        feature_matrix=feature_matrix,
        dist_to_ground=dist_to_ground,
        computed_feature_names=sorted(set(computed_feature_names)),
    )


def apply_understory_classifier_to_las(
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    feature_names: Optional[Sequence[str]] = None,
    probability_threshold: Optional[float] = None,
    slice_height: Optional[float] = 3.5,
    dist_source: str = "auto",
    backend: str = "voxel",
    voxel_size: float = 0.1,
    k_neighbors: int = 20,
    pgeof_scale: float = 0.15,
    pgeof_max_knn: int = 50000,
    understory_prob_field: str = "understory_prob",
    tree_prob_field: str = "tree_prob",
    mask_field: str = "is_tree_ml",
    write_computed_features: bool = False,
    update_classification: bool = False,
    classification_tree: int = 5,
    classification_understory: int = 3,
    compress: bool = True,
    verbose: bool = False,
) -> UnderstoryInferenceResult:
    """Apply a trained classifier to LAS/LAZ and export predictions."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"input_path must be .las/.laz, got '{input_path.suffix}'")
    if output_path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"output_path must be .las/.laz, got '{output_path.suffix}'")

    classifier, model_feature_names, metadata = load_classifier(model_path, return_metadata=True)

    selected_features = list(feature_names) if feature_names else list(model_feature_names)
    if not selected_features:
        raise ValueError("No feature names available. Provide --features or save model with feature_names.")

    threshold = probability_threshold
    if threshold is None:
        threshold = float(metadata.get("threshold", 0.5))

    las_data = laspy.read(str(input_path))

    prep = prepare_las_feature_matrix(
        las_data=las_data,
        feature_names=selected_features,
        dist_source=dist_source,
        backend=backend,
        voxel_size=voxel_size,
        k_neighbors=k_neighbors,
        pgeof_scale=pgeof_scale,
        pgeof_max_knn=pgeof_max_knn,
        verbose=verbose,
    )

    if slice_height is None:
        cls_result = classify_understory_ml(
            features=prep.feature_matrix,
            classifier=classifier,
            probability_threshold=threshold,
            verbose=verbose,
        )
        is_tree = cls_result.is_tree
        tree_probabilities = cls_result.probabilities
        n_tree = cls_result.n_tree
        n_understory = cls_result.n_understory
    else:
        cls_result = slice_classify_understory(
            features=prep.feature_matrix,
            dist_to_ground=prep.dist_to_ground,
            classifier=classifier,
            slice_height=slice_height,
            probability_threshold=threshold,
            verbose=verbose,
        )
        is_tree = cls_result.is_tree
        tree_probabilities = cls_result.probabilities
        n_tree = cls_result.n_tree
        n_understory = cls_result.n_understory

    understory_probabilities = 1.0 - np.asarray(tree_probabilities, dtype=np.float32)
    is_tree_u8 = np.asarray(is_tree, dtype=np.uint8)

    prob_dim = _upsert_extra_dimension(las_data, understory_prob_field, "float32")
    setattr(las_data, prob_dim, understory_probabilities)

    tree_prob_dim = _upsert_extra_dimension(las_data, tree_prob_field, "float32")
    setattr(las_data, tree_prob_dim, np.asarray(tree_probabilities, dtype=np.float32))

    mask_dim = _upsert_extra_dimension(las_data, mask_field, "uint8")
    setattr(las_data, mask_dim, is_tree_u8)

    if write_computed_features:
        for feature_name in prep.computed_feature_names:
            if feature_name in _dimension_name_map(las_data):
                continue

            if feature_name == "dist_to_ground":
                dim_name = _upsert_extra_dimension(las_data, "dist_to_ground", "float32")
                setattr(las_data, dim_name, np.asarray(prep.dist_to_ground, dtype=np.float32))
            elif feature_name in SUPPORTED_GEOMETRIC_FEATURES:
                # Recompute feature matrix column index by model order.
                idx = selected_features.index(feature_name)
                dim_name = _upsert_extra_dimension(las_data, feature_name, "float32")
                setattr(las_data, dim_name, np.asarray(prep.feature_matrix[:, idx], dtype=np.float32))

    if update_classification:
        cls_values = np.where(is_tree, classification_tree, classification_understory).astype(np.uint8)
        las_data.classification = cls_values

    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_laz = output_path.suffix.lower() == ".laz"
    if compress and not is_laz:
        warnings.warn(
            "compress=True ignored for .las output. Use .laz for compressed output.",
            UserWarning,
            stacklevel=2,
        )

    las_data.write(str(output_path), do_compress=(compress if is_laz else False))

    return UnderstoryInferenceResult(
        input_path=str(input_path),
        output_path=str(output_path),
        n_points=int(len(is_tree)),
        n_tree=int(n_tree),
        n_understory=int(n_understory),
        probability_threshold=float(threshold),
        slice_height=slice_height,
        feature_names=selected_features,
        computed_features=prep.computed_feature_names,
    )
