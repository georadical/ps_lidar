"""
Training data bank utilities for understory/tree ML labeling workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd


DEFAULT_FEATURE_FIELDS: Tuple[str, ...] = (
    "verticality",
    "linearity",
    "sphericity",
)


@dataclass
class TrainingFileReport:
    path: str
    point_count: int
    available_dimensions: List[str]
    label_field: str
    has_label_field: bool
    missing_feature_fields: List[str]
    class_counts: Dict[str, int]
    valid_point_count: int
    is_usable: bool
    reason: str


@dataclass
class TrainingBankBuildResult:
    n_files_scanned: int
    n_files_usable: int
    n_points_total: int
    n_points_kept: int
    reports: List[TrainingFileReport]
    output_path: Optional[str] = None


def _read_las(path: Path):
    import laspy

    return laspy.read(str(path))


def inspect_training_file(
    path: str | Path,
    label_field: str = "training_label",
    feature_fields: Sequence[str] = DEFAULT_FEATURE_FIELDS,
) -> TrainingFileReport:
    """Inspect a LAZ/LAS file for training-bank suitability."""
    p = Path(path)
    if not p.exists():
        return TrainingFileReport(
            path=str(p),
            point_count=0,
            available_dimensions=[],
            label_field=label_field,
            has_label_field=False,
            missing_feature_fields=list(feature_fields),
            class_counts={},
            valid_point_count=0,
            is_usable=False,
            reason="file_not_found",
        )

    las = _read_las(p)
    dims = [d.lower() for d in las.point_format.dimension_names]

    has_label = label_field.lower() in dims
    missing_fields = [f for f in feature_fields if f.lower() not in dims]

    class_counts: Dict[str, int] = {}
    valid_point_count = 0
    if has_label:
        labels = np.array(getattr(las, label_field), dtype=np.int32)
        labels_valid = labels[(labels == 0) | (labels == 1)]
        unique, counts = np.unique(labels_valid, return_counts=True)
        class_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
        valid_point_count = int(len(labels_valid))

    is_usable = has_label and (len(missing_fields) == 0) and (len(class_counts) >= 2)
    if not has_label:
        reason = "missing_label_field"
    elif len(missing_fields) > 0:
        reason = "missing_feature_fields"
    elif len(class_counts) < 2:
        reason = "insufficient_label_classes"
    else:
        reason = "ok"

    return TrainingFileReport(
        path=str(p),
        point_count=int(len(las.x)),
        available_dimensions=dims,
        label_field=label_field,
        has_label_field=has_label,
        missing_feature_fields=missing_fields,
        class_counts=class_counts,
        valid_point_count=valid_point_count,
        is_usable=is_usable,
        reason=reason,
    )


def _resolve_lidar_files(inputs: Sequence[str]) -> List[Path]:
    """Resolve files from explicit paths, directories, or glob patterns."""
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.exists() and p.is_file() and p.suffix.lower() in {".las", ".laz"}:
            paths.append(p)
            continue

        if p.exists() and p.is_dir():
            paths.extend(sorted(p.rglob("*.las")))
            paths.extend(sorted(p.rglob("*.laz")))
            continue

        # Treat as glob pattern
        paths.extend(sorted(Path(".").glob(item)))

    # Deduplicate while preserving order
    seen = set()
    unique_paths: List[Path] = []
    for p in paths:
        sp = str(p.resolve())
        if sp not in seen:
            seen.add(sp)
            unique_paths.append(p)
    return unique_paths


def build_training_bank(
    inputs: Sequence[str],
    output_path: Optional[str | Path] = None,
    label_field: str = "training_label",
    feature_fields: Sequence[str] = DEFAULT_FEATURE_FIELDS,
    include_xyz: bool = True,
    include_source_meta: bool = True,
) -> Tuple[pd.DataFrame, TrainingBankBuildResult]:
    """Build a unified training bank table from multiple labeled LAS/LAZ files."""
    files = _resolve_lidar_files(inputs)
    reports: List[TrainingFileReport] = []
    frames: List[pd.DataFrame] = []

    n_points_total = 0
    n_points_kept = 0

    for fp in files:
        report = inspect_training_file(fp, label_field=label_field, feature_fields=feature_fields)
        reports.append(report)
        n_points_total += report.point_count

        if not report.is_usable:
            continue

        las = _read_las(fp)
        labels = np.array(getattr(las, label_field), dtype=np.int32)
        valid_mask = (labels == 0) | (labels == 1)

        data: Dict[str, np.ndarray] = {
            "label": labels[valid_mask].astype(np.int8),
        }

        if include_xyz:
            data["x"] = np.array(las.x, dtype=np.float64)[valid_mask]
            data["y"] = np.array(las.y, dtype=np.float64)[valid_mask]
            data["z"] = np.array(las.z, dtype=np.float64)[valid_mask]
            data["dist_to_ground"] = data["z"].astype(np.float32)

        for f in feature_fields:
            data[f] = np.array(getattr(las, f), dtype=np.float32)[valid_mask]

        if include_source_meta:
            plot_id = fp.stem
            n_valid = int(np.sum(valid_mask))
            data["plot_id"] = np.array([plot_id] * n_valid)
            data["source_file"] = np.array([str(fp)] * n_valid)

        df = pd.DataFrame(data)
        frames.append(df)
        n_points_kept += len(df)

    if frames:
        bank_df = pd.concat(frames, axis=0, ignore_index=True)
    else:
        cols = ["label", *feature_fields]
        if include_xyz:
            cols = ["label", "x", "y", "z", "dist_to_ground", *feature_fields]
        if include_source_meta:
            cols += ["plot_id", "source_file"]
        bank_df = pd.DataFrame(columns=cols)

    result = TrainingBankBuildResult(
        n_files_scanned=len(files),
        n_files_usable=sum(1 for r in reports if r.is_usable),
        n_points_total=n_points_total,
        n_points_kept=n_points_kept,
        reports=reports,
        output_path=None,
    )

    if output_path is not None:
        saved_path = save_training_bank(bank_df, output_path)
        result.output_path = str(saved_path)

    return bank_df, result


def save_training_bank(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save training bank to parquet/csv based on extension.

    If parquet engine is unavailable, falls back to CSV with same stem.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    suffix = out.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(out, index=False)
            return out
        except ImportError:
            fallback = out.with_suffix(".csv")
            warnings.warn(
                "Parquet engine not available. Falling back to CSV output.",
                UserWarning,
                stacklevel=2,
            )
            df.to_csv(fallback, index=False)
            return fallback
    if suffix == ".csv":
        df.to_csv(out, index=False)
        return out

    raise ValueError(f"Unsupported output extension '{suffix}'. Use .parquet or .csv")


def split_training_bank_by_plot(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_col: str = "plot_id",
) -> Dict[str, pd.DataFrame]:
    """Split training bank by plot/group to avoid leakage across sets."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame")

    unique_groups = df[group_col].dropna().unique()
    if len(unique_groups) == 0:
        return {"train": df.iloc[0:0].copy(), "val": df.iloc[0:0].copy(), "test": df.iloc[0:0].copy()}

    rng = np.random.default_rng(seed)
    groups = np.array(unique_groups)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = max(1, int(round(n_groups * train_ratio)))
    n_val = int(round(n_groups * val_ratio))
    n_test = n_groups - n_train - n_val
    if n_test < 1 and n_groups >= 3:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    # Ensure all groups assigned
    assigned = train_groups | val_groups | test_groups
    missing = set(groups) - assigned
    for g in missing:
        test_groups.add(g)

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def save_training_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: str | Path,
    file_format: str = "parquet",
) -> Dict[str, Path]:
    """Save split DataFrames to disk.

    If parquet engine is unavailable, falls back to CSV for each split.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_format = file_format.lower()
    if file_format not in {"parquet", "csv"}:
        raise ValueError("file_format must be 'parquet' or 'csv'")

    out_paths: Dict[str, Path] = {}
    for split_name, split_df in splits.items():
        out_path = out_dir / f"{split_name}.{file_format}"
        if file_format == "parquet":
            try:
                split_df.to_parquet(out_path, index=False)
            except ImportError:
                fallback = out_dir / f"{split_name}.csv"
                warnings.warn(
                    "Parquet engine not available. Falling back to CSV split files.",
                    UserWarning,
                    stacklevel=2,
                )
                split_df.to_csv(fallback, index=False)
                out_path = fallback
        else:
            split_df.to_csv(out_path, index=False)
        out_paths[split_name] = out_path

    return out_paths


def reports_to_dataframe(reports: Sequence[TrainingFileReport]) -> pd.DataFrame:
    """Convert file reports to DataFrame."""
    return pd.DataFrame([asdict(r) for r in reports])
