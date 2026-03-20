"""Voxel-based sample bank builder for binary stem vs no-stem ML workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import glob
import math
from typing import Iterable, Sequence

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .features import compute_linearity, compute_verticality


DEFAULT_LABEL_MAP: dict[int, str] = {
    0: "no_stem",
    1: "stem",
}

_MIN_NEIGHBORS = 5
_NEIGHBORHOOD_FACTOR = 3.0
_EPS = 1e-12


@dataclass(slots=True)
class SampleBankFileReport:
    """Per-file inspection report for sample bank ingestion."""

    source_file: str
    plot_id: str
    status: str
    message: str
    n_points_total: int
    n_points_labeled: int
    n_points_kept: int
    n_voxels_total: int
    n_voxels_kept: int
    label_field: str
    labels_found: str


@dataclass(slots=True)
class SampleBankBuildResult:
    """Build summary for a complete sample-bank run."""

    n_files_scanned: int
    n_files_usable: int
    n_points_total: int
    n_points_kept: int
    output_path: str | None
    reports: list[SampleBankFileReport]


def _expand_inputs(inputs: Sequence[str | Path]) -> list[Path]:
    files: set[Path] = set()

    for raw in inputs:
        value = Path(raw)
        raw_str = str(raw)

        if any(ch in raw_str for ch in "*?[]"):
            for match in glob.glob(raw_str, recursive=True):
                path = Path(match)
                if path.is_file() and path.suffix.lower() in {".las", ".laz"}:
                    files.add(path.resolve())
            continue

        if value.is_file() and value.suffix.lower() in {".las", ".laz"}:
            files.add(value.resolve())
            continue

        if value.is_dir():
            for path in value.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".las", ".laz"}:
                    files.add(path.resolve())

    return sorted(files)


def _save_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return output_path

    if suffix == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing parquet requires optional ML dependencies. "
                "Install requirements/ml.txt or choose a .csv output path."
            ) from exc
        return output_path

    raise ValueError(f"Unsupported output format '{suffix}'. Use .csv or .parquet.")


def _compute_voxel_geometry(
    xyz: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voxel_index = np.floor(xyz / voxel_size).astype(np.int64)
    unique_voxels, inverse, counts = np.unique(
        voxel_index,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    centroids = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    np.add.at(centroids, inverse, xyz)
    centroids /= counts[:, None]

    return unique_voxels, inverse, centroids


def _compute_local_shape_features(
    centroids: np.ndarray,
    counts: np.ndarray,
    voxel_size: float,
) -> dict[str, np.ndarray]:
    n_voxels = len(centroids)
    if n_voxels == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "planarity": empty,
            "sphericity": empty,
            "anisotropy": empty,
            "surface_variation": empty,
            "roughness": empty,
            "neighbor_count": empty,
            "volume_density": empty,
        }

    radius = max(voxel_size * _NEIGHBORHOOD_FACTOR, voxel_size + 0.05)
    sphere_volume = (4.0 / 3.0) * math.pi * (radius**3)
    tree = cKDTree(centroids)

    planarity = np.zeros(n_voxels, dtype=np.float64)
    sphericity = np.zeros(n_voxels, dtype=np.float64)
    anisotropy = np.zeros(n_voxels, dtype=np.float64)
    surface_variation = np.zeros(n_voxels, dtype=np.float64)
    roughness = np.zeros(n_voxels, dtype=np.float64)
    neighbor_count = np.zeros(n_voxels, dtype=np.float64)
    volume_density = np.zeros(n_voxels, dtype=np.float64)

    k_fallback = min(max(_MIN_NEIGHBORS, 3), n_voxels)

    for idx in range(n_voxels):
        neighbor_ids = tree.query_ball_point(centroids[idx], radius)
        if len(neighbor_ids) < _MIN_NEIGHBORS and k_fallback > 0:
            _, knn_ids = tree.query(centroids[idx], k=k_fallback)
            neighbor_ids = np.atleast_1d(knn_ids).astype(np.int64).tolist()

        neighbor_ids = sorted(set(int(i) for i in neighbor_ids))
        local_xyz = centroids[neighbor_ids]
        local_counts = counts[neighbor_ids]

        neighbor_count[idx] = float(len(neighbor_ids))
        volume_density[idx] = float(np.sum(local_counts) / max(sphere_volume, _EPS))

        if len(local_xyz) < 3:
            continue

        centered = local_xyz - np.mean(local_xyz, axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False, bias=True)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        lam1, lam2, lam3 = eigvals[::-1]

        if lam1 > _EPS:
            planarity[idx] = float((lam2 - lam3) / lam1)
            sphericity[idx] = float(lam3 / lam1)
            anisotropy[idx] = float((lam1 - lam3) / lam1)

        lam_sum = lam1 + lam2 + lam3
        if lam_sum > _EPS:
            surface_variation[idx] = float(lam3 / lam_sum)

        normal = eigvecs[:, 0]
        distances = centered @ normal
        roughness[idx] = float(np.sqrt(np.mean(distances**2)))

    return {
        "planarity": planarity,
        "sphericity": sphericity,
        "anisotropy": anisotropy,
        "surface_variation": surface_variation,
        "roughness": roughness,
        "neighbor_count": neighbor_count,
        "volume_density": volume_density,
    }


def _labels_from_las(
    las: laspy.LasData,
    label_field: str,
) -> np.ndarray | None:
    dimensions = set(las.point_format.dimension_names)
    if label_field not in dimensions:
        return None
    return np.asarray(las[label_field])


def _build_rows_for_file(
    path: Path,
    label_field: str,
    voxel_size: float,
    label_map: dict[int, str],
) -> tuple[pd.DataFrame, SampleBankFileReport]:
    las = laspy.read(path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    plot_id = path.stem

    labels_raw = _labels_from_las(las, label_field)
    if labels_raw is None:
        report = SampleBankFileReport(
            source_file=str(path),
            plot_id=plot_id,
            status="missing_label_field",
            message=f"Field '{label_field}' not found.",
            n_points_total=int(len(xyz)),
            n_points_labeled=0,
            n_points_kept=0,
            n_voxels_total=0,
            n_voxels_kept=0,
            label_field=label_field,
            labels_found="",
        )
        return pd.DataFrame(), report

    labels_numeric = pd.to_numeric(pd.Series(labels_raw), errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(labels_numeric)
    valid_label_mask = np.zeros_like(valid_mask, dtype=bool)
    valid_label_mask[valid_mask] = np.isin(labels_numeric[valid_mask].astype(np.int64), list(label_map))
    valid_mask &= valid_label_mask
    n_points_labeled = int(np.sum(valid_mask))

    if n_points_labeled == 0:
        report = SampleBankFileReport(
            source_file=str(path),
            plot_id=plot_id,
            status="no_usable_labels",
            message="No points with usable binary labels were found.",
            n_points_total=int(len(xyz)),
            n_points_labeled=0,
            n_points_kept=0,
            n_voxels_total=0,
            n_voxels_kept=0,
            label_field=label_field,
            labels_found="",
        )
        return pd.DataFrame(), report

    voxel_keys, point_to_voxel, centroids = _compute_voxel_geometry(xyz, voxel_size)
    n_voxels = len(centroids)
    voxel_counts = np.bincount(point_to_voxel, minlength=n_voxels)

    vert = compute_verticality(
        xyz,
        scale=max(voxel_size * _NEIGHBORHOOD_FACTOR, voxel_size + 0.05),
        voxel_resolution_xy=voxel_size,
        voxel_resolution_z=voxel_size,
        verbose=False,
    )
    lin = compute_linearity(
        xyz,
        scale=max(voxel_size * _NEIGHBORHOOD_FACTOR, voxel_size + 0.05),
        voxel_resolution_xy=voxel_size,
        voxel_resolution_z=voxel_size,
        verbose=False,
    )

    vert_mean = np.zeros(n_voxels, dtype=np.float64)
    lin_mean = np.zeros(n_voxels, dtype=np.float64)
    np.add.at(vert_mean, point_to_voxel, vert)
    np.add.at(lin_mean, point_to_voxel, lin)
    vert_mean /= np.maximum(voxel_counts, 1)
    lin_mean /= np.maximum(voxel_counts, 1)

    local_shape = _compute_local_shape_features(centroids, voxel_counts, voxel_size)

    labels_int = np.full(len(labels_numeric), fill_value=-1, dtype=np.int64)
    labels_int[valid_mask] = labels_numeric[valid_mask].astype(np.int64)

    rows: list[dict[str, object]] = []
    kept_points = 0

    labels_found = sorted(set(int(v) for v in labels_int[labels_int >= 0]))

    for voxel_id in range(n_voxels):
        members = point_to_voxel == voxel_id
        voxel_labels = labels_int[members]
        voxel_labels = voxel_labels[voxel_labels >= 0]
        if len(voxel_labels) == 0:
            continue

        unique_labels, label_counts = np.unique(voxel_labels, return_counts=True)
        dominant_idx = int(np.argmax(label_counts))
        dominant_label = int(unique_labels[dominant_idx])
        dominant_count = int(label_counts[dominant_idx])
        dominant_fraction = float(dominant_count / max(voxel_counts[voxel_id], 1))

        if dominant_fraction < 0.8:
            continue

        kept_points += int(voxel_counts[voxel_id])
        ix, iy, iz = voxel_keys[voxel_id].tolist()
        rows.append(
            {
                "plot_id": plot_id,
                "source_file": str(path),
                "voxel_key": f"{ix}_{iy}_{iz}",
                "x": float(centroids[voxel_id, 0]),
                "y": float(centroids[voxel_id, 1]),
                "z": float(centroids[voxel_id, 2]),
                "n_points": int(voxel_counts[voxel_id]),
                "dominant_fraction": dominant_fraction,
                "label": dominant_label,
                "label_name": label_map[dominant_label],
                "verticality": float(vert_mean[voxel_id]),
                "linearity": float(lin_mean[voxel_id]),
                "planarity": float(local_shape["planarity"][voxel_id]),
                "sphericity": float(local_shape["sphericity"][voxel_id]),
                "anisotropy": float(local_shape["anisotropy"][voxel_id]),
                "surface_variation": float(local_shape["surface_variation"][voxel_id]),
                "roughness": float(local_shape["roughness"][voxel_id]),
                "neighbor_count": float(local_shape["neighbor_count"][voxel_id]),
                "volume_density": float(local_shape["volume_density"][voxel_id]),
            }
        )

    frame = pd.DataFrame(rows)
    report = SampleBankFileReport(
        source_file=str(path),
        plot_id=plot_id,
        status="ok" if not frame.empty else "no_kept_voxels",
        message="" if not frame.empty else "No voxels passed the dominant label filter.",
        n_points_total=int(len(xyz)),
        n_points_labeled=n_points_labeled,
        n_points_kept=kept_points,
        n_voxels_total=n_voxels,
        n_voxels_kept=int(len(frame)),
        label_field=label_field,
        labels_found=",".join(str(v) for v in labels_found),
    )
    return frame, report


def build_sample_bank(
    inputs: Sequence[str | Path],
    label_field: str = "training_label",
    voxel_size: float = 0.05,
    output_path: str | Path | None = None,
    label_map: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, SampleBankBuildResult]:
    """Build a voxel sample bank from manually labeled LAS/LAZ files."""
    resolved_files = _expand_inputs(inputs)
    active_label_map = dict(label_map or DEFAULT_LABEL_MAP)
    frames: list[pd.DataFrame] = []
    reports: list[SampleBankFileReport] = []
    n_points_total = 0
    n_points_kept = 0
    n_files_usable = 0

    for path in resolved_files:
        frame, report = _build_rows_for_file(
            path=path,
            label_field=label_field,
            voxel_size=voxel_size,
            label_map=active_label_map,
        )
        n_points_total += report.n_points_total
        n_points_kept += report.n_points_kept
        if report.status == "ok":
            n_files_usable += 1
        reports.append(report)
        if not frame.empty:
            frames.append(frame)

    if frames:
        bank_df = pd.concat(frames, ignore_index=True)
    else:
        bank_df = pd.DataFrame(
            columns=[
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
            ]
        )

    written_path: str | None = None
    if output_path is not None:
        saved = _save_dataframe(bank_df, Path(output_path))
        written_path = str(saved)

    result = SampleBankBuildResult(
        n_files_scanned=len(resolved_files),
        n_files_usable=n_files_usable,
        n_points_total=n_points_total,
        n_points_kept=n_points_kept,
        output_path=written_path,
        reports=reports,
    )
    return bank_df, result


def split_sample_bank_by_plot(
    bank_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_col: str = "plot_id",
) -> dict[str, pd.DataFrame]:
    """Split sample bank by plot to avoid leakage across plots."""
    if bank_df.empty:
        return {
            "train": bank_df.copy(),
            "val": bank_df.copy(),
            "test": bank_df.copy(),
        }

    if group_col not in bank_df.columns:
        raise ValueError(f"Grouping column '{group_col}' not found in sample bank.")

    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive.")

    groups = pd.Index(bank_df[group_col].dropna().unique())
    rng = np.random.default_rng(seed)
    shuffled = groups.to_numpy(copy=True)
    rng.shuffle(shuffled)

    n_groups = len(shuffled)
    n_train = int(round((train_ratio / ratio_sum) * n_groups))
    n_val = int(round((val_ratio / ratio_sum) * n_groups))
    if n_train + n_val > n_groups:
        n_val = max(0, n_groups - n_train)
    n_test = max(0, n_groups - n_train - n_val)

    train_groups = set(shuffled[:n_train])
    val_groups = set(shuffled[n_train:n_train + n_val])
    test_groups = set(shuffled[n_train + n_val:n_train + n_val + n_test])

    train_df = bank_df[bank_df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = bank_df[bank_df[group_col].isin(val_groups)].reset_index(drop=True)
    test_df = bank_df[bank_df[group_col].isin(test_groups)].reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def save_sample_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    file_format: str = "parquet",
) -> dict[str, Path]:
    """Persist train/val/test splits to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_format not in {"csv", "parquet"}:
        raise ValueError("file_format must be 'csv' or 'parquet'.")

    paths: dict[str, Path] = {}
    for split_name, df in splits.items():
        out_path = out_dir / f"{split_name}.{file_format}"
        _save_dataframe(df, out_path)
        paths[split_name] = out_path
    return paths


def reports_to_dataframe(reports: Iterable[SampleBankFileReport]) -> pd.DataFrame:
    """Convert build reports to a dataframe."""
    return pd.DataFrame([asdict(report) for report in reports])
