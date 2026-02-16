"""
Calibrate classify_understory thresholds for pgeof backend.

Goal:
- Keep current rule behavior as reference (voxel backend).
- Find pgeof thresholds that maximize agreement with voxel labels.

Usage:
  python scripts/calibrate_pgeof_thresholds.py
  python scripts/calibrate_pgeof_thresholds.py --input notebooks/trees.laz --sample-size 250000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np

# Add project root for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import compute_all_features_fast, classify_understory


def _load_xyz(path: Path) -> np.ndarray:
    import laspy

    las = laspy.read(str(path))
    return np.column_stack([las.x, las.y, las.z]).astype(np.float64)


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    den = (2 * tp) + fp + fn
    return float((2 * tp) / den) if den > 0 else 0.0


def _agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _pick_default_input() -> Path:
    candidates = [
        Path("notebooks/trees.laz"),
        Path("notebooks/trees_02.laz"),
        Path("notebooks/trees_03.laz"),
        Path("notebooks/understory.laz"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No default LAZ found in notebooks/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate pgeof thresholds")
    parser.add_argument("--input", type=str, default=None, help="Input LAS/LAZ")
    parser.add_argument("--sample-size", type=int, default=200000, help="Random sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size for features")
    parser.add_argument("--k-neighbors", type=int, default=20, help="KNN for voxel backend")
    parser.add_argument("--pgeof-scale", type=float, default=0.15, help="Scale for pgeof backend")
    parser.add_argument("--pgeof-max-knn", type=int, default=50000, help="Max KNN for pgeof backend")
    parser.add_argument("--output", type=str, default="outputs/pgeof_threshold_profile.json", help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input) if args.input else _pick_default_input()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Loading input: {input_path}")
    xyz = _load_xyz(input_path)
    print(f"Loaded points: {len(xyz):,}")

    if args.sample_size and args.sample_size < len(xyz):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(xyz), size=args.sample_size, replace=False)
        xyz_eval = xyz[idx]
    else:
        xyz_eval = xyz

    print(f"Calibration points: {len(xyz_eval):,}")

    # Baseline features + labels (voxel backend)
    print("Computing baseline (voxel) features...")
    f_voxel, d_ground, d_top = compute_all_features_fast(
        xyz_eval,
        voxel_size=args.voxel_size,
        k_neighbors=args.k_neighbors,
        backend="voxel",
        verbose=False,
    )
    base = classify_understory(
        xyz_eval,
        f_voxel.verticality,
        f_voxel.linearity,
        f_voxel.sphericity,
        d_ground,
        d_top,
        verticality_threshold=0.7,
        linearity_threshold=0.4,
        sphericity_threshold=0.3,
        use_height_adaptive=True,
        verbose=False,
    )

    # pgeof features (fixed, thresholds to calibrate)
    print("Computing pgeof features...")
    f_pgeof, d_ground_pg, d_top_pg = compute_all_features_fast(
        xyz_eval,
        voxel_size=args.voxel_size,
        backend="pgeof",
        pgeof_scale=args.pgeof_scale,
        pgeof_max_knn=args.pgeof_max_knn,
        verbose=False,
    )

    # Grid search thresholds
    v_grid = np.arange(0.35, 0.96, 0.05)
    l_grid = np.arange(0.20, 0.71, 0.05)
    s_grid = np.arange(0.10, 0.61, 0.05)

    best = None
    best_metrics: Dict[str, float] = {}

    print(f"Grid search size: {len(v_grid) * len(l_grid) * len(s_grid):,} combinations")
    tested = 0
    for vt in v_grid:
        for lt in l_grid:
            for st in s_grid:
                pred = classify_understory(
                    xyz_eval,
                    f_pgeof.verticality,
                    f_pgeof.linearity,
                    f_pgeof.sphericity,
                    d_ground_pg,
                    d_top_pg,
                    verticality_threshold=float(vt),
                    linearity_threshold=float(lt),
                    sphericity_threshold=float(st),
                    use_height_adaptive=True,
                    verbose=False,
                )

                f1_understory = _f1_binary(base.is_understory, pred.is_understory)
                f1_stem = _f1_binary(base.is_stem, pred.is_stem)
                agr_understory = _agreement(base.is_understory, pred.is_understory)
                agr_stem = _agreement(base.is_stem, pred.is_stem)
                # Balanced objective to avoid degenerate solutions in imbalanced classes.
                objective = (0.45 * f1_understory) + (0.45 * f1_stem) + (0.10 * agr_understory)

                if best is None or objective > best_metrics["objective"]:
                    best = (float(vt), float(lt), float(st))
                    best_metrics = {
                        "objective": float(objective),
                        "f1_understory": float(f1_understory),
                        "f1_stem": float(f1_stem),
                        "agreement_understory": float(agr_understory),
                        "agreement_stem": float(agr_stem),
                        "pred_understory_ratio": float(np.mean(pred.is_understory)),
                        "pred_stem_ratio": float(np.mean(pred.is_stem)),
                    }

                tested += 1
        print(f"  progress: {tested:,}/{len(v_grid) * len(l_grid) * len(s_grid):,}")

    assert best is not None
    best_v, best_l, best_s = best

    report = {
        "input": str(input_path),
        "n_points": int(len(xyz)),
        "n_points_calibrated": int(len(xyz_eval)),
        "defaults_reference": {
            "backend": "voxel",
            "verticality_threshold": 0.7,
            "linearity_threshold": 0.4,
            "sphericity_threshold": 0.3,
            "use_height_adaptive": True,
            "understory_ratio": float(np.mean(base.is_understory)),
            "stem_ratio": float(np.mean(base.is_stem)),
        },
        "recommended_for_pgeof": {
            "backend": "pgeof",
            "verticality_threshold": best_v,
            "linearity_threshold": best_l,
            "sphericity_threshold": best_s,
            "use_height_adaptive": True,
            **best_metrics,
        },
        "search_space": {
            "verticality": [float(v_grid[0]), float(v_grid[-1]), float(v_grid[1] - v_grid[0])],
            "linearity": [float(l_grid[0]), float(l_grid[-1]), float(l_grid[1] - l_grid[0])],
            "sphericity": [float(s_grid[0]), float(s_grid[-1]), float(s_grid[1] - s_grid[0])],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nCalibration result (pgeof)")
    print(f"- verticality_threshold: {best_v:.2f}")
    print(f"- linearity_threshold:   {best_l:.2f}")
    print(f"- sphericity_threshold:  {best_s:.2f}")
    print(f"- objective:             {best_metrics['objective']:.4f}")
    print(f"- F1 understory:         {best_metrics['f1_understory']:.4f}")
    print(f"- F1 stem:               {best_metrics['f1_stem']:.4f}")
    print(f"- agreement understory:  {best_metrics['agreement_understory']:.4f}")
    print(f"- agreement stem:        {best_metrics['agreement_stem']:.4f}")
    print(f"\nProfile written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
