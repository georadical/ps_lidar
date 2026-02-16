"""
Benchmark geometric-feature backends (voxel vs pgeof).

Usage examples:
  python scripts/benchmark_feature_backends.py
  python scripts/benchmark_feature_backends.py --input outputs/vegetation_normalized.laz
  python scripts/benchmark_feature_backends.py --backends voxel,pgeof --repeats 3 --sample-size 300000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import benchmark_feature_backends


def _generate_synthetic_forest(n_points: int, seed: int) -> np.ndarray:
    """Generate deterministic synthetic data with trunk-like and canopy-like structures."""
    rng = np.random.default_rng(seed)

    n_trees = max(10, n_points // 60000)
    trunk_points = int(n_points * 0.45)
    canopy_points = int(n_points * 0.40)
    noise_points = n_points - trunk_points - canopy_points

    centers = rng.uniform(-40, 40, size=(n_trees, 2))

    # Trunk-like points (vertical cylinders)
    trunk_xyz = []
    per_tree = max(100, trunk_points // n_trees)
    for cx, cy in centers:
        angles = rng.uniform(0, 2 * np.pi, per_tree)
        radii = rng.normal(0.15, 0.03, per_tree)
        z = rng.uniform(0.0, 20.0, per_tree)
        x = cx + radii * np.cos(angles)
        y = cy + radii * np.sin(angles)
        trunk_xyz.append(np.column_stack([x, y, z]))
    trunk_xyz = np.vstack(trunk_xyz)[:trunk_points]

    # Canopy-like points (spherical/clustered)
    canopy_xyz = []
    per_tree_canopy = max(100, canopy_points // n_trees)
    for cx, cy in centers:
        center = np.array([cx, cy, 17.0 + rng.uniform(-2, 2)])
        pts = center + rng.normal(0, [1.8, 1.8, 1.2], size=(per_tree_canopy, 3))
        pts[:, 2] = np.clip(pts[:, 2], 8.0, 24.0)
        canopy_xyz.append(pts)
    canopy_xyz = np.vstack(canopy_xyz)[:canopy_points]

    # Low vegetation/noise
    noise_xyz = np.column_stack(
        [
            rng.uniform(-50, 50, noise_points),
            rng.uniform(-50, 50, noise_points),
            rng.uniform(0.0, 3.0, noise_points),
        ]
    )

    xyz = np.vstack([trunk_xyz, canopy_xyz, noise_xyz]).astype(np.float64)
    return xyz


def _load_xyz_from_las(path: Path) -> np.ndarray:
    import laspy

    las = laspy.read(str(path))
    return np.column_stack([las.x, las.y, las.z]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark feature backends")
    parser.add_argument("--input", type=str, default=None, help="Path to LAS/LAZ input cloud")
    parser.add_argument("--output", type=str, default="outputs/feature_backend_benchmark.json", help="JSON report output path")
    parser.add_argument("--backends", type=str, default="voxel,pgeof", help="Comma-separated backends")
    parser.add_argument("--repeats", type=int, default=2, help="Number of repeats per backend")
    parser.add_argument("--sample-size", type=int, default=200000, help="Random sample size for benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--synthetic-points", type=int, default=500000, help="Number of synthetic points if --input is not provided")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size")
    parser.add_argument("--k-neighbors", type=int, default=20, help="KNN for voxel backend")
    parser.add_argument("--pgeof-scale", type=float, default=0.15, help="Scale for pgeof backend")
    parser.add_argument("--pgeof-max-knn", type=int, default=50000, help="Max KNN for pgeof backend")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    if not backends:
        raise ValueError("No backends provided")

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        print(f"Loading point cloud: {input_path}")
        xyz = _load_xyz_from_las(input_path)
        source = str(input_path)
    else:
        print(f"Generating synthetic cloud ({args.synthetic_points:,} points)")
        xyz = _generate_synthetic_forest(args.synthetic_points, args.seed)
        source = f"synthetic:{args.synthetic_points}"

    print(f"Benchmark input points: {len(xyz):,}")
    print(f"Backends: {', '.join(backends)}")

    report = benchmark_feature_backends(
        xyz,
        backends=backends,
        repeats=args.repeats,
        sample_size=args.sample_size,
        seed=args.seed,
        voxel_size=args.voxel_size,
        k_neighbors=args.k_neighbors,
        pgeof_scale=args.pgeof_scale,
        pgeof_max_knn=args.pgeof_max_knn,
        verbose=True,
    )
    report["metadata"]["source"] = source

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nBenchmark summary")
    for backend_name, data in report["backends"].items():
        if not data.get("ok"):
            print(f"- {backend_name}: ERROR -> {data.get('error')}")
            continue
        mean_s = data["timing_seconds"]["mean"]
        std_s = data["timing_seconds"]["std"]
        print(f"- {backend_name}: {mean_s:.3f}s +/- {std_s:.3f}s")

    comp = report.get("comparisons", {}).get("pgeof_vs_voxel_mae")
    if comp:
        print(
            "- pgeof_vs_voxel_mae: "
            f"verticality={comp['verticality']:.4f}, "
            f"linearity={comp['linearity']:.4f}, "
            f"planarity={comp['planarity']:.4f}, "
            f"sphericity={comp['sphericity']:.4f}"
        )

    print(f"\nReport written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
