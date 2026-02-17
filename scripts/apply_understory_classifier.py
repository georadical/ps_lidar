"""Apply trained understory classifier to LAS/LAZ and export predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import apply_understory_classifier_to_las  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply trained understory classifier to LAS/LAZ")

    parser.add_argument("--input", type=str, required=True, help="Input LAS/LAZ path")
    parser.add_argument("--model", type=str, required=True, help="Trained model .pkl path")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output LAS/LAZ path (default: outputs/ml/<input_stem>_ml.laz)",
    )

    parser.add_argument("--features", nargs="+", default=None, help="Override model feature names")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold (default from model metadata or 0.5)",
    )

    parser.add_argument("--slice-height", type=float, default=3.5, help="Protected upper slice height in meters")
    parser.add_argument("--no-slice", action="store_true", help="Disable slice protection")

    parser.add_argument(
        "--dist-source",
        type=str,
        default="auto",
        choices=["auto", "z", "relative"],
        help="Source for dist_to_ground when needed",
    )
    parser.add_argument("--backend", type=str, default="voxel", choices=["voxel", "pgeof"], help="Feature backend")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size for computed features")
    parser.add_argument("--k-neighbors", type=int, default=20, help="KNN neighbors for voxel backend")
    parser.add_argument("--pgeof-scale", type=float, default=0.15, help="pgeof scale")
    parser.add_argument("--pgeof-max-knn", type=int, default=50000, help="pgeof max knn")

    parser.add_argument("--understory-prob-field", type=str, default="understory_prob", help="Output field name")
    parser.add_argument("--tree-prob-field", type=str, default="tree_prob", help="Output field name")
    parser.add_argument("--mask-field", type=str, default="is_tree_ml", help="Output field name")

    parser.add_argument("--write-computed-features", action="store_true", help="Write computed feature columns to output")
    parser.add_argument("--update-classification", action="store_true", help="Overwrite LAS classification field")
    parser.add_argument("--classification-tree", type=int, default=5, help="Classification code for tree points")
    parser.add_argument("--classification-understory", type=int, default=3, help="Classification code for understory")

    parser.add_argument("--compress", action="store_true", default=True, help="Compress output if .laz")
    parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("outputs/ml") / f"{input_path.stem}_ml.laz"

    slice_height = None if args.no_slice else float(args.slice_height)
    compress = False if args.no_compress else bool(args.compress)

    result = apply_understory_classifier_to_las(
        input_path=input_path,
        model_path=args.model,
        output_path=output_path,
        feature_names=args.features,
        probability_threshold=args.threshold,
        slice_height=slice_height,
        dist_source=args.dist_source,
        backend=args.backend,
        voxel_size=args.voxel_size,
        k_neighbors=args.k_neighbors,
        pgeof_scale=args.pgeof_scale,
        pgeof_max_knn=args.pgeof_max_knn,
        understory_prob_field=args.understory_prob_field,
        tree_prob_field=args.tree_prob_field,
        mask_field=args.mask_field,
        write_computed_features=args.write_computed_features,
        update_classification=args.update_classification,
        classification_tree=args.classification_tree,
        classification_understory=args.classification_understory,
        compress=compress,
        verbose=args.verbose,
    )

    print("Inference completed")
    print(f"- input: {result.input_path}")
    print(f"- output: {result.output_path}")
    print(f"- points: {result.n_points:,}")
    print(f"- tree: {result.n_tree:,}")
    print(f"- understory: {result.n_understory:,}")
    print(f"- threshold: {result.probability_threshold:.3f}")
    print(f"- slice_height: {result.slice_height}")
    print(f"- features: {list(result.feature_names)}")
    print(f"- computed_features: {list(result.computed_features)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
