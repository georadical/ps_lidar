"""Build and split a voxel sample bank from manually labeled LAS/LAZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (  # noqa: E402
    build_sample_bank,
    reports_to_dataframe,
    save_sample_splits,
    split_sample_bank_by_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stem vs no-stem voxel sample bank")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files, folders, or glob patterns (e.g. data/labeled/*.laz)",
    )
    parser.add_argument("--label-field", type=str, default="training_label", help="Manual label field name")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Cubic voxel size in meters")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/stem_sample_bank.parquet",
        help="Output sample bank path (.parquet/.csv)",
    )
    parser.add_argument(
        "--reports-output",
        type=str,
        default="outputs/stem_sample_bank_reports.csv",
        help="Per-file report output (.csv)",
    )
    parser.add_argument("--no-split", action="store_true", help="Skip train/val/test split generation")
    parser.add_argument(
        "--split-dir",
        type=str,
        default="outputs/stem_sample_bank_splits",
        help="Directory for split files",
    )
    parser.add_argument(
        "--split-format",
        type=str,
        default="parquet",
        choices=["csv", "parquet"],
        help="Split output format",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bank_df, result = build_sample_bank(
        inputs=args.inputs,
        label_field=args.label_field,
        voxel_size=args.voxel_size,
        output_path=args.output,
    )

    print("Stem sample bank build summary")
    print(f"- files_scanned: {result.n_files_scanned}")
    print(f"- files_usable:  {result.n_files_usable}")
    print(f"- points_total:  {result.n_points_total:,}")
    print(f"- points_kept:   {result.n_points_kept:,}")
    print(f"- bank_rows:     {len(bank_df):,}")
    print(f"- bank_output:   {result.output_path or args.output}")

    reports_df = reports_to_dataframe(result.reports)
    reports_output = Path(args.reports_output)
    reports_output.parent.mkdir(parents=True, exist_ok=True)
    reports_df.to_csv(reports_output, index=False)
    print(f"- reports_output: {reports_output}")

    summary_path = reports_output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "n_files_scanned": result.n_files_scanned,
                "n_files_usable": result.n_files_usable,
                "n_points_total": result.n_points_total,
                "n_points_kept": result.n_points_kept,
                "bank_rows": int(len(bank_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if bank_df.empty or args.no_split:
        return 0

    splits = split_sample_bank_by_plot(
        bank_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_paths = save_sample_splits(
        splits,
        output_dir=args.split_dir,
        file_format=args.split_format,
    )

    print("Split summary")
    for split_name, split_df in splits.items():
        print(f"- {split_name}: {len(split_df):,} rows -> {split_paths[split_name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
