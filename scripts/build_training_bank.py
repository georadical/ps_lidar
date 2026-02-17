"""
Build and split a training data bank from labeled LAS/LAZ files.

Examples:
  python scripts/build_training_bank.py --inputs data/labeled/*.laz
  python scripts/build_training_bank.py --inputs notebooks --label-field training_label --output outputs/training_bank.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    build_training_bank,
    split_training_bank_by_plot,
    save_training_splits,
    reports_to_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training bank from labeled LAS/LAZ")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files, folders, or glob patterns (e.g. data/labeled/*.laz)",
    )
    parser.add_argument("--label-field", type=str, default="training_label", help="Label field name in LAZ")
    parser.add_argument("--output", type=str, default="outputs/training_bank.parquet", help="Output bank file (.parquet/.csv)")
    parser.add_argument(
        "--reports-output",
        type=str,
        default="outputs/training_bank_reports.csv",
        help="Per-file inspection report output (.csv)",
    )
    parser.add_argument("--no-split", action="store_true", help="Do not create train/val/test split files")
    parser.add_argument("--split-dir", type=str, default="outputs/training_bank_splits", help="Split output directory")
    parser.add_argument("--split-format", type=str, default="parquet", choices=["parquet", "csv"], help="Split file format")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Split random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bank_df, result = build_training_bank(
        inputs=args.inputs,
        output_path=args.output,
        label_field=args.label_field,
    )

    print("Training bank build summary")
    print(f"- files_scanned: {result.n_files_scanned}")
    print(f"- files_usable:  {result.n_files_usable}")
    print(f"- points_total:  {result.n_points_total:,}")
    print(f"- points_kept:   {result.n_points_kept:,}")
    print(f"- bank_output:   {result.output_path or args.output}")

    reports_df = reports_to_dataframe(result.reports)
    reports_output = Path(args.reports_output)
    reports_output.parent.mkdir(parents=True, exist_ok=True)
    reports_df.to_csv(reports_output, index=False)
    print(f"- reports_output: {reports_output}")

    # Also save compact JSON summary
    summary_path = reports_output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "n_files_scanned": result.n_files_scanned,
                "n_files_usable": result.n_files_usable,
                "n_points_total": result.n_points_total,
                "n_points_kept": result.n_points_kept,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if bank_df.empty:
        print("No usable labeled files found. Split step skipped.")
        return 0

    if args.no_split:
        return 0

    splits = split_training_bank_by_plot(
        bank_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_paths = save_training_splits(
        splits,
        output_dir=args.split_dir,
        file_format=args.split_format,
    )

    print("Split summary")
    for split_name, split_df in splits.items():
        out_path = split_paths[split_name]
        print(f"- {split_name}: {len(split_df):,} rows -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
