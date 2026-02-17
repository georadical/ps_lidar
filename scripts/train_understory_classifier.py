"""
Train and evaluate understory classifier from training-bank data.

Examples:
  python scripts/train_understory_classifier.py --bank outputs/training_bank.csv
  python scripts/train_understory_classifier.py --split-dir outputs/training_bank_splits
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (  # noqa: E402
    load_training_data_from_dataframe,
    save_classifier_bundle,
    split_training_bank_by_plot,
    train_and_evaluate_classifier,
)


DEFAULT_FEATURES = ["verticality", "linearity", "sphericity", "dist_to_ground"]


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension '{suffix}' for {path}")


def _find_split_file(split_dir: Path, split_name: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        candidate = split_dir / f"{split_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _metrics_to_dict(metrics_obj):
    data = asdict(metrics_obj)
    data["confusion_matrix"] = metrics_obj.confusion_matrix.tolist()
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Random Forest understory classifier")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bank", type=str, help="Path to training bank (.csv/.parquet)")
    source.add_argument("--split-dir", type=str, help="Directory containing train/val/test split files")

    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name")
    parser.add_argument("--group-col", type=str, default="plot_id", help="Grouping column for split from --bank")

    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio when splitting --bank")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio when splitting --bank")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio when splitting --bank")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--n-estimators", type=int, default=300, help="Random forest n_estimators")
    parser.add_argument("--max-depth", type=int, default=14, help="Random forest max_depth")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")

    parser.add_argument("--model-output", type=str, default="outputs/models/understory_rf.pkl", help="Output model path")
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="outputs/models/understory_rf_metrics.json",
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--importance-output",
        type=str,
        default="outputs/models/understory_rf_feature_importances.csv",
        help="Output feature importance CSV path",
    )
    parser.add_argument(
        "--save-generated-splits-dir",
        type=str,
        default="",
        help="Optional directory to save splits when --bank is used",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose training logs")

    return parser.parse_args()


def _load_or_build_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if args.split_dir:
        split_dir = Path(args.split_dir)
        split_paths = {
            name: _find_split_file(split_dir, name)
            for name in ("train", "val", "test")
        }
        if split_paths["train"] is None:
            raise FileNotFoundError(
                f"Train split not found in {split_dir} (expected train.csv or train.parquet)"
            )

        splits = {"train": _load_table(split_paths["train"])}
        for name in ("val", "test"):
            path = split_paths[name]
            if path is not None:
                splits[name] = _load_table(path)
        return splits

    bank_path = Path(args.bank)
    bank_df = _load_table(bank_path)
    if args.group_col not in bank_df.columns:
        raise ValueError(
            f"group column '{args.group_col}' not found in bank. Available columns: {list(bank_df.columns)}"
        )

    splits = split_training_bank_by_plot(
        bank_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_col=args.group_col,
    )

    if args.save_generated_splits_dir:
        out_dir = Path(args.save_generated_splits_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in splits.items():
            out_file = out_dir / f"{name}.csv"
            df.to_csv(out_file, index=False)

    return splits


def main() -> int:
    args = parse_args()
    splits = _load_or_build_splits(args)

    train_data = load_training_data_from_dataframe(
        splits["train"],
        feature_names=args.features,
        label_col=args.label_col,
    )
    val_data = None
    test_data = None

    if "val" in splits and not splits["val"].empty:
        val_data = load_training_data_from_dataframe(
            splits["val"],
            feature_names=args.features,
            label_col=args.label_col,
        )

    if "test" in splits and not splits["test"].empty:
        test_data = load_training_data_from_dataframe(
            splits["test"],
            feature_names=args.features,
            label_col=args.label_col,
        )

    classifier, metrics = train_and_evaluate_classifier(
        training_data=train_data,
        validation_data=val_data,
        test_data=test_data,
        probability_threshold=args.threshold,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        verbose=args.verbose,
    )

    model_path = Path(args.model_output)
    metrics_path = Path(args.metrics_output)
    importance_path = Path(args.importance_output)

    metadata = {
        "feature_names": args.features,
        "label_col": args.label_col,
        "threshold": args.threshold,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "split_sizes": {name: int(len(df)) for name, df in splits.items()},
    }
    save_classifier_bundle(
        classifier=classifier,
        filepath=model_path,
        feature_names=args.features,
        metadata=metadata,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {name: _metrics_to_dict(m) for name, m in metrics.items()}
    metrics_payload["metadata"] = metadata
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df = pd.DataFrame(
        {
            "feature": args.features,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(importance_path, index=False)

    print("Training completed")
    print(f"- model_output: {model_path}")
    print(f"- metrics_output: {metrics_path}")
    print(f"- importance_output: {importance_path}")

    for split_name, split_metrics in metrics.items():
        print(
            "- "
            f"{split_name}: "
            f"acc={split_metrics.accuracy:.3f}, "
            f"bal_acc={split_metrics.balanced_accuracy:.3f}, "
            f"f1_understory={split_metrics.f1_understory:.3f}, "
            f"f1_tree={split_metrics.f1_tree:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
