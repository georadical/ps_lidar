"""Unified Phase 5 workflow: build bank, train classifier, run inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (  # noqa: E402
    apply_understory_classifier_to_las,
    build_training_bank,
    load_training_data_from_dataframe,
    reports_to_dataframe,
    save_classifier_bundle,
    save_training_splits,
    split_training_bank_by_plot,
    train_and_evaluate_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 unified workflow")

    parser.add_argument("--training-inputs", nargs="+", default=["notebooks"], help="Files/folders/globs for labeled LAS/LAZ")
    parser.add_argument("--label-field", type=str, default="training_label", help="Training label field name")
    parser.add_argument("--features", nargs="+", default=["verticality", "linearity", "sphericity", "dist_to_ground"], help="Model feature list")

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--group-col", type=str, default="plot_id")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=14)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--inference-input", type=str, default="", help="Optional LAS/LAZ for inference")
    parser.add_argument("--slice-height", type=float, default=3.5)
    parser.add_argument("--backend", type=str, choices=["voxel", "pgeof"], default="voxel")

    parser.add_argument("--output-dir", type=str, default="outputs/phase5")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def _metrics_to_dict(metrics_obj):
    data = metrics_obj.__dict__.copy()
    data["confusion_matrix"] = metrics_obj.confusion_matrix.tolist()
    return data


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "models"
    split_dir = out_dir / "splits"
    infer_dir = out_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    bank_path = out_dir / "training_bank.csv"
    reports_path = out_dir / "training_bank_reports.csv"
    model_path = model_dir / "understory_rf_phase5.pkl"
    metrics_path = model_dir / "understory_rf_phase5_metrics.json"
    importance_path = model_dir / "understory_rf_phase5_feature_importances.csv"

    bank_df, bank_result = build_training_bank(
        inputs=args.training_inputs,
        output_path=bank_path,
        label_field=args.label_field,
    )

    reports_to_dataframe(bank_result.reports).to_csv(reports_path, index=False)

    print("Phase 5 - Build training bank")
    print(f"- files_scanned: {bank_result.n_files_scanned}")
    print(f"- files_usable: {bank_result.n_files_usable}")
    print(f"- points_total: {bank_result.n_points_total:,}")
    print(f"- points_kept: {bank_result.n_points_kept:,}")
    print(f"- bank_output: {bank_result.output_path}")
    print(f"- reports_output: {reports_path}")

    if bank_df.empty:
        print("No usable labeled data found. Stopping before training.")
        return 0

    splits = split_training_bank_by_plot(
        bank_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_col=args.group_col,
    )
    split_paths = save_training_splits(splits, split_dir, file_format="csv")

    print("Phase 5 - Splits")
    for split_name, split_df in splits.items():
        print(f"- {split_name}: {len(split_df):,} -> {split_paths[split_name]}")

    train_data = load_training_data_from_dataframe(splits["train"], feature_names=args.features, label_col="label")
    val_data = None if splits["val"].empty else load_training_data_from_dataframe(splits["val"], feature_names=args.features, label_col="label")
    test_data = None if splits["test"].empty else load_training_data_from_dataframe(splits["test"], feature_names=args.features, label_col="label")

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

    save_classifier_bundle(
        classifier=classifier,
        filepath=model_path,
        feature_names=args.features,
        metadata={
            "threshold": args.threshold,
            "feature_names": args.features,
            "split_sizes": {name: int(len(df)) for name, df in splits.items()},
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
    )

    metrics_payload = {name: _metrics_to_dict(metric) for name, metric in metrics.items()}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    pd.DataFrame(
        {
            "feature": args.features,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(importance_path, index=False)

    print("Phase 5 - Training complete")
    print(f"- model_output: {model_path}")
    print(f"- metrics_output: {metrics_path}")
    print(f"- importance_output: {importance_path}")
    for split_name, metric in metrics.items():
        print(
            f"- {split_name}: acc={metric.accuracy:.3f}, "
            f"bal_acc={metric.balanced_accuracy:.3f}, "
            f"f1_under={metric.f1_understory:.3f}, f1_tree={metric.f1_tree:.3f}"
        )

    if args.inference_input:
        inference_input = Path(args.inference_input)
        if not inference_input.exists():
            print(f"Inference input not found: {inference_input}. Skipping inference.")
            return 0

        inference_output = infer_dir / f"{inference_input.stem}_ml_phase5.laz"
        inference_result = apply_understory_classifier_to_las(
            input_path=inference_input,
            model_path=model_path,
            output_path=inference_output,
            probability_threshold=args.threshold,
            slice_height=args.slice_height,
            backend=args.backend,
            dist_source="auto",
            verbose=args.verbose,
        )

        print("Phase 5 - Inference complete")
        print(f"- inference_output: {inference_result.output_path}")
        print(f"- points: {inference_result.n_points:,}")
        print(f"- tree: {inference_result.n_tree:,}")
        print(f"- understory: {inference_result.n_understory:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
