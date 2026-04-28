"""
Benchmark voxel-level early-exit in clean_stems() against the baseline.

Compares two runs of `clean_stems` on the same input:
  - baseline:    config.voxel_early_exit = False
  - early-exit:  config.voxel_early_exit = True

Reports per-run:
  - clean_stems wall time (mean over --repeats)
  - n_points_after, n_points_removed
  - n_points_processed_verticality (the actual workload of the heavy stage)

And the comparison:
  - speedup = baseline_time / early_exit_time
  - mask_diff_pct = % of trunk points that flip classification vs baseline
  - mask_diff_lost = points kept by baseline but rejected by early-exit (recall loss)
  - mask_diff_gained = points rejected by baseline but kept by early-exit (extra retention)

Decision criteria (suggested defaults):
  - PROMOTE TO DEFAULT if  speedup >= 1.20  AND  mask_diff_pct < 0.5%  AND  mask_diff_lost == 0
  - KEEP OPT-IN          otherwise
The script just prints the recommendation; the final call is yours.

USAGE
-----

From a notebook (recommended — reuses the trunk_result you already have):

    from scripts.benchmark_voxel_early_exit import compare_early_exit
    from src.core.trunk_validation import StemCleaningConfig

    base_cfg = StemCleaningConfig(mode="global")  # or "suspicious_only"
    report = compare_early_exit(xyz, trunk_result, base_cfg, repeats=3)

From CLI (requires a pickled (xyz, trunk_result) checkpoint):

    python scripts/benchmark_voxel_early_exit.py \
        --checkpoint outputs/plot_X_trunkresult.pkl \
        --output outputs/early_exit_report.json \
        --repeats 3 \
        --mode global

To produce the checkpoint from your notebook once:

    import pickle
    with open("outputs/plot_X_trunkresult.pkl", "wb") as f:
        pickle.dump({"xyz": xyz, "trunk_result": trunk_result}, f)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.trunk_validation import StemCleaningConfig, clean_stems


def _run_once(xyz: np.ndarray, trunk_result, config: StemCleaningConfig) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = clean_stems(xyz, trunk_result, config, verbose=False)
    elapsed = time.perf_counter() - t0
    return {
        "wall_seconds": elapsed,
        "stem_mask": result.stem_mask.copy(),
        "n_points_after": int(result.n_points_after),
        "n_points_removed": int(result.n_points_removed),
        "n_points_processed_verticality": int(result.n_points_processed_verticality),
        "used_global_fallback": bool(result.used_global_fallback),
        "n_trees_processed": int(result.n_trees_processed),
        "n_trees_skipped": int(result.n_trees_skipped),
        "mode_used": result.mode_used,
    }


def _summarise_runs(label: str, runs: list[Dict[str, Any]]) -> Dict[str, Any]:
    times = np.array([r["wall_seconds"] for r in runs])
    # Sanity: all runs should produce identical masks for the same config
    first_mask = runs[0]["stem_mask"]
    for r in runs[1:]:
        if not np.array_equal(r["stem_mask"], first_mask):
            raise RuntimeError(f"{label}: runs produced different masks — non-deterministic clean_stems()")
    return {
        "label": label,
        "wall_seconds_mean": float(times.mean()),
        "wall_seconds_std": float(times.std()),
        "wall_seconds_min": float(times.min()),
        "n_points_after": runs[0]["n_points_after"],
        "n_points_removed": runs[0]["n_points_removed"],
        "n_points_processed_verticality": runs[0]["n_points_processed_verticality"],
        "used_global_fallback": runs[0]["used_global_fallback"],
        "n_trees_processed": runs[0]["n_trees_processed"],
        "n_trees_skipped": runs[0]["n_trees_skipped"],
        "mode_used": runs[0]["mode_used"],
        "mask": first_mask,
    }


def _compare_masks(baseline_mask: np.ndarray, ee_mask: np.ndarray) -> Dict[str, Any]:
    diff = baseline_mask != ee_mask
    n_total = int(baseline_mask.size)
    n_diff = int(diff.sum())
    # Lost: baseline kept but early-exit dropped → potential recall loss
    lost = int(((baseline_mask == True) & (ee_mask == False)).sum())
    # Gained: baseline dropped but early-exit kept → extra retention
    gained = int(((baseline_mask == False) & (ee_mask == True)).sum())
    return {
        "n_total_points": n_total,
        "n_mask_diff": n_diff,
        "mask_diff_pct": (100.0 * n_diff / n_total) if n_total else 0.0,
        "mask_diff_lost_baseline_yes_ee_no": lost,
        "mask_diff_gained_baseline_no_ee_yes": gained,
    }


def compare_early_exit(
    xyz: np.ndarray,
    trunk_result,
    base_config: StemCleaningConfig,
    repeats: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run clean_stems with and without voxel_early_exit, return a comparison report.

    Parameters
    ----------
    xyz : (N, 3) float64
    trunk_result : TrunkExtractionResult from the upstream pipeline
    base_config : StemCleaningConfig — used as-is for baseline; cloned with
                  voxel_early_exit=True for the second run.
    repeats : how many times to run each config (mean wall time is reported).
              Use >=3 to smooth noise; first run often slower due to warm-up.
    """
    # Force baseline to early_exit=False regardless of what the user passed
    cfg_baseline = replace(base_config, voxel_early_exit=False)
    cfg_early_exit = replace(base_config, voxel_early_exit=True)

    if verbose:
        print(f"\n=== Voxel-level early-exit benchmark ===")
        print(f"Mode:    {cfg_baseline.mode}")
        print(f"Points:  {len(xyz):,}")
        print(f"Trees:   {trunk_result.n_trees}")
        print(f"Repeats: {repeats}")
        print(f"Coarse resolution: {cfg_early_exit.voxel_early_exit_coarse_resolution} m")
        print(f"Margin:           {cfg_early_exit.voxel_early_exit_margin}")

    if verbose:
        print(f"\n[1/2] Baseline (voxel_early_exit=False)")
    baseline_runs = [_run_once(xyz, trunk_result, cfg_baseline) for _ in range(repeats)]
    baseline = _summarise_runs("baseline", baseline_runs)

    if verbose:
        print(f"      mean = {baseline['wall_seconds_mean']:.2f}s  "
              f"(min {baseline['wall_seconds_min']:.2f}s, std {baseline['wall_seconds_std']:.2f}s)")
        print(f"      points kept = {baseline['n_points_after']:,}, "
              f"verticality workload = {baseline['n_points_processed_verticality']:,}")

    if verbose:
        print(f"\n[2/2] Early-exit  (voxel_early_exit=True)")
    ee_runs = [_run_once(xyz, trunk_result, cfg_early_exit) for _ in range(repeats)]
    ee = _summarise_runs("early_exit", ee_runs)

    if verbose:
        print(f"      mean = {ee['wall_seconds_mean']:.2f}s  "
              f"(min {ee['wall_seconds_min']:.2f}s, std {ee['wall_seconds_std']:.2f}s)")
        print(f"      points kept = {ee['n_points_after']:,}, "
              f"verticality workload = {ee['n_points_processed_verticality']:,}")

    # --- comparison ---
    mask_cmp = _compare_masks(baseline["mask"], ee["mask"])
    speedup = baseline["wall_seconds_mean"] / ee["wall_seconds_mean"] if ee["wall_seconds_mean"] > 0 else float("nan")

    # workload reduction (independent of CPU noise)
    workload_baseline = baseline["n_points_processed_verticality"]
    workload_ee = ee["n_points_processed_verticality"]
    workload_reduction = (
        100.0 * (1 - workload_ee / workload_baseline) if workload_baseline > 0 else 0.0
    )

    decision_default_ok = (
        speedup >= 1.20
        and mask_cmp["mask_diff_pct"] < 0.5
        and mask_cmp["mask_diff_lost_baseline_yes_ee_no"] == 0
    )

    if verbose:
        print(f"\n=== Comparison ===")
        print(f"Speedup:                 {speedup:.2f}x")
        print(f"Workload reduction:      {workload_reduction:.1f}% "
              f"(verticality input points, baseline-vs-EE)")
        print(f"Mask diff:               {mask_cmp['n_mask_diff']:,} / "
              f"{mask_cmp['n_total_points']:,} points "
              f"({mask_cmp['mask_diff_pct']:.3f}%)")
        print(f"  baseline kept, EE dropped (recall loss): {mask_cmp['mask_diff_lost_baseline_yes_ee_no']:,}")
        print(f"  baseline dropped, EE kept (extra retain): {mask_cmp['mask_diff_gained_baseline_no_ee_yes']:,}")
        print(f"\nRecommendation: "
              f"{'PROMOTE TO DEFAULT' if decision_default_ok else 'KEEP OPT-IN'}")
        if not decision_default_ok:
            reasons = []
            if speedup < 1.20:
                reasons.append(f"speedup {speedup:.2f}x < 1.20x")
            if mask_cmp["mask_diff_pct"] >= 0.5:
                reasons.append(f"mask diff {mask_cmp['mask_diff_pct']:.3f}% >= 0.5%")
            if mask_cmp["mask_diff_lost_baseline_yes_ee_no"] > 0:
                reasons.append(
                    f"recall loss {mask_cmp['mask_diff_lost_baseline_yes_ee_no']} points "
                    f"(EE drops some that baseline keeps)"
                )
            print(f"  reason(s): {'; '.join(reasons)}")

    # Strip masks before returning (they can be huge for JSON)
    baseline_out = {k: v for k, v in baseline.items() if k != "mask"}
    ee_out = {k: v for k, v in ee.items() if k != "mask"}

    return {
        "input": {
            "n_points": int(len(xyz)),
            "n_trees": int(trunk_result.n_trees),
            "mode": cfg_baseline.mode,
            "repeats": int(repeats),
            "coarse_resolution": cfg_early_exit.voxel_early_exit_coarse_resolution,
            "margin": cfg_early_exit.voxel_early_exit_margin,
        },
        "baseline": baseline_out,
        "early_exit": ee_out,
        "comparison": {
            **mask_cmp,
            "speedup": speedup,
            "workload_reduction_pct": workload_reduction,
        },
        "recommendation": "promote_to_default" if decision_default_ok else "keep_opt_in",
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark voxel-level early-exit in clean_stems")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a pickle file with {'xyz': ndarray, 'trunk_result': TrunkExtractionResult}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/early_exit_report.json",
        help="JSON report output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["global", "suspicious_only"], default="global")
    parser.add_argument("--coarse-resolution", type=float, default=None,
                        help="Override voxel_early_exit_coarse_resolution (default 0.08)")
    parser.add_argument("--margin", type=float, default=None,
                        help="Override voxel_early_exit_margin (default 0.10)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    xyz = ckpt["xyz"]
    trunk_result = ckpt["trunk_result"]

    cfg = StemCleaningConfig(mode=args.mode)
    if args.coarse_resolution is not None:
        cfg = replace(cfg, voxel_early_exit_coarse_resolution=args.coarse_resolution)
    if args.margin is not None:
        cfg = replace(cfg, voxel_early_exit_margin=args.margin)

    report = compare_early_exit(xyz, trunk_result, cfg, repeats=args.repeats, verbose=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
