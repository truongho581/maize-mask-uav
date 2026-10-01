#!/usr/bin/env python3
"""Run the frozen SegFormer-B0 random-tile versus spatial-guard control."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 123, 3407)
ARMS = (
    ("maizemask-v1.0-spatial-guard", "standard"),
    ("maizemask-v1.0-random-tile-control", "random_tile_control"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("dataset/maizemask_v1.0_public_release"),
        help="Extracted MaizeMask v1.0 release containing the frozen manifests.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(".runtime_data"),
        help="Root produced by scripts/prepare_data_splits.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/paper_semantic_maizemask_v1_split_leakage_control"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not train; validate and aggregate six existing completed runs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run only when best.pth and test_metrics.json both exist.",
    )
    return parser.parse_args()


def run_directory(output: Path, dataset_key: str, seed: int) -> Path:
    return (
        output / "semantic" / "segformer_b0" / dataset_key /
        "fixed" / "fixed" / f"seed_{seed}"
    )


def main() -> None:
    args = parse_args()
    release = args.release_root.resolve()
    datasets = args.dataset_root.resolve()
    output = args.output_root.resolve()

    if not release.is_dir():
        raise FileNotFoundError(f"Missing release root: {release}")

    if not args.aggregate_only:
        for _, directory in ARMS:
            root = datasets / directory
            if not root.is_dir():
                raise FileNotFoundError(
                    f"Missing prepared split tree: {root}. "
                    "Run prepare_data_splits.py first."
                )
        for dataset_key, directory in ARMS:
            for seed in SEEDS:
                run_dir = run_directory(output, dataset_key, seed)
                complete = (
                    (run_dir / "best.pth").is_file()
                    and (run_dir / "test_metrics.json").is_file()
                )
                if complete and args.skip_existing:
                    print(f"Skipping completed run: {run_dir}")
                    continue
                if run_dir.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite existing run: {run_dir}. "
                        "Use a fresh output root or --skip-existing for complete runs."
                    )
                command = [
                    sys.executable,
                    "scripts/train/train_semantic.py",
                    "--dataset-root", str(datasets / directory),
                    "--dataset-key", dataset_key,
                    "--protocol", "fixed",
                    "--folds", "fixed",
                    "--model", "segformer_b0",
                    "--pretrained-backbone",
                    "--epochs", str(args.epochs),
                    "--batch-size", "2",
                    "--num-workers", "0",
                    "--seed", str(seed),
                    "--img-size", "640",
                    "--task-mode", "s3",
                    "--augment-profile", "basic",
                    "--color-profile", "none",
                    "--scheduler", "cosine",
                    "--lr", "0.00006",
                    "--weight-decay", "0.01",
                    "--reproducibility-mode", "stable_hash",
                    "--disable-amp",
                    "--skip-initial-preview",
                    "--output-root", str(output),
                ]
                print("Running:", " ".join(command))
                subprocess.run(command, check=True)

    manifests = release / "splits"
    command = [
        sys.executable,
        "scripts/evaluate/aggregate_split_leakage_control.py",
        "--runs-root", str(output),
        "--output", str(output / "summary"),
        "--seeds", *map(str, SEEDS),
        "--spatial-manifest", str(manifests / "standard_spatial_guard_split.csv"),
        "--random-manifest", str(manifests / "random_tile_control_seed42.csv"),
        "--random-audit", str(manifests / "random_tile_control_audit.json"),
    ]
    subprocess.run(command, check=True)
    print((output / "summary" / "SPLIT_LEAKAGE_CONTROL_REPORT.md").read_text())


if __name__ == "__main__":
    main()
