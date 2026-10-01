#!/usr/bin/env python3
"""Aggregate the MaizeMask random-tile versus spatial-guard control experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROTOCOLS = {
    "spatial_guard": "maizemask-v1.0-spatial-guard",
    "random_tile": "maizemask-v1.0-random-tile-control",
}
METRICS = ("target_miou", "all_miou", "crop_iou", "weed_iou", "target_mdice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--spatial-manifest", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--random-audit", type=Path, required=True)
    return parser.parse_args()


def metric_row(protocol: str, dataset_key: str, seed: int, path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "protocol": protocol,
        "dataset_key": dataset_key,
        "seed": seed,
        "checkpoint_epoch": payload["checkpoint_epoch"],
        "target_miou": payload["test_target_miou"] * 100.0,
        "all_miou": payload["test_miou_all"] * 100.0,
        "crop_iou": payload["ious"]["crop"] * 100.0,
        "weed_iou": payload["ious"]["weed"] * 100.0,
        "target_mdice": payload["test_target_mdice"] * 100.0,
        "path": str(path),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row["split"] for row in rows))


def site_split_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        output[split] = dict(
            sorted(Counter(row["field_id"] for row in rows if row["split"] == split).items())
        )
    return output


def crossing_counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    partitions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        partitions[row[column]].add(row["split"])
    return {
        "total": len(partitions),
        "crossing_count": sum(len(splits) > 1 for splits in partitions.values()),
    }


def load_config(metric_path: Path) -> dict[str, object]:
    path = metric_path.with_name("config.json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def training_contract(config: dict[str, object]) -> dict[str, object]:
    data = config["data"]
    training = dict(config["training"])
    training.pop("seed", None)
    runtime = dict(config["runtime"])
    runtime.pop("gpu", None)
    return {
        "task": config["task"],
        "model": config["model"],
        "image_size": data["img_size"],
        "split_image_counts": {
            split: data["splits"][split]["images"] for split in ("train", "valid", "test")
        },
        "training_except_seed": training,
        "augmentation": config["augmentation"],
        "input_preprocessing": config["input_preprocessing"],
        "reproducibility_except_seed": config["reproducibility"],
        "checkpoint_selection": config["checkpoint_selection"],
        "runtime_except_gpu_name": runtime,
    }


def verify_training_contracts(runs_root: Path, seeds: list[int]) -> str:
    contracts: list[dict[str, object]] = []
    labels: list[str] = []
    for protocol, dataset_key in PROTOCOLS.items():
        for seed in seeds:
            metric_path = (
                runs_root / "semantic" / "segformer_b0" / dataset_key /
                "fixed" / "fixed" / f"seed_{seed}" / "test_metrics.json"
            )
            config = load_config(metric_path)
            if int(config["training"]["seed"]) != seed:
                raise ValueError(f"Seed mismatch in {metric_path}")
            contracts.append(training_contract(config))
            labels.append(f"{protocol} seed {seed}")
    canonical = json.dumps(contracts[0], sort_keys=True, separators=(",", ":"))
    for label, contract in zip(labels[1:], contracts[1:]):
        if json.dumps(contract, sort_keys=True, separators=(",", ":")) != canonical:
            raise ValueError(f"Training contract differs at {label}.")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_split_protocol(
    spatial_manifest: Path, random_manifest: Path, random_audit_path: Path
) -> dict[str, object]:
    spatial_rows = read_csv(spatial_manifest)
    random_rows = read_csv(random_manifest)
    spatial_counts = split_counts(spatial_rows)
    if spatial_counts != split_counts(random_rows):
        raise ValueError("Split image counts differ between protocols.")
    spatial_sites = site_split_counts(spatial_rows)
    if spatial_sites != site_split_counts(random_rows):
        raise ValueError("Site-by-split counts differ between protocols.")
    if {row["public_image_id"] for row in spatial_rows} != {
        row["public_image_id"] for row in random_rows
    }:
        raise ValueError("The protocols do not contain the same tile identities.")

    spatial_source = crossing_counts(spatial_rows, "source_group_id")
    spatial_cluster = crossing_counts(spatial_rows, "split_group_id")
    if spatial_source["crossing_count"] or spatial_cluster["crossing_count"]:
        raise ValueError("Spatial guard has a correlation group crossing partitions.")

    random_audit = json.loads(random_audit_path.read_text(encoding="utf-8"))
    random_control = random_audit["random_tile_control"]
    if random_control["source_group"]["crossing_count"] <= 0:
        raise ValueError("Random control has no source-image crossing.")
    if random_control["spatial_group"]["crossing_count"] <= 0:
        raise ValueError("Random control has no GPS-component crossing.")
    return {
        "same_tile_set": True,
        "same_global_split_counts": spatial_counts,
        "same_site_by_split_counts": spatial_sites,
        "spatial_guard": {
            "source_id": spatial_source,
            "overlap_cluster": spatial_cluster,
        },
        "random_tile": {
            "seed": random_audit["assignment_seed"],
            "site_count_policy": random_audit["field_count_policy"],
            "source_id": {
                "total": random_control["source_group"]["total"],
                "crossing_count": random_control["source_group"]["crossing_count"],
            },
            "overlap_cluster": {
                "total": random_control["spatial_group"]["total"],
                "crossing_count": random_control["spatial_group"]["crossing_count"],
            },
        },
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        selected = [row for row in rows if row["protocol"] == protocol]
        summary: dict[str, object] = {"protocol": protocol, "seeds": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean_pct"] = statistics.mean(values)
            summary[f"{metric}_sample_sd_pct"] = statistics.stdev(values)
        output.append(summary)
    return output


def paired_deltas(rows: list[dict[str, object]], seeds: list[int]) -> list[dict[str, object]]:
    indexed = {(str(row["protocol"]), int(row["seed"])): row for row in rows}
    output: list[dict[str, object]] = []
    for seed in seeds:
        guarded = indexed[("spatial_guard", seed)]
        random_tile = indexed[("random_tile", seed)]
        output.append({
            "seed": seed,
            **{
                f"{metric}_random_minus_guard_pp": (
                    float(random_tile[metric]) - float(guarded[metric])
                )
                for metric in METRICS
            },
        })
    return output


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    deltas: list[dict[str, object]],
    integrity: dict[str, object],
) -> None:
    summary = {str(row["protocol"]): row for row in summaries}
    guarded = summary["spatial_guard"]
    random_tile = summary["random_tile"]
    target_deltas = [float(row["target_miou_random_minus_guard_pp"]) for row in deltas]
    index = {(str(row["protocol"]), int(row["seed"])): row for row in rows}
    lines = [
        "# MaizeMask semantic split leakage control",
        "",
        "Both arms use the same pretrained SegFormer-B0 recipe and seeds. The",
        "spatial guard keeps source/GPS-connected tiles together; the random-tile",
        "control intentionally permits correlated views to cross partitions.",
        "",
        "| Protocol | Target mIoU mean (%) | Sample SD (pp) |",
        "|---|---:|---:|",
        f"| Spatial guard | {float(guarded['target_miou_mean_pct']):.2f} | {float(guarded['target_miou_sample_sd_pct']):.2f} |",
        f"| Random tile | {float(random_tile['target_miou_mean_pct']):.2f} | {float(random_tile['target_miou_sample_sd_pct']):.2f} |",
        "",
        f"Mean seed-matched random-minus-guard difference: {statistics.mean(target_deltas):+.2f} pp (sample SD {statistics.stdev(target_deltas):.2f} pp).",
        "",
        f"Training-contract SHA-256: `{integrity['training_contract_sha256']}`.",
        "",
        "This control is descriptive and does not replace unseen-field LOSO evaluation.",
        "",
        "| Seed | Spatial guard (%) | Random tile (%) | Difference (pp) |",
        "|---:|---:|---:|---:|",
    ]
    for delta in deltas:
        seed = int(delta["seed"])
        lines.append(
            f"| {seed} | {float(index[('spatial_guard', seed)]['target_miou']):.2f} | "
            f"{float(index[('random_tile', seed)]['target_miou']):.2f} | "
            f"{float(delta['target_miou_random_minus_guard_pp']):+.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    split_protocol = verify_split_protocol(
        args.spatial_manifest, args.random_manifest, args.random_audit
    )
    integrity = {
        "status": "passed",
        "seeds": args.seeds,
        "training_contract_sha256": verify_training_contracts(args.runs_root, args.seeds),
        "split_protocol": split_protocol,
    }
    rows: list[dict[str, object]] = []
    for protocol, dataset_key in PROTOCOLS.items():
        for seed in args.seeds:
            metric_path = (
                args.runs_root / "semantic" / "segformer_b0" / dataset_key /
                "fixed" / "fixed" / f"seed_{seed}" / "test_metrics.json"
            )
            rows.append(metric_row(protocol, dataset_key, seed, metric_path))

    summaries = summarize(rows)
    deltas = paired_deltas(rows, args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "split_leakage_seed_metrics.csv", rows)
    write_csv(args.output / "split_leakage_summary.csv", summaries)
    write_csv(args.output / "split_leakage_paired_deltas.csv", deltas)
    (args.output / "experiment_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        args.output / "SPLIT_LEAKAGE_CONTROL_REPORT.md",
        rows,
        summaries,
        deltas,
        integrity,
    )
    print(f"Wrote split-control summary to {args.output}")


if __name__ == "__main__":
    main()
