#!/usr/bin/env python3
"""Run the complete 50-epoch MaizeMask v1.0 paper training protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MODELS_SCRATCH = ("attention_unet", "deeplabv3plus_resnet50", "segformer_b0")
FOLDS = ("test_D1", "test_D2", "test_D3")
SEEDS = (42, 123, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--dataset-root", type=Path, required=True, help="Prepared LOSO tree from prepare_data_splits.py")
    parser.add_argument("--cache-root", type=Path, default=root / ".cache")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_json(path: Path) -> None:
    def walk(value, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite value at {location} in {path}")
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")

    walk(json.loads(path.read_text(encoding="utf-8")), "root")


def execute(case: dict, log_path: Path, env: dict[str, str]) -> tuple[int, str | None]:
    command = case["command"]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=case["cwd"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        return return_code, f"process exited with {return_code}"
    try:
        for path in case["expected"]:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing expected artifact: {path}")
            if path.suffix == ".json":
                finite_json(path)
    except Exception as exc:
        return 90, str(exc)
    return 0, None


def semantic_case(root: Path, data: Path, output: Path, model: str, seed: int, fold: str, epochs: int, pretrained: bool) -> dict:
    case_id = f"semantic_{model}_{'pretrained' if pretrained else 'scratch'}_{fold}_seed{seed}"
    # The training script's native directory encodes model/dataset/fold/seed but
    # not backbone initialization.  Keep scratch and pretrained SegFormer runs
    # in separate roots so the full protocol cannot silently collide or require
    # --overwrite.
    init_root = output / ("pretrained" if pretrained else "scratch")
    run_dir = init_root / "semantic" / model / "v1.0" / "loso" / fold / f"seed_{seed}"
    command = [
        sys.executable,
        "-u",
        "scripts/train/train_semantic.py",
        "--dataset-root", str(data),
        "--dataset-key", "v1.0",
        "--protocol", "loso",
        "--folds", fold,
        "--model", model,
        "--epochs", str(epochs),
        "--batch-size", "2",
        "--num-workers", "0",
        "--seed", str(seed),
        "--img-size", "640",
        "--task-mode", "s3",
        "--augment-profile", "basic",
        "--color-profile", "none",
        "--scheduler", "cosine",
        "--lr", "0.00006" if pretrained else "0.001",
        "--weight-decay", "0.01" if pretrained else "0.0",
        "--reproducibility-mode", "stable_hash",
        "--disable-amp",
        "--skip-initial-preview",
        "--log-interval", "20",
        "--output-root", str(init_root),
    ]
    if pretrained:
        command.append("--pretrained-backbone")
    if model == "deeplabv3plus_resnet50":
        # D1 has 175 training images. Native BatchNorm reaches a 1x1 feature map,
        # so the final singleton batch cannot be trained.
        command.append("--drop-last-train")
    return {
        "id": case_id,
        "cwd": root,
        "command": command,
        "expected": [run_dir / "best.pth", run_dir / "last.pth", run_dir / "test_metrics.json"],
    }


def maskrcnn_case(root: Path, data: Path, output: Path, fold: str, epochs: int) -> dict:
    run_dir = output / "instance/maskrcnn_resnet50_fpn/v1.0/loso" / fold / "seed_42"
    return {
        "id": f"instance_maskrcnn_resnet50_fpn_{fold}_seed42",
        "cwd": root,
        "command": [
            sys.executable, "-u", "scripts/train/train_maskrcnn.py",
            "--dataset-root", str(data), "--dataset-key", "v1.0",
            "--protocol", "loso", "--folds", fold,
            "--model", "maskrcnn_resnet50_fpn", "--epochs", str(epochs),
            "--batch-size", "2", "--num-workers", "0", "--img-size", "640", "--seed", "42",
            "--output-root", str(output), "--log-interval", "20",
            "--lr", "0.0005", "--weight-decay", "0.0001", "--scheduler", "cosine",
            "--augment-profile", "horizontal_flip", "--pretrained",
        ],
        "expected": [run_dir / "best.pth", run_dir / "last.pth", run_dir / "test_metrics.json"],
    }


def mask2former_cases(root: Path, data: Path, output: Path, fold: str, epochs: int) -> list[dict]:
    run_dir = output / "instance/mask2former_r50/v1.0/loso" / fold / "seed_42"
    base = [
        sys.executable, "-u", "scripts/train/train_mask2former.py",
        "--dataset-root", str(data), "--dataset-key", "v1.0",
        "--protocol", "loso", "--folds", fold, "--epochs", str(epochs),
        "--batch-size", "2", "--num-workers", "0", "--img-size", "640", "--seed", "42",
        "--output-root", str(output), "--lr", "0.0001", "--weight-decay", "0.05",
        "--augment-profile", "horizontal_flip", "--pretrained",
        "--checkpoint-period-epochs", str(epochs),
    ]
    train = {
        "id": f"instance_mask2former_r50_{fold}_seed42_train",
        "cwd": root,
        "command": base,
        "expected": [run_dir / "model_best.pth", run_dir / "best_validation.json", run_dir / "model_final.pth"],
    }
    evaluate = {
        "id": f"instance_mask2former_r50_{fold}_seed42_reload_eval",
        "cwd": root,
        "command": base + ["--evaluate-checkpoint", str(run_dir / "model_best.pth"), "--evaluation-split", "test"],
        "expected": [run_dir / "post_eval/test/metrics.json"],
    }
    return [train, evaluate]


def build_cases(root: Path, data: Path, output: Path) -> list[dict]:
    cases = []
    for model in MODELS_SCRATCH:
        for seed in SEEDS:
            for fold in FOLDS:
                cases.append(semantic_case(root, data, output, model, seed, fold, 50, False))
    for seed in SEEDS:
        for fold in FOLDS:
            cases.append(semantic_case(root, data, output, "segformer_b0", seed, fold, 50, True))
    for fold in FOLDS:
        cases.append(maskrcnn_case(root, data, output, fold, 50))
    for fold in FOLDS:
        cases.extend(mask2former_cases(root, data, output, fold, 50))
    return cases


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_root or root / "results" / f"maizemask_v1_paper_{timestamp}").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output root: {output}")
    output.mkdir(parents=True)
    logs = output / "runner_logs"
    logs.mkdir()

    cache = args.cache_root.resolve()
    env = os.environ.copy()
    env.update({
        "TORCH_HOME": str(cache / "torch"),
        "MAIZEMASK_SEGFORMER_B0_SOURCE": str(cache / "weights/nvidia_mit_b0"),
        "MAIZEMASK_SEGFORMER_B0_REVISION": "80983a413c30d36a39c20203974ae7807835e2b4",
        "MAIZEMASK_MASK2FORMER_R50_WEIGHTS": str(cache / "weights/mask2former_r50_model_final_3c8ec9.pkl"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })

    cases = build_cases(root, args.dataset_root.resolve(), output)
    report = {
        "protocol": "complete 50-epoch paper protocol",
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "output_root": str(output),
        "cases": [],
    }
    report_path = output / "paper_training_report.json"
    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] {case['id']}\n{'=' * 80}", flush=True)
        item = {
            "id": case["id"],
            "started_at_utc": utc_now(),
            "command": case["command"],
            "expected": [str(path) for path in case["expected"]],
        }
        code, error = execute(case, logs / f"{index:02d}_{case['id']}.log", env)
        item.update({"finished_at_utc": utc_now(), "exit_code": code, "status": "PASS" if code == 0 else "FAIL"})
        if error:
            item["error"] = error
        report["cases"].append(item)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    failures = [item for item in report["cases"] if item["status"] != "PASS"]
    report["finished_at_utc"] = utc_now()
    report["passed"] = len(report["cases"]) - len(failures)
    report["failed"] = len(failures)
    report["status"] = "PASS" if not failures else "FAIL"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("protocol", "status", "passed", "failed", "output_root")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
