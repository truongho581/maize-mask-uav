from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Any

# Must be defined before importing torch for deterministic CUDA matrix products.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_CODE = PROJECT_ROOT / "src"
if str(SRC_CODE) not in sys.path:
    sys.path.insert(0, str(SRC_CODE))

from training.run_artifacts import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    atomic_torch_save,
    prepare_run_directory,
    protocol_datasets,
    resolve_path,
    resolve_protocol_dataset,
    run_directory,
    runtime_metadata,
    write_command,
    write_json,
    write_run_config,
)
from training.semantic_segmentation import (  # noqa: E402
    build_dataloaders,
    build_semantic_model,
    count_parameters,
    fingerprint_coco_splits,
    run_one_epoch,
    save_sample_predictions,
    seed_everything,
    select_device,
    set_training_epoch,
)
from datasets.coco_semantic import augmentation_profile_metadata, input_feature_channels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train semantic segmentation models on MaizeMask LOSO or fixed splits."
    )
    parser.add_argument("--protocol", choices=["loso", "fixed"], default="loso")
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Optional registered dataset label; public v1.0 uses --dataset-root.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Explicit custom dataset root; cannot be combined with --dataset-version.",
    )
    parser.add_argument(
        "--dataset-key",
        default=None,
        help=(
            "Stable dataset label stored in run paths/configs when using --dataset-root. "
            "For example: v1.0."
        ),
    )
    parser.add_argument(
        "--loso-root",
        type=Path,
        default=None,
        help="Deprecated alias for --dataset-root, retained for old RunPod commands.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model",
        choices=["attention_unet", "deeplabv3plus_resnet50", "segformer_b0"],
        default="attention_unet",
    )
    parser.add_argument("--folds", nargs="+", default=None)
    parser.add_argument("--task-mode", choices=["s3", "raw"], default="s3")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stopping patience. Zero disables early stopping.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print train-batch progress every N batches; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reproducibility-mode",
        choices=["legacy", "stable_hash", "strict"],
        default="legacy",
        help=(
            "legacy preserves historical runs. stable_hash fixes augmentation to tile+epoch, "
            "uses a sample-ID based epoch order, and keeps incomplete final batches. strict "
            "also disables AMP/Flash Attention, uses no loader workers, and fails on known "
            "nondeterministic CUDA operations."
        ),
    )
    parser.add_argument(
        "--drop-last-train",
        action="store_true",
        help=(
            "Drop an incomplete final training batch. Required for BatchNorm models when "
            "the final batch has one image; validation and test always retain all images."
        ),
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help=(
            "Disable CUDA automatic mixed precision. Useful with stable_hash when "
            "measuring residual run-to-run variation on hardware where strict CUDA "
            "determinism is unavailable."
        ),
    )
    parser.add_argument("--norm", choices=["bn", "gn"], default="gn")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument(
        "--color-profile",
        choices=["none", "dji-jpg-mean-std"],
        default="none",
        help="Use none for materialized color datasets; use dji-jpg-mean-std for raw images.",
    )
    parser.add_argument(
        "--augment-profile",
        choices=["basic", "none"],
        default="basic",
    )
    parser.add_argument(
        "--input-features",
        choices=["rgb", "rgb_exg", "rgb_exg_vari"],
        default="rgb",
        help=(
            "Model input representation. Derived channels preserve canonical RGB and are "
            "computed after augmentation. Non-RGB profiles require scratch SegFormer-B0."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-optimizer-state",
        action="store_true",
        help="Include optimizer/scheduler state in last.pth for resuming; uses substantially more disk.",
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--skip-initial-preview",
        action="store_true",
        help=(
            "Skip the untrained train-set prediction preview created before epoch 1. "
            "This preview is diagnostic only and does not affect training or evaluation."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def target_indices(class_names: list[str]) -> list[int]:
    selected = [index for index, name in enumerate(class_names) if name in {"crop", "weed"}]
    if len(selected) == 2:
        return selected
    selected = [index for index, name in enumerate(class_names) if name != "background"]
    if not selected:
        raise ValueError(f"No foreground classes found in {class_names}")
    return selected


def target_mean(values: np.ndarray, indices: list[int]) -> float:
    return float(np.asarray(values)[indices].mean())


def is_better(
    target_miou: float,
    target_mdice: float,
    best_target_miou: float,
    best_target_mdice: float,
    tolerance: float = 1e-12,
) -> bool:
    if target_miou > best_target_miou + tolerance:
        return True
    return (
        abs(target_miou - best_target_miou) <= tolerance
        and target_mdice > best_target_mdice + tolerance
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_target_miou: float,
    best_target_mdice: float,
    config: dict[str, Any],
    include_optimizer: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_target_miou": best_target_miou,
        "best_target_mdice": best_target_mdice,
        "checkpoint_criterion": config["checkpoint_selection"],
        "config": config,
        # Compatibility with older evaluators.
        "model": config["model"]["name"],
        "task_mode": config["task"]["mode"],
        "class_names": config["task"]["class_names"],
        "category_to_mask": config["task"]["category_to_mask"],
        "dataset_root": config["data"]["dataset_root"],
        "img_size": config["data"]["img_size"],
        "norm": config["model"]["norm"],
        "pretrained_backbone": config["model"]["pretrained_backbone"],
        "color_profile": config["augmentation"]["color_profile"],
        "input_features": config["model"]["input_features"],
        "input_channels": config["model"]["input_channels"],
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def history_fields(class_names: list[str]) -> list[str]:
    return [
        "epoch",
        "learning_rate",
        "train_loss",
        "valid_loss",
        "valid_miou_all",
        "valid_mdice_all",
        "valid_target_miou",
        "valid_target_mdice",
        *[f"valid_iou_{name}" for name in class_names],
        *[f"valid_dice_{name}" for name in class_names],
        "seconds",
        "is_best",
    ]


def train_fold(
    args: argparse.Namespace,
    dataset_root: Path,
    output_root: Path,
    fold: str,
) -> None:
    seed_everything(
        args.seed,
        deterministic=True,
        strict_cuda=(args.reproducibility_mode == "strict"),
    )
    device = select_device(args.device)
    output_dir = run_directory(
        output_root,
        task="semantic",
        model=args.model,
        dataset=args.dataset_key,
        protocol=args.protocol,
        fold=fold,
        seed=args.seed,
    )
    if args.skip_existing and (output_dir / "test_metrics.json").exists() and (output_dir / "best.pth").exists():
        print(f"Skip existing run: {output_dir}")
        return
    prepare_run_directory(output_dir, overwrite=args.overwrite)

    input_channels = input_feature_channels(args.input_features)
    if args.input_features != "rgb":
        if args.model != "segformer_b0":
            raise ValueError(
                "Derived vegetation-index channels are currently registered only for "
                "SegFormer-B0. Use --model segformer_b0."
            )
        if args.pretrained_backbone:
            raise ValueError(
                "Derived vegetation-index channels require scratch SegFormer-B0; "
                "remove --pretrained-backbone."
            )

    print("[phase] Building datasets and data loaders...", flush=True)
    coco_by_split, class_info, loaders = build_dataloaders(
        dataset_root=dataset_root,
        task_mode=args.task_mode,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        color_profile=args.color_profile,
        augment_profile=args.augment_profile,
        input_features=args.input_features,
        device=device,
        seed=args.seed,
        reproducibility_mode=args.reproducibility_mode,
        drop_last_train=args.drop_last_train,
    )
    data_fingerprints = fingerprint_coco_splits(dataset_root, coco_by_split)
    class_names = class_info["class_names"]
    selected_indices = target_indices(class_names)
    selected_names = [class_names[index] for index in selected_indices]
    num_classes = len(class_names)

    print("[phase] Building semantic model...", flush=True)
    model = build_semantic_model(
        args.model,
        num_classes=num_classes,
        norm=args.norm,
        pretrained_backbone=args.pretrained_backbone,
        input_channels=input_channels,
    ).to(device)
    use_amp = args.reproducibility_mode != "strict" and not args.disable_amp
    normalization = {
        "attention_unet": f"{args.norm.upper()} normalization",
        "deeplabv3plus_resnet50": "native BatchNorm2d",
        "segformer_b0": "native SegFormer LayerNorm",
    }[args.model]

    config: dict[str, Any] = {
        "schema_version": 4,
        "task": {
            "type": "semantic_segmentation",
            "mode": args.task_mode,
            "class_names": class_names,
            "target_classes": selected_names,
            "category_to_mask": class_info["category_to_mask"],
        },
        "model": {
            "name": args.model,
            "norm": args.norm,
            "normalization": normalization,
            "pretrained_backbone": args.pretrained_backbone,
            "input_features": args.input_features,
            "input_channels": input_channels,
            "trainable_parameters": count_parameters(model),
        },
        "data": {
            "dataset_root": str(dataset_root),
            "dataset_version": args.dataset_version or "custom",
            "dataset_key": args.dataset_key,
            "protocol": args.protocol,
            "fold": fold,
            "img_size": args.img_size,
            "splits": {
                split: {
                    "images": len(coco["images"]),
                    "annotations": len(coco["annotations"]),
                }
                for split, coco in coco_by_split.items()
            },
            "fingerprints": data_fingerprints,
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "loss": "CrossEntropyLoss",
            "loss_details": {
                "class_weights": None,
                "ignore_index": -100,
            },
            "seed": args.seed,
            "deterministic": True,
            "early_stopping_patience": args.patience,
            "num_workers": args.num_workers,
            "save_optimizer_state": args.save_optimizer_state,
            "reproducibility_mode": args.reproducibility_mode,
            "train_sampling": (
                "stable_hash_epoch_sampler"
                if args.reproducibility_mode in {"stable_hash", "strict"}
                else "random_shuffle"
            ),
            "drop_last": args.reproducibility_mode == "legacy" or args.drop_last_train,
            "mixed_precision": use_amp,
        },
        "augmentation": {
            "profile": args.augment_profile,
            "definition": augmentation_profile_metadata(args.augment_profile),
            "color_profile": args.color_profile,
            "derived_channels": (
                "none"
                if args.input_features == "rgb"
                else args.input_features.removeprefix("rgb_")
            ),
            "train_only": True,
        },
        "input_preprocessing": {
            "resize": {
                "size_px": [args.img_size, args.img_size],
                "image_interpolation": "bilinear",
                "mask_interpolation": "nearest",
            },
            "rgb_scaling": "uint8 divided by 255 to [0, 1]",
            "mean_std_normalization": None,
        },
        "reproducibility": {
            "mode": args.reproducibility_mode,
            "train_sampler": (
                "stable_hash_epoch_sampler"
                if args.reproducibility_mode in {"stable_hash", "strict"}
                else "random_shuffle"
            ),
            "augmentation_rng": (
                "blake2b(seed, epoch, file_name)"
                if args.reproducibility_mode in {"stable_hash", "strict"}
                else "global_python_random"
            ),
            "drop_last": args.reproducibility_mode == "legacy" or args.drop_last_train,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": (
                "error_on_nondeterministic_operation"
                if args.reproducibility_mode == "strict"
                else "warn_only"
            ),
            "tf32_disabled": True,
            "amp_enabled": use_amp,
            "loader_workers_effective": (
                0 if args.reproducibility_mode == "strict" else args.num_workers
            ),
        },
        "checkpoint_selection": {
            "primary": "valid_target_miou",
            "definition": "mean(IoU_crop, IoU_weed)",
            "tie_break": "valid_target_mdice",
            "background_used_for_selection": False,
        },
        "runtime": runtime_metadata(device),
    }
    write_run_config(output_dir, config)
    write_command(output_dir)

    print("\n" + "=" * 80)
    print(f"Model: {args.model} | Protocol: {args.protocol} | Fold: {fold} | Seed: {args.seed}")
    print("Dataset:", dataset_root)
    print("Output:", output_dir)
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(device.index or 0))
    print("Class names:", class_names)
    print(f"Input features: {args.input_features} ({input_channels} channels)")
    print("Checkpoint target classes:", selected_names)
    print(
        "Splits:",
        {
            split: (len(coco["images"]), len(coco["annotations"]))
            for split, coco in coco_by_split.items()
        },
    )
    print(f"Trainable params: {count_parameters(model) / 1e6:.2f}M")
    print(f"Reproducibility mode: {args.reproducibility_mode}")

    if args.skip_initial_preview:
        print("[phase] Skipping untrained preview (--skip-initial-preview).", flush=True)
    else:
        print("[phase] Writing untrained train preview...", flush=True)
        save_sample_predictions(
            model,
            loaders["train"],
            device,
            output_dir / "predictions" / "train_mask_preview",
            args.model,
        )
        print("[phase] Untrained train preview complete.", flush=True)
    if args.preview_only:
        print("Preview saved. Stop because --preview-only was set.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    history_path = output_dir / "history.csv"
    fields = history_fields(class_names)
    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()

    best_target_miou = -1.0
    best_target_mdice = -1.0
    epochs_without_improvement = 0
    final_epoch = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        set_training_epoch(loaders["train"], epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        print(f"[phase] Epoch {epoch:03d}/{args.epochs}: train...", flush=True)
        train_loss, _, _, _, train_seconds = run_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            num_classes,
            train=True,
            mixed_precision=use_amp,
            log_interval=args.log_interval,
            phase=f"Epoch {epoch:03d}/{args.epochs} train",
        )
        valid_loss, valid_ious, valid_dices, valid_miou, valid_seconds = run_one_epoch(
            model,
            loaders["valid"],
            criterion,
            None,
            scaler,
            device,
            num_classes,
            train=False,
            mixed_precision=use_amp,
            phase=f"Epoch {epoch:03d}/{args.epochs} valid",
        )
        valid_mdice = float(valid_dices.mean())
        valid_target_miou = target_mean(valid_ious, selected_indices)
        valid_target_mdice = target_mean(valid_dices, selected_indices)
        improved = is_better(
            valid_target_miou,
            valid_target_mdice,
            best_target_miou,
            best_target_mdice,
        )
        if improved:
            best_target_miou = valid_target_miou
            best_target_mdice = valid_target_mdice
            epochs_without_improvement = 0
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_target_miou,
                    best_target_mdice,
                    config,
                    include_optimizer=False,
                ),
                output_dir / "best.pth",
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_miou_all": valid_miou,
            "valid_mdice_all": valid_mdice,
            "valid_target_miou": valid_target_miou,
            "valid_target_mdice": valid_target_mdice,
            "seconds": train_seconds + valid_seconds,
            "is_best": int(improved),
        }
        row.update({f"valid_iou_{name}": float(value) for name, value in zip(class_names, valid_ious)})
        row.update({f"valid_dice_{name}": float(value) for name, value in zip(class_names, valid_dices)})
        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)

        iou_text = ", ".join(
            f"{name}: {iou * 100:.2f}" for name, iou in zip(class_names, valid_ious)
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | target_mIoU={valid_target_miou * 100:.2f}% | "
            f"all_mIoU={valid_miou * 100:.2f}% | {iou_text} | "
            f"best_target={best_target_miou * 100:.2f}%"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break

    atomic_torch_save(
        checkpoint_payload(
            model,
            optimizer,
            scheduler,
            final_epoch,
            best_target_miou,
            best_target_mdice,
            config,
            include_optimizer=args.save_optimizer_state,
        ),
        output_dir / "last.pth",
    )

    checkpoint = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_ious, test_dices, test_miou, test_seconds = run_one_epoch(
        model,
        loaders["test"],
        criterion,
        None,
        scaler,
        device,
        num_classes,
        train=False,
        mixed_precision=use_amp,
        phase="Test",
    )
    test_mdice = float(test_dices.mean())
    test_target_miou = target_mean(test_ious, selected_indices)
    test_target_mdice = target_mean(test_dices, selected_indices)
    test_metrics = {
        "checkpoint": "best.pth",
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_selection": config["checkpoint_selection"],
        "test_loss": test_loss,
        "test_miou_all": test_miou,
        "test_mdice_all": test_mdice,
        "test_target_miou": test_target_miou,
        "test_target_mdice": test_target_mdice,
        "target_classes": selected_names,
        "test_seconds": test_seconds,
        "ious": {name: float(iou) for name, iou in zip(class_names, test_ious)},
        "dice": {name: float(dice) for name, dice in zip(class_names, test_dices)},
        # Compatibility with existing aggregation scripts.
        "test_miou": test_miou,
        "test_mdice": test_mdice,
    }
    write_json(output_dir / "test_metrics.json", test_metrics)

    summary_fields = [
        "split",
        "loss",
        "miou_all",
        "mdice_all",
        "target_miou",
        "target_mdice",
        *[f"iou_{name}" for name in class_names],
        *[f"dice_{name}" for name in class_names],
        "seconds",
    ]
    summary_row = {
        "split": "test",
        "loss": test_loss,
        "miou_all": test_miou,
        "mdice_all": test_mdice,
        "target_miou": test_target_miou,
        "target_mdice": test_target_mdice,
        "seconds": test_seconds,
    }
    summary_row.update({f"iou_{name}": float(value) for name, value in zip(class_names, test_ious)})
    summary_row.update({f"dice_{name}": float(value) for name, value in zip(class_names, test_dices)})
    with (output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(summary_row)

    save_sample_predictions(
        model,
        loaders["test"],
        device,
        output_dir / "predictions" / "test",
        args.model,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test target mIoU ({', '.join(selected_names)}): {test_target_miou * 100:.2f}%")
    print(f"Test all-class mIoU: {test_miou * 100:.2f}%")
    for name, iou in zip(class_names, test_ious):
        print(f"IoU {name:>10s}: {iou * 100:.2f}%")


def main() -> None:
    args = parse_args()
    dataset_root = resolve_protocol_dataset(
        project_root=PROJECT_ROOT,
        protocol=args.protocol,
        dataset_root=args.dataset_root,
        dataset_version=args.dataset_version,
        legacy_loso_root=args.loso_root,
    )
    args.dataset_key = args.dataset_key or args.dataset_version or dataset_root.name
    output_root = resolve_path(PROJECT_ROOT, args.output_root)
    for fold, fold_dataset in protocol_datasets(dataset_root, args.protocol, args.folds):
        train_fold(args, fold_dataset, output_root, fold)


if __name__ == "__main__":
    main()
