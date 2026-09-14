from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Any

# Must be set before CUDA is initialized to make CuBLAS deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_CODE = PROJECT_ROOT / "src"
if str(SRC_CODE) not in sys.path:
    sys.path.insert(0, str(SRC_CODE))

from datasets.coco_maize_instance import (  # noqa: E402
    STAGE_CLASS_NAMES,
    CocoMaizeInstanceDataset,
    build_stage_category_mapping,
    collate_fn,
    load_coco,
)
from nets.model_zoo import INSTANCE_MODEL_NAMES, build_maskrcnn_model, get_model_spec  # noqa: E402
from training.instance_metrics import (  # noqa: E402
    build_ignore_masks,
    evaluate_instance_model,
)
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
    count_parameters,
    seed_everything,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN on MaizeMask LOSO or fixed splits."
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
        help="Stable dataset label used in run paths when --dataset-root is supplied (for example: v1.0).",
    )
    parser.add_argument(
        "--loso-root",
        type=Path,
        default=None,
        help="Deprecated alias for --dataset-root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model",
        choices=INSTANCE_MODEL_NAMES,
        default="maskrcnn_resnet50_fpn",
        help="Official Torchvision Mask R-CNN implementation variant.",
    )
    parser.add_argument("--folds", nargs="+", default=None)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--augment-profile",
        choices=["horizontal_flip", "none"],
        default="horizontal_flip",
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-optimizer-state", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Print a lightweight training heartbeat every N batches; use 0 to disable.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_loader(
    dataset: CocoMaizeInstanceDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_mask_map: float,
    best_macro_stage_ap: float,
    config: dict[str, Any],
    include_optimizer: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_mask_map": best_mask_map,
        "best_macro_per_stage_mask_ap": best_macro_stage_ap,
        "checkpoint_criterion": config["checkpoint_selection"],
        "config": config,
        # Compatibility with post-evaluation scripts made before schema v2.
        "model": config["model"]["name"],
        "dataset_root": config["data"]["dataset_root"],
        "class_names": config["task"]["class_names"],
        "category_to_label": config["task"]["category_to_label"],
        "img_size": config["data"]["img_size"],
        "pretrained": config["model"]["pretrained"],
        "score_threshold": config["inference"]["score_threshold"],
        "mask_threshold": config["inference"]["mask_threshold"],
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def is_better(
    mask_map: float,
    macro_stage_ap: float,
    mask_ap75: float,
    best: tuple[float, float, float],
    tolerance: float = 1e-12,
) -> bool:
    candidate = (mask_map, macro_stage_ap, mask_ap75)
    for value, best_value in zip(candidate, best):
        if value > best_value + tolerance:
            return True
        if value < best_value - tolerance:
            return False
    return False


def flattened_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output = {
        f"{prefix}{key}": value
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    }
    for stage, stage_metrics in metrics.get("per_stage", {}).items():
        for key, value in stage_metrics.items():
            output[f"{prefix}{stage}_{key}"] = value
    return output


def train_fold(
    args: argparse.Namespace,
    dataset_root: Path,
    output_root: Path,
    fold: str,
) -> None:
    seed_everything(args.seed, deterministic=True)
    device = select_device(args.device)
    output_dir = run_directory(
        output_root,
        task="instance",
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

    coco_by_split = {split: load_coco(dataset_root, split) for split in ("train", "valid", "test")}
    category_to_label, category_id_to_name = build_stage_category_mapping(coco_by_split["train"])
    augment = args.augment_profile != "none"
    datasets = {
        split: CocoMaizeInstanceDataset(
            dataset_root,
            split,
            coco_by_split[split],
            category_to_label,
            augment=(split == "train" and augment),
            img_size=args.img_size,
        )
        for split in ("train", "valid", "test")
    }
    workers = 0 if device.type in {"cpu", "mps"} else args.num_workers
    loaders = {
        "train": build_loader(datasets["train"], args.batch_size, workers, True, device),
        "valid": build_loader(datasets["valid"], 1, workers, False, device),
        "test": build_loader(datasets["test"], 1, workers, False, device),
    }
    ignore_masks = {
        split: build_ignore_masks(coco_by_split[split], args.img_size)
        for split in ("valid", "test")
    }

    model = build_maskrcnn_model(
        args.model,
        num_classes=len(STAGE_CLASS_NAMES),
        pretrained=args.pretrained,
    ).to(device)
    model_spec = get_model_spec(args.model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config: dict[str, Any] = {
        "schema_version": 2,
        "task": {
            "type": "instance_segmentation",
            "class_names": STAGE_CLASS_NAMES,
            "evaluated_classes": ["maize2", "maize4", "maize6"],
            "ignored_categories": ["maize-u"],
            "excluded_categories": ["weed"],
            "category_to_label": category_to_label,
            "category_id_to_name": category_id_to_name,
        },
        "model": {
            "name": args.model,
            "display_name": model_spec.name,
            "implementation": model_spec.implementation,
            "backbone": "ResNet-50 FPN",
            "pretrained": args.pretrained,
            "pretraining_source": "Torchvision COCO DEFAULT" if args.pretrained else None,
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
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "loss": "Torchvision Mask R-CNN multi-task losses",
            "seed": args.seed,
            "deterministic": True,
            "early_stopping_patience": args.patience,
            "num_workers": args.num_workers,
            "save_optimizer_state": args.save_optimizer_state,
        },
        "augmentation": {
            "profile": args.augment_profile,
            "train_only": True,
        },
        "inference": {
            "score_threshold": args.score_threshold,
            "mask_threshold": args.mask_threshold,
            "max_detections_per_image": args.max_detections,
        },
        "checkpoint_selection": {
            "primary": "valid_mask_map",
            "definition": "class-aware mask AP averaged over IoU 0.50:0.05:0.95",
            "tie_break": "valid_macro_per_stage_mask_ap",
            "tertiary_tie_break": "valid_mask_ap75",
            "test_used_for_selection": False,
            "note": (
                "With exactly three evaluated stage classes, standard mask mAP and macro "
                "per-stage AP are numerically equivalent; AP75 resolves any remaining tie."
            ),
        },
        "runtime": runtime_metadata(device),
    }
    write_run_config(output_dir, config)
    write_command(output_dir)

    print("\n" + "=" * 80)
    print(f"{model_spec.name} | Protocol: {args.protocol} | Fold: {fold} | Seed: {args.seed}")
    print("Dataset:", dataset_root)
    print("Output:", output_dir)
    print("Device:", device)
    print("Train images:", len(datasets["train"]), "Valid images:", len(datasets["valid"]))
    print("Classes:", STAGE_CLASS_NAMES)
    print(f"Trainable params: {count_parameters(model) / 1e6:.2f}M")
    if args.preview_only:
        print("Configuration and dataset preview OK. Stop because --preview-only was set.")
        return

    history_path = output_dir / "history.csv"
    history_fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "valid_mask_map",
        "valid_mask_ap50",
        "valid_mask_ap75",
        "valid_macro_per_stage_mask_ap",
        "valid_pooled_maize_mask_ap",
        *[f"valid_{stage}_ap" for stage in ("maize2", "maize4", "maize6")],
        "is_best",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=history_fields).writeheader()

    best_score = (-1.0, -1.0, -1.0)
    epochs_without_improvement = 0
    final_epoch = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        train_loss_sum = 0.0
        total_batches = len(loaders["train"])
        for batch_index, (images, targets) in enumerate(loaders["train"], start=1):
            images = [image.to(device) for image in images]
            targets_device = [
                {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            losses = model(images, targets_device)
            loss = sum(loss_value for loss_value in losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(images)
            if args.log_interval > 0 and (
                batch_index == 1
                or batch_index % args.log_interval == 0
                or batch_index == total_batches
            ):
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | batch {batch_index:03d}/{total_batches:03d} | "
                    f"batch_loss={float(loss.item()):.4f}",
                    flush=True,
                )
        train_loss = train_loss_sum / len(datasets["train"])

        valid_metrics = evaluate_instance_model(
            model,
            loaders["valid"],
            device,
            ignore_masks["valid"],
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            max_detections=args.max_detections,
        )
        current_score = (
            float(valid_metrics["mask_map"]),
            float(valid_metrics["macro_per_stage_mask_ap"]),
            float(valid_metrics["mask_ap75"]),
        )
        improved = is_better(*current_score, best_score)
        if improved:
            best_score = current_score
            epochs_without_improvement = 0
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score[0],
                    best_score[1],
                    config,
                    include_optimizer=False,
                ),
                output_dir / "best.pth",
            )
            write_json(output_dir / "best_valid_metrics.json", valid_metrics)
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "valid_mask_map": valid_metrics["mask_map"],
            "valid_mask_ap50": valid_metrics["mask_ap50"],
            "valid_mask_ap75": valid_metrics["mask_ap75"],
            "valid_macro_per_stage_mask_ap": valid_metrics["macro_per_stage_mask_ap"],
            "valid_pooled_maize_mask_ap": valid_metrics["pooled_maize_mask_ap"],
            **{
                f"valid_{stage}_ap": valid_metrics["per_stage"][stage]["ap"]
                for stage in ("maize2", "maize4", "maize6")
            },
            "is_best": int(improved),
        }
        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=history_fields).writerow(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"valid mask mAP={current_score[0] * 100:.2f}% | "
            f"AP50={valid_metrics['mask_ap50'] * 100:.2f}% | "
            f"AP75={valid_metrics['mask_ap75'] * 100:.2f}% | "
            f"best={best_score[0] * 100:.2f}%"
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
            best_score[0],
            best_score[1],
            config,
            include_optimizer=args.save_optimizer_state,
        ),
        output_dir / "last.pth",
    )

    checkpoint = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_metrics = evaluate_instance_model(
        model,
        loaders["valid"],
        device,
        ignore_masks["valid"],
        args.score_threshold,
        args.mask_threshold,
        args.max_detections,
    )
    test_metrics = evaluate_instance_model(
        model,
        loaders["test"],
        device,
        ignore_masks["test"],
        args.score_threshold,
        args.mask_threshold,
        args.max_detections,
    )
    for metrics, split in ((valid_metrics, "valid"), (test_metrics, "test")):
        metrics.update(
            {
                "split": split,
                "checkpoint": "best.pth",
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_selection": config["checkpoint_selection"],
            }
        )
    write_json(output_dir / "valid_metrics.json", valid_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)

    rows = [
        {"split": "valid", **flattened_metrics(valid_metrics)},
        {"split": "test", **flattened_metrics(test_metrics)},
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Test mask mAP: {test_metrics['mask_map'] * 100:.2f}%")
    print(f"Test mask AP50: {test_metrics['mask_ap50'] * 100:.2f}%")
    print(f"Test mask AP75: {test_metrics['mask_ap75'] * 100:.2f}%")


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
