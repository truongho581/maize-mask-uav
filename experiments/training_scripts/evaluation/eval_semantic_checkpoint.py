from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_CODE = PROJECT_ROOT / "src"
if str(SRC_CODE) not in sys.path:
    sys.path.insert(0, str(SRC_CODE))

from datasets.coco_semantic import (  # noqa: E402
    CocoSemanticDataset,
    input_feature_channels,
    load_coco,
    prepare_class_mapping,
)
from training.semantic_segmentation import build_semantic_model, logits_from_output, select_device  # noqa: E402
from eval_utils import (  # noqa: E402
    Timer,
    colorize_index_mask,
    save_mask,
    save_rgb,
    semantic_error_map,
    summarize_timing,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-evaluate a trained semantic segmentation checkpoint and export paper assets."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--task-mode", default=None, choices=["s3", "raw", None])
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--color-profile", default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def update_confusion_np(confusion: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    valid = (target >= 0) & (target < confusion.shape[0])
    ids = confusion.shape[0] * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    confusion += np.bincount(ids, minlength=confusion.size).reshape(confusion.shape)


def metrics_from_confusion(confusion: np.ndarray, class_names: list[str]) -> dict:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    target_indices = [index for index, name in enumerate(class_names) if name in {"crop", "weed"}]
    if not target_indices:
        target_indices = [index for index, name in enumerate(class_names) if name != "background"]
    return {
        "miou_all": float(iou.mean()),
        "mdice_all": float(dice.mean()),
        "target_miou": float(iou[target_indices].mean()),
        "target_mdice": float(dice[target_indices].mean()),
        "target_classes": [class_names[index] for index in target_indices],
        # Compatibility aliases.
        "miou": float(iou.mean()),
        "mdice": float(dice.mean()),
        "iou": {name: float(value) for name, value in zip(class_names, iou)},
        "dice": {name: float(value) for name, value in zip(class_names, dice)},
        "precision": {name: float(value) for name, value in zip(class_names, precision)},
        "recall": {name: float(value) for name, value in zip(class_names, recall)},
        "confusion_matrix": confusion.tolist(),
    }


def cover_regression_metrics(gt_values: list[float], pred_values: list[float]) -> dict[str, float]:
    gt = np.asarray(gt_values, dtype=np.float64)
    pred = np.asarray(pred_values, dtype=np.float64)
    if len(gt) == 0:
        return {"weed_cover_mae": 0.0, "weed_cover_rmse": 0.0, "weed_cover_corr": 0.0}
    difference = pred - gt
    correlation = 0.0
    if len(gt) > 1 and gt.std() > 0 and pred.std() > 0:
        correlation = float(np.corrcoef(gt, pred)[0, 1])
    return {
        "weed_cover_mae": float(np.abs(difference).mean()),
        "weed_cover_rmse": float(np.sqrt(np.square(difference).mean())),
        "weed_cover_corr": correlation,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_root = resolve(args.dataset_root)
    output_dir = resolve(args.output_dir) if args.output_dir else checkpoint_path.parent / "post_eval" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model or checkpoint.get("model", "attention_unet")
    task_mode = args.task_mode or checkpoint.get("task_mode", "s3")
    img_size = args.img_size or int(checkpoint.get("img_size", 640))
    color_profile = args.color_profile if args.color_profile is not None else checkpoint.get("color_profile", "none")
    model_config = checkpoint.get("config", {}).get("model", {})
    input_features = model_config.get("input_features", checkpoint.get("input_features", "rgb"))
    input_channels = int(
        model_config.get("input_channels", checkpoint.get("input_channels", input_feature_channels(input_features)))
    )
    if input_channels != input_feature_channels(input_features):
        raise ValueError(
            f"Checkpoint input configuration is inconsistent: {input_features=} {input_channels=}"
        )

    train_coco = load_coco(dataset_root, "train")
    split_coco = load_coco(dataset_root, args.split)
    class_info = prepare_class_mapping(train_coco, task_mode)
    class_names = class_info["class_names"]
    device = select_device(args.device)
    dataset = CocoSemanticDataset(
        dataset_root,
        args.split,
        split_coco,
        class_info["category_to_mask"],
        class_info["category_id_to_name"],
        augment=False,
        img_size=img_size,
        color_profile=color_profile,
        input_features=input_features,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type in {"cpu", "mps"} else args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_semantic_model(
        model_name,
        num_classes=len(class_names),
        norm=checkpoint.get("norm", "gn"),
        pretrained_backbone=False,
        input_channels=input_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    per_image_rows = []
    loss_sum = 0.0
    sample_count = 0
    timing_seconds = 0.0
    weed_index = class_names.index("weed") if "weed" in class_names else None
    gt_weed_cover = []
    pred_weed_cover = []

    with torch.no_grad():
        for images, masks, names in loader:
            images = images.to(device)
            masks = masks.to(device)
            with Timer() as timer:
                output = model(images)
                logits = logits_from_output(output, masks.shape[-2:])
            timing_seconds += timer.seconds
            loss = criterion(logits, masks)
            loss_sum += float(loss.item()) * images.shape[0]
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            targets = masks.detach().cpu().numpy()
            rgbs = images[:, :3].detach().cpu().permute(0, 2, 3, 1).numpy()

            for item_index, name in enumerate(names):
                pred = preds[item_index]
                target = targets[item_index]
                item_confusion = np.zeros_like(confusion)
                update_confusion_np(confusion, pred, target)
                update_confusion_np(item_confusion, pred, target)
                item_metrics = metrics_from_confusion(item_confusion, class_names)
                if weed_index is not None:
                    gt_cover = float((target == weed_index).mean())
                    pred_cover = float((pred == weed_index).mean())
                    gt_weed_cover.append(gt_cover)
                    pred_weed_cover.append(pred_cover)
                else:
                    gt_cover = 0.0
                    pred_cover = 0.0
                per_image_rows.append(
                    {
                        "image": name,
                        "miou_all": item_metrics["miou_all"],
                        "target_miou": item_metrics["target_miou"],
                        "gt_weed_cover_fraction": gt_cover,
                        "pred_weed_cover_fraction": pred_cover,
                        "weed_cover_error": pred_cover - gt_cover,
                        **{f"iou_{cls}": item_metrics["iou"][cls] for cls in class_names},
                    }
                )

                if sample_count < args.samples:
                    stem = f"{sample_count:02d}_{Path(name).stem}"
                    sample_dir = output_dir / "qualitative" / args.split
                    rgb = np.clip(rgbs[item_index] * 255.0, 0, 255).astype(np.uint8)
                    save_rgb(sample_dir / f"{stem}_rgb.png", rgb)
                    save_mask(sample_dir / f"{stem}_gt.png", colorize_index_mask(target))
                    save_mask(sample_dir / f"{stem}_pred_{model_name}.png", colorize_index_mask(pred))
                    save_mask(sample_dir / f"{stem}_error.png", semantic_error_map(target, pred, weed_index))
                    sample_count += 1

    metrics = metrics_from_confusion(confusion, class_names)
    metrics.update(cover_regression_metrics(gt_weed_cover, pred_weed_cover))
    metrics.update(
        {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
            "split": args.split,
            "input_features": input_features,
            "input_channels": input_channels,
            "images": len(dataset),
            "loss": loss_sum / max(len(dataset), 1),
            **summarize_timing(timing_seconds, len(dataset)),
        }
    )
    write_json(output_dir / f"{args.split}_metrics.json", metrics)
    write_csv(output_dir / f"{args.split}_per_image_metrics.csv", per_image_rows)

    summary_row = {
        "split": args.split,
        "loss": metrics["loss"],
        "miou_all": metrics["miou_all"],
        "mdice_all": metrics["mdice_all"],
        "target_miou": metrics["target_miou"],
        "target_mdice": metrics["target_mdice"],
        "weed_cover_mae": metrics["weed_cover_mae"],
        "weed_cover_rmse": metrics["weed_cover_rmse"],
        "weed_cover_corr": metrics["weed_cover_corr"],
        "seconds_per_image": metrics["seconds_per_image"],
        "fps": metrics["fps"],
    }
    for class_name in class_names:
        summary_row[f"iou_{class_name}"] = metrics["iou"][class_name]
        summary_row[f"dice_{class_name}"] = metrics["dice"][class_name]
        summary_row[f"precision_{class_name}"] = metrics["precision"][class_name]
        summary_row[f"recall_{class_name}"] = metrics["recall"][class_name]
    write_csv(output_dir / "summary_metrics.csv", [summary_row])

    print(f"Saved semantic post-eval to: {output_dir}")
    print(
        f"Target mIoU: {metrics['target_miou'] * 100:.2f}% | "
        f"All-class mIoU: {metrics['miou_all'] * 100:.2f}% | FPS: {metrics['fps']:.2f}"
    )


if __name__ == "__main__":
    main()
