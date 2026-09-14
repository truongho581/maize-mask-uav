from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_CODE = PROJECT_ROOT / "src"
if str(SRC_CODE) not in sys.path:
    sys.path.insert(0, str(SRC_CODE))

from datasets.coco_maize_instance import (  # noqa: E402
    STAGE_CLASS_NAMES,
    CocoMaizeInstanceDataset,
    build_annotations_by_image,
    build_stage_category_mapping,
    collate_fn,
    load_coco,
)
from datasets.coco_semantic import annotation_to_binary_mask  # noqa: E402
from nets.model_zoo import build_maskrcnn_model  # noqa: E402
from training.semantic_segmentation import select_device  # noqa: E402
from training.instance_metrics import (  # noqa: E402
    build_ignore_masks,
    evaluate_instance_records,
)
from utils import (  # noqa: E402
    BinaryMetricAccumulator,
    Timer,
    count_regression_metrics,
    colorize_stage_mask,
    save_mask,
    save_rgb,
    stage_error_map,
    summarize_timing,
    write_csv,
    write_json,
)


LABEL_TO_NAME = {1: "maize2", 2: "maize4", 3: "maize6"}
NAME_TO_LABEL = {name: label for label, name in LABEL_TO_NAME.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-evaluate a trained Mask R-CNN checkpoint and export paper assets."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=48)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resize_bool(mask: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (size, size),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    ) > 0


def build_gt_and_ignore(coco: dict, img_size: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    category_id_to_name = {category["id"]: category["name"] for category in coco["categories"]}
    annotations_by_image = build_annotations_by_image(coco)
    output = {}
    for image_info in coco["images"]:
        height = int(image_info["height"])
        width = int(image_info["width"])
        gt = np.zeros((img_size, img_size), dtype=np.uint8)
        ignore = np.zeros((img_size, img_size), dtype=bool)
        for annotation in annotations_by_image.get(image_info["id"], []):
            name = category_id_to_name.get(annotation["category_id"])
            binary = annotation_to_binary_mask(annotation, height, width)
            binary = resize_bool(binary, img_size)
            if name == "maize-u":
                ignore |= binary
            elif name in NAME_TO_LABEL:
                gt[binary] = NAME_TO_LABEL[name]
        output[image_info["id"]] = (gt, ignore)
    return output


def prediction_to_stage_mask(output: dict, img_size: int, score_threshold: float, mask_threshold: float) -> np.ndarray:
    pred = np.zeros((img_size, img_size), dtype=np.uint8)
    if len(output.get("scores", [])) == 0:
        return pred
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    masks = output["masks"][:, 0].detach().cpu().numpy()
    for label, score, mask in zip(labels, scores, masks):
        if int(label) not in LABEL_TO_NAME or float(score) < score_threshold:
            continue
        pred[mask >= mask_threshold] = int(label)
    return pred


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_root = resolve(args.dataset_root)
    output_dir = resolve(args.output_dir) if args.output_dir else checkpoint_path.parent / "post_eval" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = args.img_size or int(checkpoint.get("img_size", 640))
    score_threshold = args.score_threshold if args.score_threshold is not None else float(checkpoint.get("score_threshold", 0.25))
    mask_threshold = args.mask_threshold if args.mask_threshold is not None else float(checkpoint.get("mask_threshold", 0.5))

    train_coco = load_coco(dataset_root, "train")
    split_coco = load_coco(dataset_root, args.split)
    category_to_label, _ = build_stage_category_mapping(train_coco)
    device = select_device(args.device)
    dataset = CocoMaizeInstanceDataset(
        dataset_root,
        args.split,
        split_coco,
        category_to_label,
        augment=False,
        img_size=img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type in {"cpu", "mps"} else args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    gt_ignore_by_image = build_gt_and_ignore(split_coco, img_size)
    ignore_masks_by_image = build_ignore_masks(split_coco, img_size)

    model_name = checkpoint.get("model", "maskrcnn_resnet50_fpn")
    model = build_maskrcnn_model(
        model_name,
        num_classes=len(STAGE_CLASS_NAMES),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = ["maize", "maize2", "maize4", "maize6"]
    acc = BinaryMetricAccumulator(labels)
    per_image_rows = []
    gt_counts = []
    pred_counts = []
    timing_seconds = 0.0
    sample_count = 0
    instance_records = []

    with torch.no_grad():
        for images, targets in loader:
            images_device = [image.to(device) for image in images]
            with Timer() as timer:
                outputs = model(images_device)
            timing_seconds += timer.seconds

            for image, target, output in zip(images, targets, outputs):
                image_id = int(target["image_id"][0].item())
                gt, ignore = gt_ignore_by_image[image_id]
                pred = prediction_to_stage_mask(output, img_size, score_threshold, mask_threshold)
                valid = ~ignore
                target_masks = target["masks"].detach().cpu().numpy().astype(bool)
                target_labels = target["labels"].detach().cpu().numpy().astype(np.int64)
                output_masks = output["masks"][:, 0].detach().cpu().numpy() >= mask_threshold
                output_labels = output["labels"].detach().cpu().numpy().astype(np.int64)
                output_scores = output["scores"].detach().cpu().numpy().astype(np.float64)
                metric_valid = ~ignore_masks_by_image[image_id]
                target_masks &= metric_valid[None, ...]
                output_masks &= metric_valid[None, ...]
                target_nonempty = (
                    target_masks.reshape(len(target_masks), -1).any(axis=1)
                    if len(target_masks)
                    else np.zeros(0, dtype=bool)
                )
                output_nonempty = (
                    output_masks.reshape(len(output_masks), -1).any(axis=1)
                    if len(output_masks)
                    else np.zeros(0, dtype=bool)
                )
                target_masks = target_masks[target_nonempty]
                target_labels = target_labels[target_nonempty]
                output_masks = output_masks[output_nonempty]
                output_labels = output_labels[output_nonempty]
                output_scores = output_scores[output_nonempty]
                instance_records.append(
                    {
                        "image_id": image_id,
                        "shape": metric_valid.shape,
                        "gt_masks": target_masks,
                        "gt_labels": target_labels,
                        "pred_masks": output_masks,
                        "pred_labels": output_labels,
                        "pred_scores": output_scores,
                    }
                )
                gt_visible_count = int(len(target_labels))
                pred_visible_count = int((output_scores >= score_threshold).sum())
                gt_counts.append(gt_visible_count)
                pred_counts.append(pred_visible_count)

                acc.update("maize", pred > 0, gt > 0, valid)
                for label, name in LABEL_TO_NAME.items():
                    acc.update(name, pred == label, gt == label, valid)

                row = {
                    "image_id": image_id,
                    "gt_visible_maize_count": gt_visible_count,
                    "pred_visible_maize_count": pred_visible_count,
                    "visible_maize_count_error": pred_visible_count - gt_visible_count,
                    "visible_maize_count_abs_error": abs(pred_visible_count - gt_visible_count),
                }
                row_acc = BinaryMetricAccumulator(labels)
                row_acc.update("maize", pred > 0, gt > 0, valid)
                for label, name in LABEL_TO_NAME.items():
                    row_acc.update(name, pred == label, gt == label, valid)
                row.update(row_acc.metrics())
                per_image_rows.append(row)

                if sample_count < args.samples:
                    stem = f"{sample_count:02d}_image_{image_id}"
                    sample_dir = output_dir / "qualitative" / args.split
                    rgb = np.clip(image.permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
                    save_rgb(sample_dir / f"{stem}_rgb.png", rgb)
                    save_mask(sample_dir / f"{stem}_gt_stage.png", colorize_stage_mask(gt, ignore))
                    save_mask(sample_dir / f"{stem}_pred_maskrcnn.png", colorize_stage_mask(pred, ignore))
                    save_mask(sample_dir / f"{stem}_error.png", stage_error_map(gt, pred, ignore))
                    sample_count += 1

    metrics = acc.metrics()
    ap_metrics = evaluate_instance_records(
        instance_records,
        score_threshold=score_threshold,
        max_detections=args.max_detections,
    )
    metrics.update(ap_metrics)
    metrics.update(
        {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
            "split": args.split,
            "images": len(dataset),
            "score_threshold": score_threshold,
            "mask_threshold": mask_threshold,
            **summarize_timing(timing_seconds, len(dataset)),
        }
    )
    metrics["mean_stage_iou"] = float(np.mean([metrics[f"iou_{name}"] for name in ["maize2", "maize4", "maize6"]]))
    metrics["mean_stage_dice"] = float(np.mean([metrics[f"dice_{name}"] for name in ["maize2", "maize4", "maize6"]]))
    metrics.update(count_regression_metrics(gt_counts, pred_counts))
    metrics["selection_metric_primary"] = "mask_map"
    metrics["selection_metric_secondary"] = "macro_per_stage_mask_ap"
    metrics["selection_metric_value"] = metrics["mask_map"]

    write_json(output_dir / f"{args.split}_metrics.json", metrics)
    write_csv(output_dir / f"{args.split}_per_image_metrics.csv", per_image_rows)
    summary = {key: value for key, value in metrics.items() if not isinstance(value, (dict, list))}
    for stage, values in metrics["per_stage"].items():
        for key, value in values.items():
            summary[f"{stage}_{key}"] = value
    write_csv(output_dir / "summary_metrics.csv", [summary])

    print(f"Saved {model_name} post-eval to: {output_dir}")
    print(
        f"Mask mAP: {metrics['mask_map'] * 100:.2f}% | "
        f"Maize region IoU: {metrics['iou_maize'] * 100:.2f}% | FPS: {metrics['fps']:.2f}"
    )


if __name__ == "__main__":
    main()
