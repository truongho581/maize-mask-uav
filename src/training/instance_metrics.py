from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

try:
    from pycocotools import mask as coco_mask
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover - the project requirements include pycocotools.
    coco_mask = None
    COCO = None
    COCOeval = None

from datasets.coco_maize_instance import build_annotations_by_image
from datasets.coco_semantic import annotation_to_binary_mask


IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
STAGE_LABELS = {1: "maize2", 2: "maize4", 3: "maize6"}


def _resize_mask(mask: np.ndarray, img_size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (img_size, img_size),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    ).astype(bool)


def build_ignore_masks(
    coco: dict[str, Any],
    img_size: int,
    ignore_category_names: Iterable[str] = ("maize-u",),
) -> dict[int, np.ndarray]:
    ignored = set(ignore_category_names)
    category_names = {category["id"]: category["name"] for category in coco["categories"]}
    annotations_by_image = build_annotations_by_image(coco)
    output: dict[int, np.ndarray] = {}
    for image in coco["images"]:
        ignore = np.zeros((img_size, img_size), dtype=bool)
        for annotation in annotations_by_image.get(image["id"], []):
            if category_names.get(annotation["category_id"]) not in ignored:
                continue
            mask = annotation_to_binary_mask(
                annotation,
                int(image["height"]),
                int(image["width"]),
            )
            ignore |= _resize_mask(mask, img_size)
        output[int(image["id"])] = ignore
    return output


def _mask_iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)

    if coco_mask is not None:
        # COCO's compiled RLE implementation avoids allocating a large
        # (predictions, ground-truth, height, width) boolean tensor.
        pred_rles = coco_mask.encode(np.asfortranarray(pred_masks.transpose(1, 2, 0).astype(np.uint8)))
        gt_rles = coco_mask.encode(np.asfortranarray(gt_masks.transpose(1, 2, 0).astype(np.uint8)))
        return np.asarray(
            coco_mask.iou(pred_rles, gt_rles, [0] * len(gt_rles)),
            dtype=np.float64,
        )

    # Retain a bounded-memory fallback for environments without pycocotools.
    pred = pred_masks.reshape(len(pred_masks), -1).astype(bool)
    gt = gt_masks.reshape(len(gt_masks), -1).astype(bool)
    ious = np.zeros((len(pred), len(gt)), dtype=np.float64)
    for start in range(0, len(pred), 8):
        chunk = pred[start : start + 8]
        intersection = np.logical_and(chunk[:, None, :], gt[None, :, :]).sum(axis=2)
        union = np.logical_or(chunk[:, None, :], gt[None, :, :]).sum(axis=2)
        ious[start : start + len(chunk)] = intersection / np.maximum(union, 1)
    return ious


def _interpolated_ap(tp: np.ndarray, fp: np.ndarray, gt_count: int) -> float:
    if gt_count == 0:
        return float("nan")
    tp_cumulative = np.cumsum(tp)
    fp_cumulative = np.cumsum(fp)
    recall = tp_cumulative / gt_count
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(precision[recall >= point].max()) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    return float(np.mean(interpolated))


@dataclass
class _MatchContext:
    predictions: list[tuple[float, int, np.ndarray]]
    gt_counts: dict[int, int]
    gt_count: int


def _build_match_context(
    records: list[dict[str, Any]],
    label: int | None,
    max_detections: int,
) -> _MatchContext:
    """Prepare scores and IoUs once for all AP thresholds of one class."""
    gt_counts: dict[int, int] = {}
    predictions: list[tuple[float, int, np.ndarray]] = []

    for record in records:
        image_id = int(record["image_id"])
        gt_keep = np.ones(len(record["gt_labels"]), dtype=bool)
        pred_keep = np.ones(len(record["pred_labels"]), dtype=bool)
        if label is not None:
            gt_keep = record["gt_labels"] == label
            pred_keep = record["pred_labels"] == label
        gt_masks = record["gt_masks"][gt_keep]
        gt_counts[image_id] = len(gt_masks)

        indices = np.where(pred_keep)[0]
        if len(indices):
            order = indices[np.argsort(-record["pred_scores"][indices])][:max_detections]
            ious = _mask_iou_matrix(record["pred_masks"][order], gt_masks)
            for row, index in enumerate(order):
                score = float(record["pred_scores"][index])
                predictions.append((score, image_id, ious[row]))

    predictions.sort(key=lambda item: item[0], reverse=True)
    return _MatchContext(
        predictions=predictions,
        gt_counts=gt_counts,
        gt_count=sum(gt_counts.values()),
    )


def _match_context(
    context: _MatchContext,
    iou_threshold: float,
    minimum_score: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    matched = {
        image_id: np.zeros(count, dtype=bool)
        for image_id, count in context.gt_counts.items()
    }
    kept_predictions = [
        prediction
        for prediction in context.predictions
        if minimum_score is None or prediction[0] >= minimum_score
    ]
    true_positive = np.zeros(len(kept_predictions), dtype=np.float64)
    false_positive = np.zeros(len(kept_predictions), dtype=np.float64)

    for index, (_, image_id, ious) in enumerate(kept_predictions):
        available = np.where(~matched.get(image_id, np.zeros(0, dtype=bool)))[0]
        if len(available) == 0:
            false_positive[index] = 1.0
            continue
        available_ious = ious[available]
        best_local = int(np.argmax(available_ious))
        if float(available_ious[best_local]) >= iou_threshold:
            matched[image_id][available[best_local]] = True
            true_positive[index] = 1.0
        else:
            false_positive[index] = 1.0

    return true_positive, false_positive, context.gt_count


def _label_ap(
    context: _MatchContext,
) -> dict[str, float]:
    ap_by_threshold = []
    for threshold in IOU_THRESHOLDS:
        tp, fp, gt_count = _match_context(context, float(threshold))
        ap_by_threshold.append(_interpolated_ap(tp, fp, gt_count))
    values = np.asarray(ap_by_threshold, dtype=np.float64)
    return {
        "ap": float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan"),
        "ap50": float(values[0]),
        "ap75": float(values[5]),
    }


def _operating_point(
    context: _MatchContext,
    score_threshold: float,
) -> tuple[float, float]:
    tp, fp, gt_count = _match_context(context, iou_threshold=0.5, minimum_score=score_threshold)
    true_positives = float(tp.sum())
    precision = true_positives / max(true_positives + float(fp.sum()), 1e-12)
    recall = true_positives / max(float(gt_count), 1e-12)
    return precision, recall


def _region_metrics(
    records: list[dict[str, Any]],
    score_threshold: float,
) -> dict[str, float]:
    intersections: defaultdict[str, int] = defaultdict(int)
    unions: defaultdict[str, int] = defaultdict(int)
    denominators: defaultdict[str, int] = defaultdict(int)

    for record in records:
        keep = record["pred_scores"] >= score_threshold
        for label, name in [*STAGE_LABELS.items(), (None, "maize")]:
            if label is None:
                gt_selected = record["gt_masks"]
                pred_selected = record["pred_masks"][keep]
            else:
                gt_selected = record["gt_masks"][record["gt_labels"] == label]
                pred_selected = record["pred_masks"][keep & (record["pred_labels"] == label)]
            shape = record["shape"]
            gt_mask = gt_selected.any(axis=0) if len(gt_selected) else np.zeros(shape, dtype=bool)
            pred_mask = pred_selected.any(axis=0) if len(pred_selected) else np.zeros(shape, dtype=bool)
            intersections[name] += int(np.logical_and(gt_mask, pred_mask).sum())
            unions[name] += int(np.logical_or(gt_mask, pred_mask).sum())
            denominators[name] += int(gt_mask.sum() + pred_mask.sum())

    metrics: dict[str, float] = {}
    for name in ["maize", *STAGE_LABELS.values()]:
        metrics[f"region_iou_{name}"] = intersections[name] / max(unions[name], 1e-12)
        metrics[f"region_dice_{name}"] = 2 * intersections[name] / max(
            denominators[name], 1e-12
        )
    metrics["region_mean_stage_iou"] = float(
        np.mean([metrics[f"region_iou_{name}"] for name in STAGE_LABELS.values()])
    )
    metrics["region_mean_stage_dice"] = float(
        np.mean([metrics[f"region_dice_{name}"] for name in STAGE_LABELS.values()])
    )
    return metrics


def _json_rle(mask: np.ndarray) -> dict[str, Any]:
    if coco_mask is None:
        raise ImportError("Official COCO evaluation requires pycocotools.")
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts,
    }


def evaluate_official_coco_mask_ap(
    records: list[dict[str, Any]],
    max_detections: int = 100,
) -> dict[str, float]:
    """Evaluate decoded RLE targets using the official pycocotools COCOeval path.

    The supplied records are already restricted to `maize2`, `maize4`, `maize6`
    and have the same 640-pixel raster used by model inference.  Thus this avoids
    polygon conversion while making the Mask R-CNN selection metric directly
    comparable to Detectron2's COCO `segm` evaluator for Mask2Former.
    """
    if coco_mask is None or COCO is None or COCOeval is None:
        raise ImportError("Official COCO evaluation requires pycocotools.")

    images: list[dict[str, int]] = []
    annotations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    annotation_id = 1
    for record in records:
        image_id = int(record["image_id"])
        height, width = (int(record["shape"][0]), int(record["shape"][1]))
        images.append({"id": image_id, "height": height, "width": width})
        for mask, label in zip(record["gt_masks"], record["gt_labels"]):
            if not bool(mask.any()):
                continue
            rle = _json_rle(mask)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "segmentation": rle,
                    "area": float(coco_mask.area(rle)),
                    "bbox": [float(value) for value in coco_mask.toBbox(rle)],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        order = np.argsort(-record["pred_scores"])[:max_detections]
        for index in order:
            mask = record["pred_masks"][index]
            if not bool(mask.any()):
                continue
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(record["pred_labels"][index]),
                    "segmentation": _json_rle(mask),
                    "score": float(record["pred_scores"][index]),
                }
            )

    categories = [{"id": label, "name": name} for label, name in STAGE_LABELS.items()]
    ground_truth = COCO()
    ground_truth.dataset = {
        # pycocotools 2.0.10's ``loadRes`` unconditionally accesses this
        # optional COCO field.  Keep the in-memory evaluation dataset valid
        # across both older and newer pycocotools releases.
        "info": {"description": "MaizeMask in-memory instance evaluation"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ground_truth.createIndex()
    if not predictions:
        return {
            "official_coco_mask_ap": 0.0,
            "official_coco_mask_ap50": 0.0,
            "official_coco_mask_ap75": 0.0,
        }
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, "segm")
    evaluator.params.catIds = list(STAGE_LABELS)
    evaluator.params.maxDets = [1, 10, max_detections]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "official_coco_mask_ap": float(evaluator.stats[0]),
        "official_coco_mask_ap50": float(evaluator.stats[1]),
        "official_coco_mask_ap75": float(evaluator.stats[2]),
    }


def evaluate_instance_records(
    records: list[dict[str, Any]],
    score_threshold: float = 0.25,
    max_detections: int = 100,
) -> dict[str, Any]:
    per_stage: dict[str, dict[str, float]] = {}
    for label, name in STAGE_LABELS.items():
        context = _build_match_context(records, label, max_detections)
        values = _label_ap(context)
        precision, recall = _operating_point(
            context,
            score_threshold,
        )
        per_stage[name] = {**values, "precision50": precision, "recall50": recall}

    stage_ap = [values["ap"] for values in per_stage.values()]
    stage_ap50 = [values["ap50"] for values in per_stage.values()]
    stage_ap75 = [values["ap75"] for values in per_stage.values()]
    pooled_context = _build_match_context(records, None, max_detections)
    pooled = _label_ap(pooled_context)
    pooled_precision, pooled_recall = _operating_point(
        pooled_context,
        score_threshold,
    )
    metrics: dict[str, Any] = {
        "mask_map": float(np.nanmean(stage_ap)),
        "mask_ap50": float(np.nanmean(stage_ap50)),
        "mask_ap75": float(np.nanmean(stage_ap75)),
        "macro_per_stage_mask_ap": float(np.nanmean(stage_ap)),
        "pooled_maize_mask_ap": pooled["ap"],
        "pooled_maize_mask_ap50": pooled["ap50"],
        "pooled_maize_mask_ap75": pooled["ap75"],
        "pooled_maize_precision50": pooled_precision,
        "pooled_maize_recall50": pooled_recall,
        "per_stage": per_stage,
        "iou_thresholds": [float(value) for value in IOU_THRESHOLDS],
        "score_threshold_for_precision_recall": score_threshold,
        "max_detections_per_image": max_detections,
    }
    official = evaluate_official_coco_mask_ap(records, max_detections=max_detections)
    metrics.update(official)
    # The paper checkpoint rule uses official COCO `segm` AP.  The macro values
    # remain as transparent class-balance diagnostics and deterministic tie-breaks.
    metrics["mask_map"] = official["official_coco_mask_ap"]
    metrics["mask_ap50"] = official["official_coco_mask_ap50"]
    metrics["mask_ap75"] = official["official_coco_mask_ap75"]
    metrics["mask_ap_evaluator"] = "pycocotools.COCOeval segm"
    metrics.update(_region_metrics(records, score_threshold))
    return metrics


def collect_instance_records(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_masks: dict[int, np.ndarray],
    mask_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"][0].item())
                ignore = ignore_masks.get(
                    image_id,
                    np.zeros(tuple(target["masks"].shape[-2:]), dtype=bool),
                )
                valid = ~ignore
                gt_masks = target["masks"].detach().cpu().numpy().astype(bool)
                gt_labels = target["labels"].detach().cpu().numpy().astype(np.int64)
                pred_masks = (
                    output["masks"][:, 0].detach().cpu().numpy() >= mask_threshold
                    if len(output["masks"])
                    else np.zeros((0, *valid.shape), dtype=bool)
                )
                pred_labels = output["labels"].detach().cpu().numpy().astype(np.int64)
                pred_scores = output["scores"].detach().cpu().numpy().astype(np.float64)

                gt_masks &= valid[None, ...]
                pred_masks &= valid[None, ...]
                gt_nonempty = (
                    gt_masks.reshape(len(gt_masks), -1).any(axis=1)
                    if len(gt_masks)
                    else np.zeros(0, dtype=bool)
                )
                pred_nonempty = (
                    pred_masks.reshape(len(pred_masks), -1).any(axis=1)
                    if len(pred_masks)
                    else np.zeros(0, dtype=bool)
                )
                records.append(
                    {
                        "image_id": image_id,
                        "shape": valid.shape,
                        "gt_masks": gt_masks[gt_nonempty],
                        "gt_labels": gt_labels[gt_nonempty],
                        "pred_masks": pred_masks[pred_nonempty],
                        "pred_labels": pred_labels[pred_nonempty],
                        "pred_scores": pred_scores[pred_nonempty],
                    }
                )
    return records


def evaluate_instance_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_masks: dict[int, np.ndarray],
    score_threshold: float = 0.25,
    mask_threshold: float = 0.5,
    max_detections: int = 100,
) -> dict[str, Any]:
    records = collect_instance_records(model, loader, device, ignore_masks, mask_threshold)
    return evaluate_instance_records(records, score_threshold, max_detections)
