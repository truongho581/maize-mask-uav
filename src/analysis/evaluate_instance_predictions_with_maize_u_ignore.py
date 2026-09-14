"""Re-evaluate COCO instance predictions with MaizeMask ``maize-u`` pixels ignored.

Mask R-CNN masks its stage targets and predictions by the canonical ``maize-u``
region before COCO evaluation.  This utility applies the same rule to exported
COCO prediction RLEs, allowing an apples-to-apples post-evaluation of models
whose native evaluator only received a stage-only COCO file (e.g. Mask2Former).
It is read-only with respect to the model run and canonical dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as error:  # pragma: no cover - documented runtime dependency.
    raise SystemExit("pycocotools is required; install the project training dependencies.") from error


STAGE_NAMES = {"maize2", "maize4", "maize6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True, help="Canonical COCO for one evaluated split.")
    parser.add_argument("--predictions", type=Path, required=True, help="COCO RLE predictions JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Metrics JSON to write.")
    parser.add_argument(
        "--export-clipped-predictions",
        type=Path,
        help="Optional path for the clipped COCO detections JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rle_from_annotation(annotation: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    return segmentation


def serializable_rle(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts,
    }


def category_ap(evaluator: COCOeval, category_id: int) -> float:
    """Return COCO AP averaged over IoU thresholds for one category."""
    copy = COCOeval(evaluator.cocoGt, evaluator.cocoDt, "segm")
    copy.params.imgIds = evaluator.params.imgIds
    copy.params.catIds = [category_id]
    copy.params.maxDets = evaluator.params.maxDets
    copy.evaluate()
    copy.accumulate()
    # precision has shape IoU x recall x category x area x maxDet.  The first
    # area is ``all`` and the final maxDet is 100, matching headline COCO AP.
    values = copy.eval["precision"][:, :, 0, 0, -1]
    values = values[values > -1]
    return float(values.mean()) if values.size else float("nan")


def main() -> None:
    args = parse_args()
    source = load_json(args.coco)
    predictions = load_json(args.predictions)
    category_names = {int(row["id"]): row["name"] for row in source["categories"]}
    stage_ids = sorted(category_id for category_id, name in category_names.items() if name in STAGE_NAMES)
    maize_u_ids = {category_id for category_id, name in category_names.items() if name == "maize-u"}
    if len(maize_u_ids) != 1 or len(stage_ids) != 3:
        raise ValueError(f"Expected one maize-u and three stage classes, found {category_names}")

    images = {int(row["id"]): row for row in source["images"]}
    ignore_masks = {
        image_id: np.zeros((int(row["height"]), int(row["width"])), dtype=bool)
        for image_id, row in images.items()
    }
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for annotation in source["annotations"]:
        image_id = int(annotation["image_id"])
        height, width = int(images[image_id]["height"]), int(images[image_id]["width"])
        category_id = int(annotation["category_id"])
        binary = mask_utils.decode(rle_from_annotation(annotation, height, width)).astype(bool)
        if category_id in maize_u_ids:
            ignore_masks[image_id] |= binary
            continue
        if category_id not in stage_ids:
            continue
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": serializable_rle(binary),
                "area": float(binary.sum()),
                "bbox": [float(value) for value in mask_utils.toBbox(serializable_rle(binary))],
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    clipped: list[dict[str, Any]] = []
    dropped_empty_predictions = 0
    clipped_prediction_pixels = 0
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        if int(prediction["category_id"]) not in stage_ids or image_id not in images:
            continue
        mask = mask_utils.decode(prediction["segmentation"]).astype(bool)
        before = int(mask.sum())
        mask &= ~ignore_masks[image_id]
        clipped_prediction_pixels += before - int(mask.sum())
        if not mask.any():
            dropped_empty_predictions += 1
            continue
        clipped.append(
            {
                "image_id": image_id,
                "category_id": int(prediction["category_id"]),
                "segmentation": serializable_rle(mask),
                "score": float(prediction["score"]),
            }
        )

    ground_truth = COCO()
    ground_truth.dataset = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [row for row in source["categories"] if int(row["id"]) in stage_ids],
    }
    ground_truth.createIndex()
    detections = ground_truth.loadRes(clipped) if clipped else COCO()
    evaluator = COCOeval(ground_truth, detections, "segm")
    evaluator.params.catIds = stage_ids
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    per_stage = {
        category_names[category_id]: category_ap(evaluator, category_id)
        for category_id in stage_ids
    }
    result = {
        "evaluator": "pycocotools.COCOeval segm",
        "target_policy": "maize2/maize4/maize6; maize-u pixels removed from predictions before COCO evaluation",
        "iou_range": "0.50:0.05:0.95",
        "max_detections_per_image": 100,
        "mask_ap": float(evaluator.stats[0]),
        "mask_ap50": float(evaluator.stats[1]),
        "mask_ap75": float(evaluator.stats[2]),
        "per_stage_mask_ap": per_stage,
        "source_predictions": len(predictions),
        "retained_predictions": len(clipped),
        "dropped_empty_predictions": dropped_empty_predictions,
        "prediction_pixels_removed_by_maize_u_ignore": clipped_prediction_pixels,
        "stage_ground_truth_instances": len(annotations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.export_clipped_predictions:
        args.export_clipped_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.export_clipped_predictions.write_text(json.dumps(clipped) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
