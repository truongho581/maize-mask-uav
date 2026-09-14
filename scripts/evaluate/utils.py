from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


SEMANTIC_COLORS = np.array(
    [
        [0, 0, 0],
        [34, 197, 94],
        [239, 68, 68],
        [132, 204, 22],
        [234, 179, 8],
        [59, 130, 246],
    ],
    dtype=np.uint8,
)

STAGE_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([132, 204, 22], dtype=np.uint8),
    2: np.array([34, 197, 94], dtype=np.uint8),
    3: np.array([234, 179, 8], dtype=np.uint8),
    255: np.array([168, 85, 247], dtype=np.uint8),
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def colorize_index_mask(mask: np.ndarray, colors: np.ndarray = SEMANTIC_COLORS) -> np.ndarray:
    clipped = np.clip(mask, 0, len(colors) - 1)
    return colors[clipped]


def colorize_stage_mask(mask: np.ndarray, ignore_mask: np.ndarray | None = None) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in STAGE_COLORS.items():
        if label == 255:
            continue
        output[mask == label] = color
    if ignore_mask is not None:
        output[ignore_mask] = STAGE_COLORS[255]
    return output


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def save_mask(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def semantic_error_map(gt: np.ndarray, pred: np.ndarray, weed_index: int | None) -> np.ndarray:
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)
    mismatch = gt != pred
    error[mismatch] = np.array([239, 68, 68], dtype=np.uint8)
    if weed_index is not None:
        correct_weed = (gt == weed_index) & (pred == weed_index)
        missed_weed = (gt == weed_index) & (pred != weed_index)
        false_weed = (gt != weed_index) & (pred == weed_index)
        error[correct_weed] = np.array([34, 197, 94], dtype=np.uint8)
        error[missed_weed] = np.array([59, 130, 246], dtype=np.uint8)
        error[false_weed] = np.array([239, 68, 68], dtype=np.uint8)
    return error


def stage_error_map(gt: np.ndarray, pred: np.ndarray, ignore_mask: np.ndarray | None = None) -> np.ndarray:
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = np.ones(gt.shape, dtype=bool) if ignore_mask is None else ~ignore_mask
    gt_maize = (gt > 0) & valid
    pred_maize = (pred > 0) & valid
    correct = gt_maize & pred_maize
    missed = gt_maize & ~pred_maize
    false = ~gt_maize & pred_maize & valid
    stage_mismatch = correct & (gt != pred)
    error[correct] = np.array([34, 197, 94], dtype=np.uint8)
    error[missed] = np.array([59, 130, 246], dtype=np.uint8)
    error[false] = np.array([239, 68, 68], dtype=np.uint8)
    error[stage_mismatch] = np.array([234, 179, 8], dtype=np.uint8)
    if ignore_mask is not None:
        error[ignore_mask] = np.array([168, 85, 247], dtype=np.uint8)
    return error


def metric_from_counts(intersection: float, union: float, pred_sum: float, target_sum: float) -> dict[str, float]:
    return {
        "iou": float(intersection / (union + 1e-7)),
        "dice": float((2.0 * intersection) / (pred_sum + target_sum + 1e-7)),
        "precision": float(intersection / (pred_sum + 1e-7)),
        "recall": float(intersection / (target_sum + 1e-7)),
    }


class BinaryMetricAccumulator:
    def __init__(self, labels: Iterable[str]):
        self.intersections = Counter({label: 0.0 for label in labels})
        self.unions = Counter({label: 0.0 for label in labels})
        self.pred_sums = Counter({label: 0.0 for label in labels})
        self.target_sums = Counter({label: 0.0 for label in labels})

    def update(self, label: str, pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> None:
        pred = pred.astype(bool)
        target = target.astype(bool)
        if valid is not None:
            pred = pred & valid
            target = target & valid
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        self.intersections[label] += float(intersection)
        self.unions[label] += float(union)
        self.pred_sums[label] += float(pred.sum())
        self.target_sums[label] += float(target.sum())

    def metrics(self) -> dict[str, float]:
        output = {}
        for label in self.intersections:
            values = metric_from_counts(
                self.intersections[label],
                self.unions[label],
                self.pred_sums[label],
                self.target_sums[label],
            )
            for metric_name, value in values.items():
                output[f"{metric_name}_{label}"] = value
        return output


def count_connected_components(mask: np.ndarray, min_area: int = 8) -> int:
    binary = mask.astype(np.uint8)
    if binary.size == 0 or binary.max() == 0:
        return 0
    try:
        import cv2
    except ImportError:
        labeled, total = _count_components_numpy(binary)
        if total == 0:
            return 0
        return sum(1 for component_id in range(1, total + 1) if (labeled == component_id).sum() >= min_area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    count = 0
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            count += 1
    return count


def _count_components_numpy(binary: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(binary.shape, dtype=np.int32)
    component_id = 0
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            component_id += 1
            stack = [(y, x)]
            labels[y, x] = component_id
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = component_id
                            stack.append((ny, nx))
    return labels, component_id


def count_regression_metrics(gt_counts: list[int], pred_counts: list[int]) -> dict[str, float]:
    if not gt_counts:
        return {
            "visible_maize_count_mae": 0.0,
            "visible_maize_count_rmse": 0.0,
            "visible_maize_count_bias": 0.0,
            "visible_maize_count_corr": 0.0,
        }
    gt = np.asarray(gt_counts, dtype=np.float64)
    pred = np.asarray(pred_counts, dtype=np.float64)
    diff = pred - gt
    corr = 0.0
    if len(gt) > 1 and gt.std() > 0 and pred.std() > 0:
        corr = float(np.corrcoef(gt, pred)[0, 1])
    return {
        "visible_maize_count_mae": float(np.abs(diff).mean()),
        "visible_maize_count_rmse": float(np.sqrt(np.square(diff).mean())),
        "visible_maize_count_bias": float(diff.mean()),
        "visible_maize_count_corr": corr,
    }


def summarize_timing(total_seconds: float, images: int) -> dict[str, float]:
    seconds_per_image = total_seconds / max(images, 1)
    return {
        "seconds_total": float(total_seconds),
        "seconds_per_image": float(seconds_per_image),
        "fps": float(1.0 / seconds_per_image) if seconds_per_image > 0 else 0.0,
    }


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start
