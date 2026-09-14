from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.coco_semantic import annotation_to_binary_mask


STAGE_CLASS_NAMES = ["background", "maize2", "maize4", "maize6"]


def load_coco(dataset_root: Path, split: str) -> dict:
    annotation_path = Path(dataset_root) / split / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation not found: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_stage_category_mapping(coco: dict) -> tuple[dict[int, int], dict[int, str]]:
    category_id_to_name = {category["id"]: category["name"] for category in coco["categories"]}
    stage_name_to_label = {"maize2": 1, "maize4": 2, "maize6": 3}
    category_to_label = {
        category_id: stage_name_to_label[name]
        for category_id, name in category_id_to_name.items()
        if name in stage_name_to_label
    }
    return category_to_label, category_id_to_name


def build_annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    mapping = {image["id"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        mapping.setdefault(annotation["image_id"], []).append(annotation)
    return mapping


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


class CocoMaizeInstanceDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        coco: dict,
        category_to_label: dict[int, int],
        augment: bool = False,
        img_size: int = 640,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.coco = coco
        self.category_to_label = category_to_label
        self.augment = augment
        self.img_size = img_size
        self.image_dir = self.dataset_root / split
        self.images = sorted(coco["images"], key=lambda item: item["id"])
        self.annotations_by_image = build_annotations_by_image(coco)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        info = self.images[index]
        image_path = self.image_dir / info["file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        masks = []
        labels = []
        boxes = []
        areas = []

        for annotation in self.annotations_by_image.get(info["id"], []):
            category_id = annotation["category_id"]
            if category_id not in self.category_to_label:
                continue
            mask = annotation_to_binary_mask(annotation, height, width)
            box = mask_to_box(mask)
            if box is None:
                continue
            masks.append(mask.astype(np.uint8))
            labels.append(self.category_to_label[category_id])
            boxes.append(box)
            areas.append(float(mask.sum()))

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        if masks:
            mask_arr = np.stack(masks, axis=0)
        else:
            mask_arr = np.zeros((0, height, width), dtype=np.uint8)
        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels_arr = np.asarray(labels, dtype=np.int64)
        areas_arr = np.asarray(areas, dtype=np.float32)

        if image.size != (self.img_size, self.img_size):
            scale_x = self.img_size / width
            scale_y = self.img_size / height
            image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            image_arr = np.asarray(image, dtype=np.float32) / 255.0
            resized_masks = []
            for mask in mask_arr:
                mask_img = Image.fromarray(mask)
                resized_masks.append(
                    np.asarray(
                        mask_img.resize(
                            (self.img_size, self.img_size),
                            Image.Resampling.NEAREST,
                        ),
                        dtype=np.uint8,
                    )
                )
            mask_arr = (
                np.stack(resized_masks, axis=0)
                if resized_masks
                else np.zeros((0, self.img_size, self.img_size), dtype=np.uint8)
            )
            if len(boxes_arr):
                boxes_arr[:, [0, 2]] *= scale_x
                boxes_arr[:, [1, 3]] *= scale_y
            areas_arr = np.asarray([float(mask.sum()) for mask in mask_arr], dtype=np.float32)

        if self.augment and random.random() < 0.5:
            image_arr = np.ascontiguousarray(np.flip(image_arr, axis=1))
            if len(mask_arr):
                mask_arr = np.ascontiguousarray(np.flip(mask_arr, axis=2))
                x1 = boxes_arr[:, 0].copy()
                x2 = boxes_arr[:, 2].copy()
                boxes_arr[:, 0] = self.img_size - x2
                boxes_arr[:, 2] = self.img_size - x1

        target = {
            "boxes": torch.as_tensor(boxes_arr, dtype=torch.float32),
            "labels": torch.as_tensor(labels_arr, dtype=torch.int64),
            "masks": torch.as_tensor(mask_arr, dtype=torch.uint8),
            "image_id": torch.tensor([info["id"]], dtype=torch.int64),
            "area": torch.as_tensor(areas_arr, dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels_arr),), dtype=torch.int64),
        }
        image_tensor = torch.from_numpy(image_arr.transpose(2, 0, 1)).float()
        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
