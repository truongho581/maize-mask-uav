import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

try:
    from pycocotools import mask as coco_mask
except ImportError as exc:
    raise ImportError(
        "pycocotools is required for COCO polygon/RLE masks. "
        "Install it in RunPod with: pip install pycocotools"
    ) from exc


COLOR_PROFILES = {
    "dji-jpg-mean-std": {
        "source_mean_rgb": [104.43135561342592, 96.65664556100218, 76.06031377655229],
        "target_mean_rgb": [141.28682683142702, 126.38719626565904, 84.66497004357298],
        "scale_rgb": [1.2249357978160198, 1.3787848001427505, 1.7405787640974035],
        "strength": 0.85,
        "sharpen_radius": 0.65,
        "sharpen_percent": 22,
        "sharpen_threshold": 5,
    }
}


# These profiles preserve the canonical RGB image.  Derived channels are added
# only to the tensor presented to a model, after geometric augmentation.
INPUT_FEATURE_PROFILES = {
    "rgb": 3,
    "rgb_exg": 4,
    "rgb_exg_vari": 5,
}


# This is deliberately data-only metadata: it mirrors ``apply_augmentation``
# below so every run can record the concrete meaning of a profile name.
SEMANTIC_AUGMENTATION_PROFILES = {
    "none": {
        "horizontal_flip_probability": 0.0,
        "vertical_flip_probability": 0.0,
        "rotation_degrees": [0],
        "brightness_probability": 0.0,
        "brightness_scale_range": None,
    },
    "basic": {
        "horizontal_flip_probability": 0.5,
        "vertical_flip_probability": 0.5,
        "rotation_degrees": [0, 90, 180, 270],
        "brightness_probability": 0.5,
        "brightness_scale_range": [0.85, 1.15],
    },
}


def augmentation_profile_metadata(profile_name: str) -> dict:
    """Return a JSON-safe description of a semantic augmentation profile."""
    try:
        return dict(SEMANTIC_AUGMENTATION_PROFILES[profile_name])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported augment_profile: {profile_name}. "
            f"Expected one of {sorted(SEMANTIC_AUGMENTATION_PROFILES)}"
        ) from exc


def input_feature_channels(profile_name: str) -> int:
    try:
        return INPUT_FEATURE_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported input_features: {profile_name}. "
            f"Expected one of {sorted(INPUT_FEATURE_PROFILES)}"
        ) from exc


def append_vegetation_features(rgb: np.ndarray, profile_name: str) -> np.ndarray:
    """Append scale-invariant visible-RGB vegetation indices to an RGB tensor.

    ``rgb`` must be float32 in [0, 1], HxWx3.  ExG follows the normalized-RGB
    formulation ``2g-r-b``.  Both indices are clipped and linearly mapped to
    [0, 1] so the extra channels remain numerically compatible with RGB.
    """
    input_feature_channels(profile_name)
    if profile_name == "rgb":
        return rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB input, got {rgb.shape}")

    denominator = np.maximum(rgb.sum(axis=2, keepdims=True), 1e-6)
    normalized = rgb / denominator
    red, green, blue = (normalized[:, :, channel] for channel in range(3))
    exg = 2.0 * green - red - blue
    features = [rgb, np.clip((exg + 1.0) / 3.0, 0.0, 1.0)[:, :, None]]

    if profile_name == "rgb_exg_vari":
        vari_denominator = np.where(
            np.abs(green + red - blue) < 1e-6,
            1e-6,
            green + red - blue,
        )
        vari = (green - red) / vari_denominator
        features.append(np.clip((vari + 1.0) / 2.0, 0.0, 1.0)[:, :, None])

    return np.ascontiguousarray(np.concatenate(features, axis=2), dtype=np.float32)


def load_coco(dataset_root, split):
    annotation_path = Path(dataset_root) / split / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation not found: {annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_class_mapping(train_coco, task_mode):
    categories = sorted(
        [category for category in train_coco["categories"] if category["id"] != 0],
        key=lambda category: category["id"],
    )
    raw_category_names = [category["name"] for category in categories]
    category_id_to_name = {category["id"]: category["name"] for category in categories}

    if task_mode == "s3":
        class_names = ["background", "crop", "weed"]
        category_to_mask = {
            category_id: (2 if name == "weed" else 1)
            for category_id, name in category_id_to_name.items()
            if name == "weed" or name.startswith("maize")
        }
    elif task_mode == "raw":
        class_names = ["background", *raw_category_names]
        category_to_mask = {
            category["id"]: index + 1 for index, category in enumerate(categories)
        }
    else:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    return {
        "categories": categories,
        "raw_category_names": raw_category_names,
        "category_id_to_name": category_id_to_name,
        "class_names": class_names,
        "category_to_mask": category_to_mask,
    }


def build_annotations_by_image(coco):
    mapping = {image["id"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        mapping.setdefault(annotation["image_id"], []).append(annotation)
    return mapping


def annotation_to_binary_mask(annotation, height, width):
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return np.zeros((height, width), dtype=bool)

    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
        decoded = coco_mask.decode(rle)
    elif isinstance(segmentation, dict):
        decoded = coco_mask.decode(segmentation)
    else:
        raise TypeError(f"Unsupported COCO segmentation type: {type(segmentation)}")

    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def apply_color_profile(rgb, profile_name):
    if profile_name in {None, "none"}:
        return rgb
    if profile_name not in COLOR_PROFILES:
        raise ValueError(f"Unsupported color profile: {profile_name}")

    profile = COLOR_PROFILES[profile_name]
    source = rgb.astype(np.float32)
    source_mean = np.asarray(profile["source_mean_rgb"], dtype=np.float32)
    target_mean = np.asarray(profile["target_mean_rgb"], dtype=np.float32)
    scale = np.asarray(profile["scale_rgb"], dtype=np.float32)
    strength = float(profile["strength"])

    mapped = (source - source_mean) * scale + target_mean
    blended = source * (1.0 - strength) + mapped * strength
    output = np.clip(blended, 0, 255).astype(np.uint8)

    radius = float(profile.get("sharpen_radius", 0))
    percent = int(profile.get("sharpen_percent", 0))
    threshold = int(profile.get("sharpen_threshold", 0))
    if radius > 0 and percent > 0:
        image = Image.fromarray(output, mode="RGB")
        y_channel, cb_channel, cr_channel = image.convert("YCbCr").split()
        y_channel = y_channel.filter(
            ImageFilter.UnsharpMask(
                radius=radius,
                percent=percent,
                threshold=threshold,
            )
        )
        output = np.asarray(
            Image.merge("YCbCr", (y_channel, cb_channel, cr_channel)).convert("RGB"),
            dtype=np.uint8,
        )

    return output


class CocoSemanticDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        coco,
        category_to_mask,
        category_id_to_name,
        augment=False,
        augment_profile="basic",
        img_size=640,
        color_profile="none",
        input_features="rgb",
        augment_seed: int | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.coco = coco
        self.category_to_mask = category_to_mask
        self.category_id_to_name = category_id_to_name
        self.augment = augment
        self.augment_profile = augment_profile
        self.img_size = img_size
        self.color_profile = color_profile
        self.input_features = input_features
        self.augment_seed = augment_seed
        self.epoch = 0
        input_feature_channels(input_features)
        self.image_dir = self.dataset_root / split
        self.images = sorted(coco["images"], key=lambda item: item["id"])
        self.annotations_by_image = build_annotations_by_image(coco)

    def __len__(self):
        return len(self.images)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by order-independent deterministic augmentation."""
        self.epoch = int(epoch)

    def augmentation_rng(self, file_name: str) -> random.Random | None:
        if self.augment_seed is None:
            return None
        payload = f"{self.augment_seed}:{self.epoch}:{file_name}".encode("utf-8")
        seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        return random.Random(seed)

    def __getitem__(self, index):
        info = self.images[index]
        image_path = self.image_dir / info["file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)

        annotations = self.annotations_by_image.get(info["id"], [])
        annotations = sorted(
            annotations,
            key=lambda annotation: (
                1
                if self.category_id_to_name.get(annotation["category_id"]) == "weed"
                else 0
            ),
        )

        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id not in self.category_to_mask:
                continue
            binary = annotation_to_binary_mask(annotation, height, width)
            mask[binary] = self.category_to_mask[category_id]

        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            mask_image = Image.fromarray(mask)
            mask = mask_image.resize(
                (self.img_size, self.img_size), Image.Resampling.NEAREST
            )
            mask = np.array(mask, dtype=np.uint8)

        image = np.array(image, dtype=np.uint8)
        image = apply_color_profile(image, self.color_profile)
        image = image.astype(np.float32) / 255.0

        if self.augment:
            image, mask = self.apply_augmentation(
                image,
                mask,
                self.augment_profile,
                rng=self.augmentation_rng(info["file_name"]),
            )

        image = append_vegetation_features(image, self.input_features)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor, info["file_name"]

    @staticmethod
    def apply_augmentation(image, mask, profile="basic", rng: random.Random | None = None):
        if profile in {None, "none"}:
            return image, mask

        rng = rng or random

        if rng.random() < 0.5:
            image = np.ascontiguousarray(np.flip(image, axis=1))
            mask = np.ascontiguousarray(np.flip(mask, axis=1))
        if rng.random() < 0.5:
            image = np.ascontiguousarray(np.flip(image, axis=0))
            mask = np.ascontiguousarray(np.flip(mask, axis=0))

        rotations = rng.randint(0, 3)
        if rotations:
            image = np.ascontiguousarray(np.rot90(image, rotations, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, rotations, axes=(0, 1)))

        if rng.random() < 0.5:
            image = np.clip(image * rng.uniform(0.85, 1.15), 0.0, 1.0)

        if profile == "basic":
            return image, mask

        raise ValueError(f"Unsupported augment_profile: {profile}")
