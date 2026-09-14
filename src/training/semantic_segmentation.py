from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# CuBLAS reads this before the first CUDA operation when deterministic mode is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Sampler

from datasets.coco_semantic import CocoSemanticDataset, load_coco, prepare_class_mapping
from nets.model_zoo import build_semantic_model as build_registered_semantic_model


COLOR_MAP = np.array(
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


def seed_everything(
    seed: int,
    deterministic: bool = True,
    strict_cuda: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # Strict mode fails instead of silently accepting a known
        # nondeterministic CUDA operation. It is deliberately slower than the
        # legacy AMP/Flash path and is intended for retained paper evidence.
        torch.use_deterministic_algorithms(True, warn_only=not strict_cuda)
        if torch.cuda.is_available():
            # The deterministic CuBLAS workspace is configured above before importing torch.
            # Disabling TF32 avoids precision-dependent trajectories between compatible GPUs.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if strict_cuda:
                torch.set_float32_matmul_precision("highest")
                for name, enabled in (
                    ("enable_flash_sdp", False),
                    ("enable_mem_efficient_sdp", False),
                    ("enable_math_sdp", True),
                ):
                    configure = getattr(torch.backends.cuda, name, None)
                    if configure is not None:
                        configure(enabled)


def seed_data_worker(worker_id: int) -> None:
    """Seed libraries used inside DataLoader workers from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class StableHashEpochSampler(Sampler[int]):
    """Deterministic, sample-ID based order for one training epoch.

    Unlike ``RandomSampler``, adding a new tile does not change the relative
    order of already-existing tiles.  The new tile can still fall between old
    tiles, so it is not an exact continuation of an older experiment; it is a
    controlled order that makes the source of any remaining change explicit.
    """

    def __init__(self, dataset: CocoSemanticDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.entries = [(str(image["file_name"]), index) for index, image in enumerate(dataset.images)]
        names = [name for name, _ in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("StableHashEpochSampler requires unique COCO file_name values.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        def order_key(entry: tuple[str, int]) -> tuple[bytes, str]:
            file_name, _ = entry
            payload = f"{self.seed}:{self.epoch}:{file_name}".encode("utf-8")
            return hashlib.blake2b(payload, digest_size=16).digest(), file_name

        return iter([index for _, index in sorted(self.entries, key=order_key)])

    def __len__(self) -> int:
        return len(self.entries)


def set_training_epoch(loader: DataLoader, epoch: int) -> None:
    """Synchronize the epoch used by deterministic sampling and augmentation."""
    dataset = loader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_coco_splits(
    dataset_root: Path,
    coco_by_split: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return content fingerprints needed to verify a semantic training input.

    The aggregate image hash includes the filename and hash of every image in
    the split. This catches a changed tile even when its COCO JSON is unchanged.
    """
    fingerprints: dict[str, dict[str, Any]] = {}
    for split, coco in coco_by_split.items():
        split_root = dataset_root / split
        annotations_path = split_root / "_annotations.coco.json"
        image_digest = hashlib.sha256()
        images = sorted(coco["images"], key=lambda image: str(image["file_name"]))
        for image in images:
            file_name = str(image["file_name"])
            image_path = split_root / file_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Cannot fingerprint missing COCO image: {image_path}")
            image_digest.update(file_name.encode("utf-8"))
            image_digest.update(b"\0")
            image_digest.update(sha256_file(image_path).encode("ascii"))
            image_digest.update(b"\n")
        fingerprints[split] = {
            "annotations_sha256": sha256_file(annotations_path),
            "images_sha256": image_digest.hexdigest(),
            "image_count": len(images),
        }
    return fingerprints


def colorize(mask: np.ndarray) -> np.ndarray:
    return COLOR_MAP[np.clip(mask, 0, len(COLOR_MAP) - 1)]


def update_confusion(
    confusion: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    pred = pred.view(-1).detach().cpu()
    target = target.view(-1).detach().cpu()
    valid = (target >= 0) & (target < num_classes)
    inds = num_classes * target[valid] + pred[valid]
    confusion += torch.bincount(
        inds,
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return confusion


def iou_from_confusion(confusion: torch.Tensor) -> torch.Tensor:
    confusion = confusion.float()
    true_positive = torch.diag(confusion)
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    return true_positive / (true_positive + false_positive + false_negative + 1e-7)


def dice_from_confusion(confusion: torch.Tensor) -> torch.Tensor:
    confusion = confusion.float()
    true_positive = torch.diag(confusion)
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    return (2 * true_positive) / (2 * true_positive + false_positive + false_negative + 1e-7)


def build_semantic_model(
    model_name: str,
    num_classes: int,
    norm: str = "gn",
    pretrained_backbone: bool = False,
    input_channels: int = 3,
) -> nn.Module:
    return build_registered_semantic_model(
        model_name=model_name,
        num_classes=num_classes,
        norm=norm,
        pretrained_backbone=pretrained_backbone,
        input_channels=input_channels,
    )


def logits_from_output(output: Any, target_hw: tuple[int, int]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, dict) and "logits" in output:
        logits = output["logits"]
    elif hasattr(output, "logits"):
        logits = output.logits
    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")

    if logits.shape[-2:] != target_hw:
        logits = torch.nn.functional.interpolate(
            logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
    return logits


def build_dataloaders(
    dataset_root: Path,
    task_mode: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    color_profile: str,
    augment_profile: str,
    input_features: str,
    device: torch.device,
    seed: int,
    reproducibility_mode: str = "legacy",
    drop_last_train: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, DataLoader]]:
    stable_modes = {"stable_hash", "strict"}
    if reproducibility_mode not in {"legacy", *stable_modes}:
        raise ValueError(
            "Unsupported reproducibility_mode: "
            f"{reproducibility_mode}. Expected legacy, stable_hash, or strict."
        )
    coco_by_split = {split: load_coco(dataset_root, split) for split in ["train", "valid", "test"]}
    class_info = prepare_class_mapping(coco_by_split["train"], task_mode)
    category_to_mask = class_info["category_to_mask"]
    category_id_to_name = class_info["category_id_to_name"]

    datasets = {
        "train": CocoSemanticDataset(
            dataset_root,
            "train",
            coco_by_split["train"],
            category_to_mask,
            category_id_to_name,
            augment=True,
            augment_profile=augment_profile,
            img_size=img_size,
            color_profile=color_profile,
            input_features=input_features,
            augment_seed=(seed if reproducibility_mode in stable_modes else None),
        ),
        "valid": CocoSemanticDataset(
            dataset_root,
            "valid",
            coco_by_split["valid"],
            category_to_mask,
            category_id_to_name,
            augment=False,
            img_size=img_size,
            color_profile=color_profile,
            input_features=input_features,
        ),
        "test": CocoSemanticDataset(
            dataset_root,
            "test",
            coco_by_split["test"],
            category_to_mask,
            category_id_to_name,
            augment=False,
            img_size=img_size,
            color_profile=color_profile,
            input_features=input_features,
        ),
    }

    workers = 0 if device.type in {"cpu", "mps"} or reproducibility_mode == "strict" else num_workers
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": (device.type == "cuda"),
    }
    if reproducibility_mode in stable_modes:
        train_sampler = StableHashEpochSampler(datasets["train"], seed)
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=drop_last_train,
            worker_init_fn=seed_data_worker,
            generator=torch.Generator().manual_seed(seed),
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

    loaders = {
        "train": train_loader,
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    }
    return coco_by_split, class_info, loaders


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_classes: int,
    train: bool,
    mixed_precision: bool = True,
    log_interval: int = 0,
    phase: str = "",
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    model.train(train)
    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    start = time.time()

    total_batches = len(loader)
    for batch_index, (images, masks, _) in enumerate(loader, start=1):
        should_log = log_interval > 0 and (
            batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches
        )
        if should_log:
            print(
                f"[phase] {phase} batch {batch_index}/{total_batches}: start",
                flush=True,
            )
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(
                "cuda",
                enabled=(device.type == "cuda" and mixed_precision),
            ):
                output = model(images)
                logits = logits_from_output(output, masks.shape[-2:])
                loss = criterion(logits, masks)

            if train and optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        confusion = update_confusion(confusion, logits.argmax(dim=1), masks, num_classes)
        if should_log:
            print(
                f"[phase] {phase} batch {batch_index}/{total_batches}: complete",
                flush=True,
            )

    avg_loss = total_loss / len(loader.dataset)
    ious = iou_from_confusion(confusion)
    dices = dice_from_confusion(confusion)
    return (
        avg_loss,
        ious.numpy(),
        dices.numpy(),
        float(ious.mean().item()),
        time.time() - start,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_miou: float,
    config: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_miou": best_miou,
        **config,
    }
    torch.save(payload, path)


def save_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    title: str,
) -> None:
    output_dir = output_path.with_suffix("") if output_path.suffix else output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    images, masks, names = next(iter(loader))
    with torch.no_grad():
        output = model(images.to(device))
        logits = logits_from_output(output, masks.shape[-2:])
        preds = logits.argmax(dim=1).cpu().numpy()

    images_np = images.permute(0, 2, 3, 1).numpy()
    masks_np = masks.numpy()
    sample_count = min(4, images_np.shape[0])

    for index in range(sample_count):
        stem = Path(names[index]).stem
        rgb = np.clip(images_np[index] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(output_dir / f"{index:02d}_{stem}_rgb.png")
        Image.fromarray(colorize(masks_np[index])).save(output_dir / f"{index:02d}_{stem}_gt.png")
        Image.fromarray(colorize(preds[index])).save(output_dir / f"{index:02d}_{stem}_pred_{title}.png")

    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Sample predictions",
                "",
                "Files are saved separately instead of a fixed panel:",
                "",
                "- `*_rgb.png`: input RGB image after resizing/color profile.",
                "- `*_gt.png`: colorized ground-truth semantic mask.",
                f"- `*_pred_{title}.png`: colorized prediction mask.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_metrics_header(metrics_path: Path, class_names: list[str]) -> None:
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "valid_loss",
                "valid_miou",
                "valid_mdice",
                *[f"iou_{name}" for name in class_names],
                *[f"dice_{name}" for name in class_names],
                "seconds",
            ]
        )


def append_metrics_row(
    metrics_path: Path,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    valid_miou: float,
    valid_dices: np.ndarray,
    valid_ious: np.ndarray,
    seconds: float,
) -> None:
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                train_loss,
                valid_loss,
                valid_miou,
                float(valid_dices.mean()),
                *valid_ious.tolist(),
                *valid_dices.tolist(),
                seconds,
            ]
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
