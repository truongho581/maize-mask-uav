#!/usr/bin/env python3
"""Download immutable pretrained inputs and verify complete SHA-256 digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=root / ".cache")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_download(url: str, destination: Path, expected: str, check_only: bool) -> None:
    if destination.is_file() and sha256(destination) == expected:
        print(f"verified cached: {destination}")
        return
    if check_only:
        raise FileNotFoundError(f"Missing or invalid cached weight: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    print(f"downloading: {url}")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink()
        raise ValueError(f"SHA-256 mismatch for {url}: expected {expected}, got {actual}")
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    lock = json.loads((root / "environment.lock.json").read_text(encoding="utf-8"))
    sources = lock["pretrained_sources"]
    cache = args.cache_root.resolve()

    torch_weight = sources["maskrcnn_resnet50_fpn_coco"]
    maskrcnn_path = cache / "torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    checked_download(torch_weight["url"], maskrcnn_path, torch_weight["sha256"], args.check_only)

    m2f_weight = sources["mask2former_r50_coco_instance"]
    mask2former_path = cache / "weights/mask2former_r50_model_final_3c8ec9.pkl"
    checked_download(m2f_weight["url"], mask2former_path, m2f_weight["sha256"], args.check_only)

    segformer = sources["segformer_b0_encoder"]
    segformer_path = cache / "weights/nvidia_mit_b0"
    expected_files = {
        "config.json": segformer["config_sha256"],
        "pytorch_model.bin": segformer["pytorch_model_sha256"],
    }
    valid = all((segformer_path / name).is_file() and sha256(segformer_path / name) == digest for name, digest in expected_files.items())
    if not valid:
        if args.check_only:
            raise FileNotFoundError(f"Missing or invalid pinned SegFormer snapshot: {segformer_path}")
        from huggingface_hub import snapshot_download

        segformer_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=segformer["repository"],
            revision=segformer["revision"],
            local_dir=segformer_path,
            allow_patterns=list(expected_files),
        )
    for name, expected in expected_files.items():
        actual = sha256(segformer_path / name)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for SegFormer {name}: expected {expected}, got {actual}")

    report = {
        "status": "PASS",
        "cache_root": str(cache),
        "environment": {
            "TORCH_HOME": str(cache / "torch"),
            "MAIZEMASK_SEGFORMER_B0_SOURCE": str(segformer_path),
            "MAIZEMASK_SEGFORMER_B0_REVISION": segformer["revision"],
            "MAIZEMASK_MASK2FORMER_R50_WEIGHTS": str(mask2former_path),
        },
        "weights": {
            "maskrcnn": {"path": str(maskrcnn_path), "sha256": sha256(maskrcnn_path)},
            "mask2former": {"path": str(mask2former_path), "sha256": sha256(mask2former_path)},
            "segformer_config": {"path": str(segformer_path / "config.json"), "sha256": sha256(segformer_path / "config.json")},
            "segformer_model": {"path": str(segformer_path / "pytorch_model.bin"), "sha256": sha256(segformer_path / "pytorch_model.bin")},
        },
    }
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "PRETRAINED_WEIGHTS_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
