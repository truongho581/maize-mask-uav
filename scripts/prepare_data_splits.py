#!/usr/bin/env python3
"""Create loader-compatible LOSO/fixed trees without duplicating image bytes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=root / ".runtime_data")
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "symlink", "copy"), default="auto")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def safe_file_name(value: str) -> str:
    path = Path(value)
    if path.name != value or path.is_absolute() or value in {"", ".", ".."}:
        raise ValueError(f"Unsafe COCO file_name: {value!r}")
    return value


def place_image(source: Path, destination: Path, mode: str) -> str:
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    if mode in {"auto", "symlink"}:
        try:
            destination.symlink_to(source.resolve())
            return "symlink"
        except OSError:
            if mode == "symlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def materialize_partition(annotation: Path, images_root: Path, destination: Path, mode: str) -> dict:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(annotation, destination / "_annotations.coco.json")
    methods: dict[str, int] = {}
    for image in payload["images"]:
        name = safe_file_name(str(image["file_name"]))
        source = images_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        used = place_image(source, destination / name, mode)
        methods[used] = methods.get(used, 0) + 1
    return {"images": len(payload["images"]), "annotations": len(payload["annotations"]), "methods": methods}


def verify_partition(annotation: Path, destination: Path) -> dict:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    copied = json.loads((destination / "_annotations.coco.json").read_text(encoding="utf-8"))
    if payload != copied:
        raise ValueError(f"Materialized annotation differs: {destination}")
    missing = [item["file_name"] for item in payload["images"] if not (destination / item["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images in {destination}: {missing[:3]}")
    return {"images": len(payload["images"]), "annotations": len(payload["annotations"])}


def main() -> None:
    args = parse_args()
    release = args.release_root.resolve()
    output = args.output_root.resolve()
    plans: list[tuple[Path, Path]] = []
    for fold in ("test_D1", "test_D2", "test_D3"):
        for split in ("train", "valid", "test"):
            plans.append((release / "annotations/loso" / fold / f"{split}.json", output / "loso" / fold / split))
    for split in ("train", "valid", "test"):
        plans.append((release / "annotations/standard" / f"{split}.json", output / "standard" / split))

    report = {}
    if args.verify_only:
        for annotation, destination in plans:
            report[str(destination.relative_to(output))] = verify_partition(annotation, destination)
    else:
        if output.exists():
            raise FileExistsError(
                f"Refusing to merge into existing runtime tree: {output}. "
                "Delete only that generated directory or choose a new --output-root."
            )
        output.mkdir(parents=True)
        try:
            for annotation, destination in plans:
                report[str(destination.relative_to(output))] = materialize_partition(
                    annotation, release / "images", destination, args.link_mode
                )
        except Exception:
            shutil.rmtree(output)
            raise

    target = output / "MATERIALIZATION_REPORT.json"
    target.write_text(json.dumps({"status": "PASS", "partitions": report}, indent=2) + "\n", encoding="utf-8")
    print(f"Materialized and verified {len(plans)} partitions at {output}")


if __name__ == "__main__":
    main()
