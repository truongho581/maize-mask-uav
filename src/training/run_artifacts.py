from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import torch


DEFAULT_OUTPUT_ROOT = Path("results/local_runs")

# The public v1.0 release materializes loader-compatible split trees at runtime.
# Keep the registry empty so a manual invocation cannot silently select a
# private or stale dataset path; use --dataset-root from that materialization.
DATASET_REGISTRY: dict[str, dict[str, Path]] = {}


def resolve_path(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def resolve_protocol_dataset(
    project_root: Path,
    protocol: str,
    dataset_root: Path | None,
    dataset_version: str | None = None,
    legacy_loso_root: Path | None = None,
) -> Path:
    if dataset_root is not None and legacy_loso_root is not None:
        raise ValueError("Use only one of --dataset-root and --loso-root.")

    selected = dataset_root or legacy_loso_root
    if selected is not None and dataset_version is not None:
        raise ValueError(
            "Use --dataset-version for a registered dataset or --dataset-root/--loso-root "
            "for a custom dataset, not both."
        )
    if selected is None and dataset_version is None:
        raise ValueError(
            "Dataset selection is required. Run scripts/prepare_data_splits.py "
            "and provide its --dataset-root explicitly."
        )
    if selected is None:
        if dataset_version not in DATASET_REGISTRY:
            supported = ", ".join(sorted(DATASET_REGISTRY))
            raise ValueError(f"Unknown dataset version {dataset_version!r}; choose {supported}.")
        if protocol not in DATASET_REGISTRY[dataset_version]:
            raise ValueError(f"Unknown protocol {protocol!r} for dataset {dataset_version}.")
        selected = DATASET_REGISTRY[dataset_version][protocol]
    return resolve_path(project_root, selected)


def protocol_datasets(
    dataset_root: Path,
    protocol: str,
    folds: list[str] | None,
) -> list[tuple[str, Path]]:
    if protocol == "fixed":
        if folds and folds != ["fixed"]:
            raise ValueError("The fixed protocol accepts no folds, or --folds fixed.")
        datasets = [("fixed", dataset_root)]
    else:
        selected_folds = folds or ["test_D1", "test_D2", "test_D3"]
        datasets = [(fold, dataset_root / fold) for fold in selected_folds]

    for label, path in datasets:
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset for {label} does not exist: {path}")
        for split in ("train", "valid", "test"):
            if not (path / split).is_dir():
                raise FileNotFoundError(f"Missing {split} split in {path}")
    return datasets


def run_directory(
    output_root: Path,
    task: str,
    model: str,
    dataset: str,
    protocol: str,
    fold: str,
    seed: int,
) -> Path:
    return output_root / task / model / dataset / protocol / fold / f"seed_{seed}"


def prepare_run_directory(path: Path, overwrite: bool = False) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Run directory is not empty: {path}. Use --overwrite only when intentional."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_run_config(output_dir: Path, payload: dict[str, Any]) -> None:
    # JSON is valid YAML 1.2 and avoids adding PyYAML as a training dependency.
    write_json(output_dir / "config.yaml", payload)
    write_json(output_dir / "config.json", payload)


def write_command(output_dir: Path) -> None:
    command = shlex.join([sys.executable, *sys.argv])
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device.index or 0)

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "transformers": package_version("transformers"),
        "ultralytics": package_version("ultralytics"),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": gpu_name,
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
