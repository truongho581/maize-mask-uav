"""Train official Mask2Former R50 on MaizeMask COCO/RLE instance targets.

This runner deliberately uses the upstream Detectron2 Mask2Former code copied into
the RunPod bundle.  The upstream ``MaskFormerInstanceDatasetMapper`` natively
decodes COCO RLE masks, so disconnected components of one annotation remain one
binary target rather than being converted into separate polygons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "Mask2Former"
DETECTRON2_ROOT = PROJECT_ROOT / "third_party" / "detectron2"
STAGE_NAMES = ("maize2", "maize4", "maize6")
COCO_INSTANCE_R50_WEIGHTS = os.environ.get(
    "MAIZEMASK_MASK2FORMER_R50_WEIGHTS",
    (
        "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
        "maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate official Mask2Former R50 with native COCO/RLE instances."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-key", default="v1.0")
    parser.add_argument("--protocol", choices=("loso", "fixed"), default="loso")
    parser.add_argument("--folds", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--checkpoint-period-epochs",
        type=int,
        default=50,
        help=(
            "Save resumable periodic checkpoints every N epochs. The default keeps only "
            "the final periodic checkpoint; the validation-selected model_best.pth is "
            "always saved independently."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--augment-profile", choices=("horizontal_flip", "none"), default="horizontal_flip")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted Mask2Former run from its latest periodic checkpoint.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run one real 640-pixel forward/backward batch, then stop before training.",
    )
    parser.add_argument("--evaluate-checkpoint", type=Path, default=None)
    parser.add_argument("--evaluation-split", choices=("valid", "test"), default="test")
    return parser.parse_args()


def import_upstream() -> None:
    for path in (UPSTREAM_ROOT, DETECTRON2_ROOT):
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing bundled upstream source: {path}. Rebuild the RunPod bundle."
            )
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_coco(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_stage_only_coco(source: Path, destination: Path) -> tuple[int, int]:
    """Keep RLE payloads verbatim while retaining only stage-specific maize targets."""
    payload = load_coco(source)
    selected_categories = [item for item in payload["categories"] if item["name"] in STAGE_NAMES]
    selected_ids = {item["id"] for item in selected_categories}
    annotations = [
        item for item in payload["annotations"] if item["category_id"] in selected_ids
    ]
    target = {
        **payload,
        "categories": selected_categories,
        "annotations": annotations,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(target, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(target["images"]), len(annotations)


def fold_names(dataset_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("test_D"))


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    from detectron2.utils.env import seed_all_rng

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_all_rng(seed)


def register_fold_datasets(fold_root: Path, cache_root: Path, fold: str):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    names: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        source = fold_root / split / "_annotations.coco.json"
        if not source.exists():
            raise FileNotFoundError(f"Missing COCO split: {source}")
        annotation = cache_root / fold / f"{split}_stage_only.coco.json"
        images, annotations = write_stage_only_coco(source, annotation)
        name = f"maizemask_{fold}_{split}_stage_r50"
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        register_coco_instances(name, {}, str(annotation), str(fold_root / split))
        MetadataCatalog.get(name).thing_classes = list(STAGE_NAMES)
        names[split] = name
        print(f"{split}: {images} images, {annotations} retained stage instances")
    return names


def result_directory(output_root: Path, dataset_key: str, protocol: str, fold: str, seed: int) -> Path:
    return (
        output_root / "instance" / "mask2former_r50" / dataset_key / protocol / fold /
        f"seed_{seed}"
    )


def extract_segm_ap(results: dict[str, Any]) -> float:
    direct_segm = results.get("segm")
    if isinstance(direct_segm, dict):
        ap = direct_segm.get("AP")
        if ap is not None:
            return float(ap)
    for value in results.values():
        if isinstance(value, dict) and isinstance(value.get("segm"), dict):
            ap = value["segm"].get("AP")
            if ap is not None:
                return float(ap)
    raise KeyError(f"Could not find validation segm AP in evaluator output: {results}")


def build_cfg(args: argparse.Namespace, names: dict[str, str], run_dir: Path, train_images: int):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    # Match the upstream Mask2Former setup order before merging its official YAML.
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(
        str(UPSTREAM_ROOT / "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    )
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(STAGE_NAMES)
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    # COCO AP ranks all detections; do not discard them with the upstream demo
    # confidence gate before the evaluator receives the predictions.
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.WEIGHTS = COCO_INSTANCE_R50_WEIGHTS if args.pretrained else ""
    cfg.DATASETS.TRAIN = (names["train"],)
    cfg.DATASETS.TEST = (names["valid"],)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_instance"
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TRAIN = (args.img_size,)
    cfg.INPUT.MAX_SIZE_TRAIN = args.img_size
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = args.img_size
    cfg.INPUT.MAX_SIZE_TEST = args.img_size
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.RANDOM_FLIP = "horizontal" if args.augment_profile == "horizontal_flip" else "none"
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.MAX_ITER = math.ceil(train_images / args.batch_size) * args.epochs
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_ITERS = min(10, max(0, math.ceil(train_images / args.batch_size) - 1))
    iterations_per_epoch = math.ceil(train_images / args.batch_size)
    cfg.SOLVER.CHECKPOINT_PERIOD = min(
        cfg.SOLVER.MAX_ITER,
        iterations_per_epoch * args.checkpoint_period_epochs,
    )
    cfg.TEST.EVAL_PERIOD = math.ceil(train_images / args.batch_size)
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.OUTPUT_DIR = str(run_dir)
    cfg.SEED = args.seed
    cfg.freeze()
    return cfg


def make_trainer(cfg, selection_path: Path):
    import torch
    from detectron2.engine import hooks
    from detectron2.evaluation import COCOEvaluator
    from train_net import Trainer as UpstreamTrainer

    class BestValidationHook(hooks.EvalHook):
        def __init__(self, period, eval_function, checkpointer):
            super().__init__(period, eval_function)
            self.checkpointer = checkpointer
            self.best_ap = float("-inf")

        def _do_eval(self):
            super()._do_eval()
            results = self.trainer._last_eval_results
            ap = extract_segm_ap(results)
            if ap > self.best_ap:
                self.best_ap = ap
                self.checkpointer.save("model_best", iteration=self.trainer.iter + 1)
                selection_path.write_text(
                    json.dumps(
                        {
                            "checkpoint": "model_best.pth",
                            "iteration": self.trainer.iter + 1,
                            "validation_mask_ap": ap,
                            "criterion": "official COCOEvaluator segm AP on validation",
                        },
                        indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )

    class MaizeMaskTrainer(UpstreamTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, tasks=("segm",), output_dir=output_folder)

        def build_hooks(self):
            built = super().build_hooks()
            for index, hook in enumerate(built):
                if isinstance(hook, hooks.EvalHook):
                    built[index] = BestValidationHook(
                        cfg.TEST.EVAL_PERIOD,
                        hook._func,
                        self.checkpointer,
                    )
            return built

    return MaizeMaskTrainer(cfg)


def run_fold(args: argparse.Namespace, fold: str) -> None:
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.evaluation import inference_on_dataset

    fold_root = args.dataset_root / fold
    if not fold_root.is_dir():
        raise FileNotFoundError(f"Missing fold: {fold_root}")
    run_dir = result_directory(args.output_root, args.dataset_key, args.protocol, fold, args.seed)
    if args.skip_existing and (run_dir / "model_final.pth").exists() and not args.evaluate_checkpoint:
        print(f"Skipping existing complete run: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = run_dir / "stage_only_coco"
    names = register_fold_datasets(fold_root, cache_root, fold)
    train_images = len(load_coco(fold_root / "train/_annotations.coco.json")["images"])
    cfg = build_cfg(args, names, run_dir, train_images)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": "mask2former_r50",
                "model_weights": cfg.MODEL.WEIGHTS or None,
                "backbone": "ResNet-50",
                "input_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "iterations": cfg.SOLVER.MAX_ITER,
                "checkpoint_period_epochs": args.checkpoint_period_epochs,
                "optimizer": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "augmentation": args.augment_profile,
                "target_policy": "maize2/maize4/maize6 only; original RLE retained as one binary instance",
                "checkpoint_selection": "validation official COCO segm AP",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    if args.preview_only:
        print(f"Preview passed: {fold} -> {run_dir}")
        return
    trainer = make_trainer(cfg, run_dir / "best_validation.json")
    if args.smoke_only:
        trainer.model.train()
        batch = next(iter(trainer.build_train_loader(cfg)))
        losses = trainer.model(batch)
        total_loss = sum(losses.values())
        total_loss.backward()
        print(
            f"Smoke passed: batch={len(batch)} | loss={float(total_loss.detach().cpu()):.6f} | "
            f"terms={sorted(losses)}"
        )
        return
    if args.evaluate_checkpoint:
        checkpoint = args.evaluate_checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        DetectionCheckpointer(trainer.model).load(str(checkpoint))
        cfg_test = cfg.clone()
        cfg_test.defrost()
        cfg_test.DATASETS.TEST = (names[args.evaluation_split],)
        cfg_test.freeze()
        evaluator = trainer.build_evaluator(cfg_test, names[args.evaluation_split], str(run_dir / "post_eval" / args.evaluation_split))
        loader = trainer.build_test_loader(cfg_test, names[args.evaluation_split])
        results = inference_on_dataset(trainer.model, loader, evaluator)
        target = run_dir / "post_eval" / args.evaluation_split
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        return
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    if not (run_dir / "model_best.pth").exists():
        raise RuntimeError("Mask2Former training finished without a selected validation checkpoint.")


def main() -> None:
    args = parse_args()
    import_upstream()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Mask2Former R50 paper training requires CUDA.")
    seed_everything(args.seed)
    for fold in fold_names(args.dataset_root, args.folds):
        print(f"\n{'=' * 80}\nMask2Former-R50 | {fold} | seed {args.seed}\n{'=' * 80}")
        run_fold(args, fold)


if __name__ == "__main__":
    main()
