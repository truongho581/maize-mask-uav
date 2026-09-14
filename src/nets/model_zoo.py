"""Model registry for the MaizeMask paper experiments.

This module centralizes the five model families used in the manuscript. Some
architectures are implemented locally, while others intentionally use official
library implementations to stay close to the reference models.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import torch.nn as nn


SEGFORMER_B0_PRETRAINED = os.environ.get(
    "MAIZEMASK_SEGFORMER_B0_SOURCE", "nvidia/mit-b0"
)
SEGFORMER_B0_REVISION = os.environ.get(
    "MAIZEMASK_SEGFORMER_B0_REVISION",
    "80983a413c30d36a39c20203974ae7807835e2b4",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    implementation: str
    reference: str
    notes: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "attention_unet": ModelSpec(
        name="Attention U-Net",
        task="semantic",
        implementation="src/nets/attention_unet/AttentionUNet",
        reference="Oktay et al., Attention U-Net: Learning Where to Look for the Pancreas, arXiv:1804.03999",
        notes="Local encoder-decoder with attention gates on skip connections.",
    ),
    "deeplabv3plus_resnet50": ModelSpec(
        name="DeepLabV3+ ResNet-50",
        task="semantic",
        implementation="src/nets/deeplabv3plus/modeling.py",
        reference="Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, ECCV 2018",
        notes="Local DeepLabV3+ implementation with ResNet-50 backbone and ASPP/decoder head.",
    ),
    "segformer_b0": ModelSpec(
        name="SegFormer-B0",
        task="semantic",
        implementation="transformers.SegformerForSemanticSegmentation",
        reference="Xie et al., SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers, NeurIPS 2021",
        notes=(
            "Official Hugging Face Transformers implementation. With pretrained_backbone=True, "
            f"loads only the ImageNet-1K pretrained encoder from {SEGFORMER_B0_PRETRAINED}; "
            "the all-MLP decoder and class head are initialized for MaizeMask."
        ),
    ),
    "maskrcnn_resnet50_fpn": ModelSpec(
        name="Mask R-CNN ResNet-50 FPN",
        task="instance",
        implementation="torchvision.models.detection.maskrcnn_resnet50_fpn",
        reference="He et al., Mask R-CNN, ICCV 2017; Torchvision Mask R-CNN model builder",
        notes="Official Torchvision detection implementation with COCO-compatible instance targets.",
    ),
}


def build_semantic_model(
    model_name: str,
    num_classes: int,
    norm: str = "gn",
    pretrained_backbone: bool = False,
    input_channels: int = 3,
) -> nn.Module:
    if model_name == "attention_unet":
        from nets.attention_unet import AttentionUNet

        if pretrained_backbone:
            raise ValueError(
                "Attention U-Net has no registered pretrained encoder. "
                "Run it without --pretrained-backbone."
            )
        return AttentionUNet(in_channels=input_channels, num_classes=num_classes, norm=norm)
    if model_name == "deeplabv3plus_resnet50":
        if input_channels != 3:
            raise ValueError("DeepLabV3+ currently supports RGB input only (input_channels=3).")
        from nets.deeplabv3plus.modeling import deeplabv3plus_resnet50

        return deeplabv3plus_resnet50(
            num_classes=num_classes,
            output_stride=8,
            pretrained_backbone=pretrained_backbone,
        )
    if model_name == "segformer_b0":
        try:
            from transformers import (
                SegformerConfig,
                SegformerForImageClassification,
                SegformerForSemanticSegmentation,
            )
        except ImportError as exc:
            raise ImportError(
                "SegFormer-B0 requires transformers. Install with: pip install transformers"
            ) from exc
        id2label = {index: str(index) for index in range(num_classes)}
        label2id = {str(index): index for index in range(num_classes)}
        if pretrained_backbone:
            if input_channels != 3:
                raise ValueError(
                    "SegFormer ImageNet pretrained_backbone requires RGB input. "
                    "Use scratch training for derived vegetation-index channels."
                )
            config = SegformerConfig.from_pretrained(
                SEGFORMER_B0_PRETRAINED,
                revision=SEGFORMER_B0_REVISION,
            )
            config.num_labels = num_classes
            config.id2label = id2label
            config.label2id = label2id
            model = SegformerForSemanticSegmentation(config)
            image_classifier = SegformerForImageClassification.from_pretrained(
                SEGFORMER_B0_PRETRAINED,
                revision=SEGFORMER_B0_REVISION,
            )
            model.segformer.load_state_dict(image_classifier.segformer.state_dict(), strict=True)
            return model
        config = SegformerConfig(
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            num_channels=input_channels,
        )
        return SegformerForSemanticSegmentation(config)
    raise ValueError(f"Unsupported semantic model: {model_name}")


INSTANCE_MODEL_NAMES = (
    "maskrcnn_resnet50_fpn",
)


def build_maskrcnn_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
):
    try:
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except ImportError as exc:
        raise ImportError("Mask R-CNN training requires torchvision.") from exc

    if model_name == "maskrcnn_resnet50_fpn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None,
            weights_backbone="DEFAULT" if pretrained else None,
        )
    else:
        raise ValueError(
            f"Unsupported instance model: {model_name}. "
            f"Choose one of {', '.join(INSTANCE_MODEL_NAMES)}."
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        num_classes,
    )
    return model


def build_maskrcnn_resnet50_fpn(num_classes: int, pretrained: bool = False):
    """Backward-compatible v1 Mask R-CNN builder."""
    return build_maskrcnn_model(
        "maskrcnn_resnet50_fpn",
        num_classes=num_classes,
        pretrained=pretrained,
    )


def get_model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model spec: {model_name}") from exc
