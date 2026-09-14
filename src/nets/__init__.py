"""Model architecture package for MaizeMask.

Keep this package initializer lightweight. Training code should import model
builders from `nets.model_zoo` or from the specific architecture subpackage so
that using one model does not force-import all optional dependencies.
"""

__all__ = [
    "AttentionUNet",
    "deeplabv3plus_resnet50",
]


def __getattr__(name: str):
    if name == "AttentionUNet":
        from .attention_unet import AttentionUNet

        return AttentionUNet
    if name == "deeplabv3plus_resnet50":
        from .deeplabv3plus.modeling import deeplabv3plus_resnet50

        return deeplabv3plus_resnet50
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
