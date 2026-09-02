"""Shortcut-suppression modules shared across model owners.

Currently: T18's Lung-Region Attention Module (Candidate A), built on
DenseNet121 and designed to be backbone-agnostic for reuse in T23
(applying the winning module to ResNet50). Import from here, not from
`src.modules.lung_attention` directly, matching this repo's convention
for `src.utils` and `src.datasets` (see CLAUDE.md).
"""

from .lung_attention import (
    CBAMSpatialAttention,
    DenseNetLungAttention,
    LogitsOnly,
    LungRegionAttention,
    attention_guidance_loss,
    build_model,
    compute_total_loss,
    freeze_backbone,
    print_trainable_parameters,
    unfreeze_final_blocks,
)

__all__ = [
    "LungRegionAttention",
    "CBAMSpatialAttention",
    "DenseNetLungAttention",
    "LogitsOnly",
    "build_model",
    "attention_guidance_loss",
    "compute_total_loss",
    "freeze_backbone",
    "unfreeze_final_blocks",
    "print_trainable_parameters",
]
