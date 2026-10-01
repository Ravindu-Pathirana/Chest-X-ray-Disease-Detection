"""Inference-compatible ViT wrapper reconstructed from the T26 notebook.

This architecture is available for tests and architecture-only efficiency
measurement. It must be strict-loaded against a supplied T26 checkpoint before
any ViT prediction or explanation is reported.

NOTE: not exported from ``src.modules``. ``src.modules.ViTLungAttention`` (and
``build_model(backbone_name="vit_...")``) is ``lung_attention.ViTLungAttention``,
which is what the T26 notebooks train. This class differs from it in one way
that matters: ``global_pool="avg"`` makes timm apply the final LayerNorm as
``fc_norm`` AFTER pooling (and therefore after the gate), whereas the T26
architecture applies ``norm`` to the tokens BEFORE the gate. The state-dict
keys differ accordingly (``backbone.fc_norm.*`` here, ``backbone.norm.*`` in
T26 checkpoints), so a T26 checkpoint does not strict-load into this class.
Parameter counts are identical, so architecture-only efficiency numbers are
unaffected. Kept for its own test; use the exported class for anything that
loads a checkpoint.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .lung_attention import LungRegionAttention


class ViTLungAttention(nn.Module):
    """Average-pool patch-token ViT with an optional spatial lung gate."""

    def __init__(self, num_classes: int = 4, pretrained: bool = False,
                 use_attention: bool = True, reduction: int = 8,
                 gate_mode: str = "residual", drop_rate: float = 0.0) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained,
            num_classes=num_classes, global_pool="avg", drop_rate=drop_rate,
        )
        self.use_attention = use_attention
        self.attn = (LungRegionAttention(self.backbone.num_features,
                                        reduction=reduction, gate_mode=gate_mode)
                     if use_attention else None)
        self.post_attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        tokens = self.backbone.forward_features(x)
        patches = tokens[:, 1:]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError("ViT patch count must form a square grid")
        spatial = patches.transpose(1, 2).reshape(
            patches.shape[0], patches.shape[2], side, side)
        if self.attn is None:
            attention, attention_logits = None, None
        else:
            spatial, attention, attention_logits = self.attn(spatial)
        spatial = self.post_attn(spatial)
        pooled = spatial.flatten(2).mean(dim=2)
        pooled = self.backbone.fc_norm(pooled)
        pooled = self.backbone.head_drop(pooled)
        logits = self.backbone.head(pooled)
        return logits, attention, attention_logits
