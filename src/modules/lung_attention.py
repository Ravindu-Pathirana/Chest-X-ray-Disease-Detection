"""Candidate A: Lung-Region Attention Module (T18).

A backbone-agnostic spatial attention gate whose attention map is supervised
by the ground-truth lung mask. Designed so the same class can be reused on
ResNet50 (in_channels=2048) for T23.

See Claude Working Files/T18_Lung_Region_Attention_WBS.md sections 3-4 for
the formal definition and the reasoning behind every design choice here
(logits vs. probabilities for the loss, residual vs. multiplicative gating,
zero-init, soft vs. hard mask targets).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LungRegionAttention(nn.Module):
    """1-channel spatial attention over a backbone feature map.

    Returns (gated_features, attention_probs, attention_logits).

    The *logits* are returned so the guidance loss can use
    ``binary_cross_entropy_with_logits`` -- numerically stable and, unlike
    ``binary_cross_entropy``, safe inside ``torch.autocast``.

    The final conv is zero-initialised, so at step 0 the attention map is a
    uniform 0.5 and the gate is an exact uniform rescale: the pretrained
    feature map is never hit with structured random noise.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 8,
        gate_mode: str = "residual",   # "residual" | "multiply" | "none"
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"residual", "multiply", "none"}:
            raise ValueError(f"unknown gate_mode: {gate_mode}")
        hidden = max(in_channels // reduction, 16)
        self.gate_mode = gate_mode
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)          # -> logits == 0 at init
        nn.init.constant_(self.conv2.bias, init_bias)

    def forward(self, feat: torch.Tensor):
        att_logits = self.conv2(self.act(self.conv1(feat)))   # [B,1,H,W]
        att = torch.sigmoid(att_logits)
        if self.gate_mode == "residual":
            out = feat * (1.0 + att)      # amplify lungs, never zero background
        elif self.gate_mode == "multiply":
            out = feat * att              # ablation A3
        else:
            out = feat                    # ablation A4: supervise only, don't gate
        return out, att, att_logits


def attention_guidance_loss(
    att_logits: torch.Tensor,
    masks: torch.Tensor,
    target_mode: str = "soft",       # "soft" | "hard"
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE between the attention logits and the down-pooled lung mask.

    ``masks`` is [B,1,224,224] binary. 224/7 == 32 exactly, so adaptive average
    pooling gives, per feature cell, the exact fraction of lung pixels it
    covers -- a soft target that BCEWithLogits consumes natively.
    """
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    masks = masks.float()
    h, w = att_logits.shape[-2:]
    target = F.adaptive_avg_pool2d(masks, (h, w)).clamp(0.0, 1.0)
    if target_mode == "hard":
        target = (target > 0.5).float()
    return F.binary_cross_entropy_with_logits(att_logits, target, pos_weight=pos_weight)


def compute_total_loss(
    class_logits, att_logits, labels, masks,
    classification_criterion, lambda_att: float = 0.5, target_mode: str = "soft",
):
    """L = CE_w + lambda * L_att.  Returns (total, cls, att) for logging."""
    cls_loss = classification_criterion(class_logits, labels)
    if att_logits is None or lambda_att == 0.0:
        zero = torch.zeros((), device=cls_loss.device, dtype=cls_loss.dtype)
        att_loss = (attention_guidance_loss(att_logits, masks, target_mode)
                    if att_logits is not None else zero)
        return cls_loss, cls_loss, att_loss.detach()
    att_loss = attention_guidance_loss(att_logits, masks, target_mode)
    return cls_loss + lambda_att * att_loss, cls_loss, att_loss
