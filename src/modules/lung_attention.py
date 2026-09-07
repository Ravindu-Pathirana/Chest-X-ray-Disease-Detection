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

import timm
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


class CBAMSpatialAttention(nn.Module):
    """Woo et al. 2018 (CBAM), spatial branch only -- the published comparator
    for arm A5. M_s(F) = sigma(f^{7x7}([AvgPool_c(F); MaxPool_c(F)])).

    Multiplicative gate (as published), no logits returned -- A5 is never
    mask-supervised, so there is nothing for a guidance loss to consume.
    On a 7x7 feature map a 7x7 conv's receptive field covers the whole map,
    which is CBAM's intended global behaviour; do not shrink the kernel.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, feat: torch.Tensor):
        avg_out = feat.mean(dim=1, keepdim=True)
        max_out = feat.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return feat * att, att, None


class DenseNetLungAttention(nn.Module):
    """Backbone + LungRegionAttention (or CBAM, for arm A5) inserted before
    the classifier head.

    Uses timm's own ``forward_head`` so the classification head (pooling +
    dropout + Linear) is *identical* to the vanilla baseline -- the only
    difference between arm A0 and arm A2 is the module itself. Despite the
    class name, ``backbone_name`` is not restricted to DenseNet121: T23
    reuses this unchanged for ResNet50 (in_channels=2048), which is why the
    freeze registry below is keyed by architecture family, not by name.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        use_attention: bool = True,
        reduction: int = 8,
        gate_mode: str = "residual",
        backbone_name: str = "densenet121",
        attention: str = "lung",          # "lung" (ours) | "cbam" (arm A5)
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
        self.backbone_name = backbone_name       # read by _family() below
        self.use_attention = use_attention
        self.attention_kind = attention
        c = self.backbone.num_features           # 1024 densenet121 / 2048 resnet50
        if not use_attention:
            self.attn = None
        elif attention == "lung":
            self.attn = LungRegionAttention(c, reduction=reduction, gate_mode=gate_mode)
        elif attention == "cbam":
            self.attn = CBAMSpatialAttention()
        else:
            raise ValueError(f"unknown attention: {attention}")
        # Grad-CAM tap: a named module whose output is the tensor entering the head.
        self.post_attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        f = self.backbone.forward_features(x)     # [B,C,7,7]
        if self.attn is not None:
            f, att, att_logits = self.attn(f)
        else:
            att, att_logits = None, None
        f = self.post_attn(f)
        logits = self.backbone.forward_head(f)    # global_pool -> drop -> classifier
        return logits, att, att_logits


def build_model(
    num_classes: int = 4,
    use_attention: bool = True,
    reduction: int = 8,
    gate_mode: str = "residual",
    backbone_name: str = "densenet121",
    pretrained: bool = True,
    attention: str = "lung",
) -> DenseNetLungAttention:
    return DenseNetLungAttention(num_classes, pretrained, use_attention, reduction, gate_mode, backbone_name, attention)


class LogitsOnly(nn.Module):
    """Adapter for tools that expect ``model(x) -> Tensor``
    (pytorch-grad-cam, fvcore FLOP counting, efficiency.py)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]


# ---------------------------------------------------------------------------
# Backbone-agnostic freeze / unfreeze registry.
#
# The repo otherwise duplicates architecture-specific freeze helpers per
# notebook (features.denseblock* for DenseNet, layer4/fc for ResNet50,
# blocks.*/head for ViT). Written once here so T23 needs no rewrite -- it's
# the same build_model(backbone_name="resnet50") call with in_channels=2048.
# ---------------------------------------------------------------------------

# name -> callable(backbone, n) -> list of module prefixes to unfreeze, last-block-first
_TAIL_MODULES = {
    "densenet": lambda bb, n: (
        ["features.norm5"]
        + [f"features.denseblock{i}" for i in range(4, 4 - n, -1)]
        + [f"features.transition{i}" for i in range(4 - n + 1, 4)]
    ),
    "resnet": lambda bb, n: [f"layer{i}" for i in range(4, 4 - n, -1)],
    "efficientnet": lambda bb, n: (
        ["conv_head", "bn2"]
        + [f"blocks.{i}" for i in range(len(bb.blocks) - 1, len(bb.blocks) - 1 - n, -1)]
    ),
    "vit": lambda bb, n: (
        ["norm"] + [f"blocks.{i}" for i in range(len(bb.blocks) - 1, len(bb.blocks) - 1 - n, -1)]
    ),
}


def _family(backbone_name: str) -> str:
    for key in _TAIL_MODULES:
        if backbone_name.startswith(key):
            return key
    raise ValueError(f"No freeze policy registered for '{backbone_name}'. Add one to _TAIL_MODULES.")


def _set_heads_trainable(model: DenseNetLungAttention) -> None:
    """The classification head and the attention module are trainable in BOTH phases."""
    for p in model.backbone.get_classifier().parameters():
        p.requires_grad = True
    if model.attn is not None:
        for p in model.attn.parameters():
            p.requires_grad = True


def freeze_backbone(model: DenseNetLungAttention) -> None:
    """Phase 1: backbone frozen; classifier head + attention trainable."""
    for p in model.parameters():
        p.requires_grad = False
    _set_heads_trainable(model)


def unfreeze_final_blocks(model: DenseNetLungAttention, num_blocks: int = 1) -> None:
    """Phase 2: + the last N backbone blocks (architecture-appropriate) + the tail norm.

    Raises RuntimeError if a prefix matches nothing, rather than silently
    training zero extra parameters -- the existing per-notebook freeze
    helpers match with ``any(name.startswith(u) for u in ...)`` and fail
    silently if a name changes between timm versions, which would run
    phase 2 as a 40-epoch no-op.
    """
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1")
    for p in model.parameters():
        p.requires_grad = False
    _set_heads_trainable(model)

    prefixes = _TAIL_MODULES[_family(model.backbone_name)](model.backbone, num_blocks)
    matched = set()
    for name, p in model.backbone.named_parameters():
        for pre in prefixes:
            if name.startswith(pre):
                p.requires_grad = True
                matched.add(pre)
                break
    missing = set(prefixes) - matched
    if missing:
        raise RuntimeError(f"Freeze policy matched no parameters for: {sorted(missing)}")


def print_trainable_parameters(model: nn.Module, tag: str = "") -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{tag}Trainable: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")
