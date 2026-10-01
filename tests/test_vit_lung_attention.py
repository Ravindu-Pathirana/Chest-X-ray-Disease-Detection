"""Tests for ViTLungAttention (T26/T48): the avg-pool ViT wrapper, its freeze
policy, and its Grad-CAM taps.

`pretrained=False` throughout -- these test structure/wiring, not learned
weights, and avoid a network dependency (real ImageNet weights) in CI.
"""
from __future__ import annotations

import pytest
import timm
import torch

from src.modules import (
    CBAMSpatialAttention,
    DenseNetLungAttention,
    LogitsOnly,
    ViTLungAttention,
    build_model,
    cam_for,
    compute_total_loss,
    energy_inside_lung,
    freeze_backbone,
    get_taps,
    unfreeze_final_blocks,
)

VIT = "vit_base_patch16_224"
VIT_PARAMS = 85_801_732          # timm vit_base_patch16_224, 4 classes
ATTN_PARAMS = 73_921             # 768 -> 96 -> 1, reduction 8
HEAD_PARAMS = 768 * 4 + 4


def _count(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224)


def test_build_model_dispatches_by_backbone_family():
    assert isinstance(build_model(pretrained=False, backbone_name=VIT), ViTLungAttention)
    assert isinstance(build_model(pretrained=False, backbone_name="resnet50"), DenseNetLungAttention)


def test_forward_output_shapes_use_14x14_attention_map(images):
    model = build_model(use_attention=True, pretrained=False, backbone_name=VIT).eval()
    with torch.no_grad():
        logits, att, att_logits = model(images)
    assert logits.shape == (2, 4)
    assert att.shape == att_logits.shape == (2, 1, 14, 14)
    # zero-initialised final conv -> uniform 0.5 at step 0, same as on the CNNs
    assert torch.allclose(att, torch.full_like(att, 0.5))


def test_a0_is_avgpool_over_patch_tokens_not_cls(images):
    """A0 must equal head(mean(patch tokens)) of plain timm ViT -- and differ
    from timm's own CLS-token forward, which the gate could never influence."""
    a0 = build_model(use_attention=False, pretrained=False, backbone_name=VIT).eval()
    plain = timm.create_model(VIT, pretrained=False, num_classes=4).eval()
    plain.load_state_dict(a0.backbone.state_dict())
    with torch.no_grad():
        logits, att, att_logits = a0(images)
        manual = plain.head(plain.forward_features(images)[:, 1:].mean(dim=1))
        cls_logits = plain(images)
    assert att is None and att_logits is None
    torch.testing.assert_close(logits, manual, rtol=0, atol=1e-6)
    assert not torch.allclose(logits, cls_logits, atol=1e-4)
    assert _count(a0) == _count(plain) == VIT_PARAMS


def test_gate_none_is_bit_identical_to_a0(images):
    a0 = build_model(use_attention=False, pretrained=False, backbone_name=VIT).eval()
    none = build_model(use_attention=True, gate_mode="none", pretrained=False, backbone_name=VIT).eval()
    none.backbone.load_state_dict(a0.backbone.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(none(images)[0], a0(images)[0], rtol=0, atol=0)


def test_gate_changes_the_logits(images):
    """The whole point of the avg-pool head: the gate must reach the classifier."""
    a0 = build_model(use_attention=False, pretrained=False, backbone_name=VIT).eval()
    gated = build_model(use_attention=True, gate_mode="residual", pretrained=False, backbone_name=VIT).eval()
    gated.backbone.load_state_dict(a0.backbone.state_dict())
    torch.nn.init.normal_(gated.attn.conv2.weight, std=0.5)   # make the map non-uniform
    with torch.no_grad():
        assert not torch.allclose(gated(images)[0], a0(images)[0], atol=1e-4)


def test_parameter_counts_and_freeze_policy():
    model = build_model(use_attention=True, pretrained=False, backbone_name=VIT)
    assert _count(model) == VIT_PARAMS + ATTN_PARAMS

    freeze_backbone(model)
    assert _count(model, trainable_only=True) == HEAD_PARAMS + ATTN_PARAMS

    unfreeze_final_blocks(model, num_blocks=2)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("backbone.blocks.11.") for n in trainable)
    assert any(n.startswith("backbone.blocks.10.") for n in trainable)
    assert any(n.startswith("backbone.norm.") for n in trainable)
    assert not any(n.startswith("backbone.blocks.9.") for n in trainable)
    assert _count(model, trainable_only=True) == 14_254_277


def test_cbam_arm_uses_the_spatial_only_comparator(images):
    model = build_model(use_attention=True, attention="cbam", pretrained=False, backbone_name=VIT).eval()
    assert isinstance(model.attn, CBAMSpatialAttention)
    with torch.no_grad():
        logits, att, att_logits = model(images)
    assert logits.shape == (2, 4) and att.shape == (2, 1, 14, 14) and att_logits is None


def test_losses_pool_the_mask_to_the_14x14_map(images):
    model = build_model(use_attention=True, pretrained=False, backbone_name=VIT)
    masks = torch.zeros(2, 1, 224, 224)
    masks[:, :, 64:160, 48:176] = 1.0
    logits, att, att_logits = model(images)
    total, cls, att_loss, bg_loss = compute_total_loss(
        logits, att_logits, torch.tensor([0, 1]), masks, torch.nn.CrossEntropyLoss(),
        lambda_att=1.0, att=att, lambda_bg=1.0,
    )
    assert torch.isfinite(total)
    assert abs(att_loss.item() - 0.6931) < 1e-3      # BCE at uniform 0.5 == log 2
    total.backward()
    assert model.attn.conv2.weight.grad is not None


def test_gradcam_taps_are_the_wrapper_identities(images):
    a0 = build_model(use_attention=False, pretrained=False, backbone_name=VIT).eval()
    tap_post, tap_pre = get_taps(a0)                 # backbone_name read from the model
    assert tap_post is a0.post_attn and tap_pre is a0.pre_attn

    mask = torch.zeros(2, 1, 224, 224)
    mask[:, :, 64:160, 48:176] = 1.0
    cam_post, preds = cam_for(a0, images, tap_post, torch.device("cpu"))
    cam_pre, _ = cam_for(a0, images, tap_pre, torch.device("cpu"))
    assert cam_post.shape == (2, 1, 224, 224) and preds.shape == (2,)
    assert float(cam_post.min()) >= 0.0 and float(cam_post.max()) <= 1.0
    # A0 has no gate, so both taps see the same map
    assert torch.allclose(energy_inside_lung(cam_post, mask), energy_inside_lung(cam_pre, mask), atol=1e-4)


def test_logits_only_adapter(images):
    model = build_model(use_attention=True, pretrained=False, backbone_name=VIT).eval()
    with torch.no_grad():
        assert LogitsOnly(model)(images).shape == (2, 4)
