"""Tests for the S4 model assembly: DenseNetLungAttention, the backbone-agnostic
freeze/unfreeze registry, and the CBAM comparator (arm A5).

`pretrained=False` throughout -- these test structure/wiring, not learned
weights, and avoid a network dependency (real ImageNet weights) in CI.
"""
from __future__ import annotations

import timm
import torch

from src.modules import (
    CBAMSpatialAttention,
    LogitsOnly,
    build_model,
    freeze_backbone,
    unfreeze_final_blocks,
)
import src.modules.lung_attention as lung_attention_module


def test_timm_api_contract():
    m = timm.create_model("densenet121", pretrained=False, num_classes=4)
    assert hasattr(m, "forward_features") and hasattr(m, "forward_head")
    f = m.forward_features(torch.zeros(2, 3, 224, 224))
    assert f.shape == (2, 1024, 7, 7)
    assert m.num_features == 1024


def test_forward_output_shapes():
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, att, att_logits = model(x)
    assert logits.shape == (2, 4)
    assert att.shape == (2, 1, 7, 7)
    assert att_logits.shape == (2, 1, 7, 7)


def test_gate_none_is_bit_identical_to_vanilla_timm():
    """Arm A0's premise: gate_mode='none' must not perturb the vanilla path at all."""
    torch.manual_seed(0)
    wrapped = build_model(num_classes=4, use_attention=True, gate_mode="none", pretrained=False)

    plain = timm.create_model("densenet121", pretrained=False, num_classes=4)
    plain.load_state_dict(wrapped.backbone.state_dict())
    plain.eval()
    wrapped.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        plain_logits = plain(x)
        wrapped_logits, _, _ = wrapped(x)
    torch.testing.assert_close(wrapped_logits, plain_logits, rtol=0, atol=0)


def test_a0_arm_has_exact_vanilla_param_count():
    """use_attention=False must produce EXACTLY the vanilla parameter count --
    no attention module attached at all, not even a disabled one."""
    plain = timm.create_model("densenet121", pretrained=False, num_classes=4)
    a0 = build_model(num_classes=4, use_attention=False, pretrained=False)
    vanilla_params = sum(p.numel() for p in plain.parameters())
    a0_params = sum(p.numel() for p in a0.parameters())
    assert a0_params == vanilla_params == 6_957_956


def test_freeze_backbone_trainable_count_exact():
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 135_429  # classifier (4,100) + attn (131,329)


def test_unfreeze_final_blocks_trainable_count_in_range():
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    freeze_backbone(model)
    unfreeze_final_blocks(model, num_blocks=1)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 2_000_000 < trainable < 2_400_000  # WBS estimate: ~2.2M


def test_attn_trainable_in_both_phases():
    """If the attention module weren't kept trainable in phase 1, guidance
    would never learn -- this is the single most impactful wiring bug S4
    could introduce silently."""
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    freeze_backbone(model)
    assert all(p.requires_grad for p in model.attn.parameters())
    unfreeze_final_blocks(model, 1)
    assert all(p.requires_grad for p in model.attn.parameters())


def test_resnet50_freeze_registry_exact_counts():
    """T23 reuses this unchanged on ResNet50 -- verify it now rather than
    discover a naming mismatch during that task."""
    model = build_model(
        num_classes=4, use_attention=True, backbone_name="resnet50", reduction=8, pretrained=False
    )
    freeze_backbone(model)
    p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert p1 == 532_997  # classifier (8,196) + attn (524,801)

    unfreeze_final_blocks(model, 1)
    p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 14_500_000 < p2 < 16_500_000  # WBS estimate: ~15.5M

    attn_params = sum(p.numel() for p in model.attn.parameters())
    assert attn_params == 524_801

    x = torch.randn(2, 3, 224, 224)
    logits, att, _ = model(x)
    assert logits.shape == (2, 4) and att.shape == (2, 1, 7, 7)


def test_cbam_forward_contract():
    """Arm A5: att_logits must be None -- CBAM is never mask-supervised."""
    model = build_model(num_classes=4, use_attention=True, attention="cbam", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, att, att_logits = model(x)
    assert logits.shape == (2, 4)
    assert att.shape == (2, 1, 7, 7)
    assert att_logits is None


def test_cbam_is_multiplicative_as_published():
    m = CBAMSpatialAttention()
    feat = torch.randn(2, 1024, 7, 7)
    out, att, logits = m(feat)
    assert logits is None
    torch.testing.assert_close(out, feat * att)


def test_cbam_freeze_trainable_count():
    model = build_model(num_classes=4, use_attention=True, attention="cbam", pretrained=False)
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 4_198  # classifier (4,100) + CBAM's single 7x7 conv (98)


def test_logits_only_adapter():
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    wrapped = LogitsOnly(model)
    x = torch.randn(2, 3, 224, 224)
    out = wrapped(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4)


def test_unfreeze_raises_on_unregistered_backbone_family():
    model = build_model(num_classes=4, use_attention=False, pretrained=False)
    model.backbone_name = "vgg16"  # not a substring-prefix of any registered family
    try:
        unfreeze_final_blocks(model, 1)
    except ValueError as e:
        assert "vgg16" in str(e)
    else:
        raise AssertionError("expected ValueError for an unregistered backbone family")


def test_unfreeze_raises_on_unmatched_prefix(monkeypatch):
    """A stand-in for a future timm version renaming an internal module --
    must fail loudly (RuntimeError), never silently train zero extra params."""
    model = build_model(num_classes=4, use_attention=False, pretrained=False)
    monkeypatch.setitem(
        lung_attention_module._TAIL_MODULES,
        "densenet",
        lambda bb, n: ["features.this_prefix_does_not_exist"],
    )
    try:
        unfreeze_final_blocks(model, 1)
    except RuntimeError as e:
        assert "this_prefix_does_not_exist" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an unmatched freeze prefix")
