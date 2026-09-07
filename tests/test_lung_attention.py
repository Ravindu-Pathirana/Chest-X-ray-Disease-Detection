import pytest, torch, torch.nn as nn
from src.modules import (
    LungRegionAttention, attention_guidance_loss, compute_total_loss,
)

C, B, H, W = 1024, 2, 7, 7


def test_output_shapes():
    m = LungRegionAttention(C)
    f = torch.randn(B, C, H, W)
    out, att, logits = m(f)
    assert out.shape == f.shape
    assert att.shape == (B, 1, H, W) == logits.shape


def test_attention_range():
    m = LungRegionAttention(C)
    _, att, _ = m(torch.randn(B, C, H, W) * 5)
    assert torch.all(att > 0) and torch.all(att < 1)


def test_zero_init_is_uniform():
    m = LungRegionAttention(C)
    _, att, logits = m(torch.randn(B, C, H, W))
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)
    assert torch.allclose(att, torch.full_like(att, 0.5), atol=1e-6)


def test_residual_gate_identity_scale():
    m = LungRegionAttention(C, gate_mode="residual")
    f = torch.randn(B, C, H, W)
    out, _, _ = m(f)
    torch.testing.assert_close(out, f * 1.5, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["multiply", "none"])
def test_gate_modes(mode):
    m = LungRegionAttention(C, gate_mode=mode)
    f = torch.randn(B, C, H, W)
    out, att, _ = m(f)
    expected = f * att if mode == "multiply" else f
    torch.testing.assert_close(out, expected)


def test_gradients_flow():
    m = LungRegionAttention(C)
    f = torch.randn(B, C, H, W, requires_grad=True)
    out, _, logits = m(f)
    (out.mean() + logits.mean()).backward()
    for n, p in m.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n
    assert m.conv2.weight.grad.abs().sum() > 0     # zero-init still learns


def test_guidance_loss_perfect_vs_inverted():
    """BCE against a SOFT target cannot reach 0 -- its floor is the target's own
    binary entropy H(t), because boundary cells are genuinely fractional at 7x7.
    See WBS section 4.4a: expect att_loss to plateau near 0.2, not near 0."""
    import torch.nn.functional as Fn
    mask = torch.zeros(B, 1, 224, 224); mask[:, :, 64:160, 48:176] = 1.0
    target = Fn.adaptive_avg_pool2d(mask, (H, W))
    good = 20.0 * (2 * target - 1)                       # near-perfect logits
    floor = -(target * torch.log(target + 1e-12)
              + (1 - target) * torch.log(1 - target + 1e-12)).mean().item()
    assert abs(attention_guidance_loss(good, mask).item() - floor) < 1e-3
    assert attention_guidance_loss(-good, mask).item() > 10 * (floor + 0.1)

    # a fully binary target does have floor 0
    hard_logits = 20.0 * (2 * (target > 0.5).float() - 1)
    assert attention_guidance_loss(hard_logits, mask, target_mode="hard").item() < 1e-5

    # uninformative uniform attention sits exactly at log 2
    assert abs(attention_guidance_loss(torch.zeros(B, 1, H, W), mask).item() - 0.69315) < 1e-4


def test_soft_target_has_partial_cells():
    mask = torch.zeros(1, 1, 224, 224); mask[:, :, :, :100] = 1.0   # cuts mid-cell
    t = torch.nn.functional.adaptive_avg_pool2d(mask, (H, W))
    assert ((t > 0) & (t < 1)).any(), "soft target collapsed to binary"


def test_lambda_zero_contributes_no_gradient():
    m = LungRegionAttention(C)
    f = torch.randn(B, C, H, W)
    _, _, logits = m(f)
    crit = nn.CrossEntropyLoss()
    logit_cls = torch.randn(B, 4, requires_grad=True)
    y = torch.tensor([0, 1])
    mask = torch.zeros(B, 1, 224, 224); mask[:, :, 64:160, 48:176] = 1.0
    total, cls, att = compute_total_loss(logit_cls, logits, y, mask, crit, lambda_att=0.0)
    torch.testing.assert_close(total, cls)
    assert att.requires_grad is False and torch.isfinite(att)


def test_param_count():
    m = LungRegionAttention(1024, reduction=8)
    assert sum(p.numel() for p in m.parameters()) == 131329


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_autocast_safe():
    m = LungRegionAttention(C).cuda()
    f = torch.randn(B, C, H, W, device="cuda")
    mask = torch.zeros(B, 1, 224, 224, device="cuda"); mask[:, :, 64:160, 48:176] = 1.0
    with torch.autocast(device_type="cuda"):
        _, _, logits = m(f)
        loss = attention_guidance_loss(logits, mask)
    assert torch.isfinite(loss)
