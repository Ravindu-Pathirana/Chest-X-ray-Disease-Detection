import pytest, torch, torch.nn as nn
from src.modules import (
    LungRegionAttention, attention_guidance_loss, background_suppression_loss, compute_total_loss,
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
    total, cls, att, bg = compute_total_loss(logit_cls, logits, y, mask, crit, lambda_att=0.0)
    torch.testing.assert_close(total, cls)
    assert att.requires_grad is False and torch.isfinite(att)
    assert bg.requires_grad is False and torch.isfinite(bg)  # lambda_bg defaults to 0.0 too


# ---------------------------------------------------------------------------
# background_suppression_loss / compute_total_loss's lambda_bg (opt-in,
# not part of the frozen A0-A5 ablation -- see configs/densenet121_lung_attention.yaml)
# ---------------------------------------------------------------------------

def test_background_suppression_loss_zero_when_attention_confined_to_lung():
    """Attention with zero mass outside the lung mask -> loss is exactly 0,
    regardless of how much mass is inside (mirrors ILAR/background_attention's
    own convention of ignoring in-lung mass entirely).

    Mask boundary (column 96) is deliberately cell-aligned (96 = 3*32) so
    every 7x7 cell's lung fraction is exactly 0 or 1, with no partial
    boundary cell -- otherwise setting att=target would still leave a
    boundary cell partially attending to background, by construction.
    """
    mask = torch.zeros(1, 1, 224, 224); mask[:, :, :, :96] = 1.0  # left 3 cells are lung
    target = torch.nn.functional.adaptive_avg_pool2d(mask, (H, W))
    assert set(target.unique().tolist()) <= {0.0, 1.0}  # confirm no partial cell
    att = target.clone()  # attention == the (binary) lung mask itself
    loss = background_suppression_loss(att, mask)
    assert loss.shape == (1,)
    torch.testing.assert_close(loss, torch.zeros(1), atol=1e-6, rtol=0)


def test_background_suppression_loss_positive_when_attention_leaks_into_background():
    mask = torch.zeros(1, 1, 224, 224); mask[:, :, :, :112] = 1.0
    att_leaky = torch.full((1, 1, H, W), 0.5)  # uniform attention -> half the mass is on background
    loss = background_suppression_loss(att_leaky, mask)
    assert loss.item() > 0.1  # should sit near 0.5 for a uniform map over a half-lung image


def test_background_suppression_loss_worse_than_confined_attention():
    """A sanity ordering check: an attention map concentrated in the
    background must score strictly worse than one concentrated in the lung."""
    mask = torch.zeros(1, 1, 224, 224); mask[:, :, :, :112] = 1.0
    target = torch.nn.functional.adaptive_avg_pool2d(mask, (H, W))
    confined = background_suppression_loss(target, mask)
    inverted = background_suppression_loss(1.0 - target, mask)
    assert confined.item() < inverted.item()


def test_lambda_bg_zero_contributes_no_gradient_and_no_change_to_total():
    """lambda_bg=0.0 (the value every frozen A0-A5 arm uses) must leave
    `total` and `att` identical to not passing lambda_bg/att at all."""
    m = LungRegionAttention(C)
    f = torch.randn(B, C, H, W)
    _, att, logits = m(f)
    crit = nn.CrossEntropyLoss()
    logit_cls = torch.randn(B, 4, requires_grad=True)
    y = torch.tensor([0, 1])
    mask = torch.zeros(B, 1, 224, 224); mask[:, :, 64:160, 48:176] = 1.0

    baseline = compute_total_loss(logit_cls, logits, y, mask, crit, lambda_att=0.5)
    with_bg_off = compute_total_loss(
        logit_cls, logits, y, mask, crit, lambda_att=0.5, att=att, lambda_bg=0.0,
    )
    torch.testing.assert_close(baseline[0], with_bg_off[0])  # total
    torch.testing.assert_close(baseline[1], with_bg_off[1])  # cls
    torch.testing.assert_close(baseline[2], with_bg_off[2])  # att_loss
    assert with_bg_off[3].requires_grad is False and torch.isfinite(with_bg_off[3])


def test_lambda_bg_positive_adds_to_total_and_moves_gradient():
    m = LungRegionAttention(C)
    f = torch.randn(B, C, H, W)
    _, att, logits = m(f)
    crit = nn.CrossEntropyLoss()
    logit_cls = torch.randn(B, 4, requires_grad=True)
    y = torch.tensor([0, 1])
    mask = torch.zeros(B, 1, 224, 224); mask[:, :, 64:160, 48:176] = 1.0

    total, cls, att_loss, bg_loss = compute_total_loss(
        logit_cls, logits, y, mask, crit, lambda_att=0.0, att=att, lambda_bg=1.0,
    )
    expected_bg = background_suppression_loss(att, mask).mean()
    torch.testing.assert_close(bg_loss, expected_bg)
    torch.testing.assert_close(total, cls + 1.0 * bg_loss)
    assert bg_loss.requires_grad is True  # unlike the lambda_bg=0.0 branch, this one backprops

    total.backward()
    assert m.conv2.weight.grad is not None
    assert torch.isfinite(m.conv2.weight.grad).all()


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
