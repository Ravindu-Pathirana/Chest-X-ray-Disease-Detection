"""Tests for attention rollout (P14 axis 3, ViT explanations).

`pretrained=False` and a tiny ViT throughout: these test the maths and the
wiring, not learned weights, and need no network.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.modules import build_model, energy_inside_lung, evaluate_rollout_eil, rollout_cams
from src.modules.attention_rollout import (
    AttentionRecorder,
    rollout_matrix,
    rollout_patch_scores,
    scores_to_cam,
)

TINY = "vit_tiny_patch16_224"
CLASSES = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(use_attention=True, pretrained=False, backbone_name=TINY).eval()


def _center_mask(n: int) -> torch.Tensor:
    mask = torch.zeros(n, 224, 224)
    mask[:, 56:168, 56:168] = 1.0
    return mask


def _softmax_maps(depth=3, batch=2, heads=2, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.softmax(torch.randn(batch, heads, n, n, generator=g), dim=-1) for _ in range(depth)]


def test_recorder_captures_every_block_and_restores_fused_flag(model, images):
    flags = [getattr(b.attn, "fused_attn", None) for b in model.backbone.blocks]
    with AttentionRecorder(model.backbone) as rec, torch.no_grad():
        model(images)
    assert len(rec.maps) == len(model.backbone.blocks)
    assert all(m.shape == (2, 3, 197, 197) for m in rec.maps)
    # post-softmax: every row is a distribution
    assert torch.allclose(rec.maps[0].sum(-1), torch.ones(2, 3, 197), atol=1e-5)
    assert [getattr(b.attn, "fused_attn", None) for b in model.backbone.blocks] == flags


def test_recorded_attention_equals_softmax_of_each_blocks_own_q_and_k(model, images):
    """Recompute attention by hand from what each block actually received."""
    inputs = []
    hooks = [b.attn.register_forward_pre_hook(lambda _m, args: inputs.append(args[0].detach()))
             for b in model.backbone.blocks]
    try:
        with AttentionRecorder(model.backbone) as rec, torch.no_grad():
            model(images)
    finally:
        for h in hooks:
            h.remove()
    assert len(inputs) == len(rec.maps) == len(model.backbone.blocks)
    for blk, x, got in zip(model.backbone.blocks, inputs, rec.maps):
        a = blk.attn
        b, n, c = x.shape
        qkv = a.qkv(x).reshape(b, n, 3, a.num_heads, c // a.num_heads).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        q = getattr(a, "q_norm", torch.nn.Identity())(q)
        k = getattr(a, "k_norm", torch.nn.Identity())(k)
        want = ((q * a.scale) @ k.transpose(-2, -1)).softmax(-1)
        assert torch.allclose(want, got, atol=1e-6)


def test_recorder_does_not_change_the_logits(model, images):
    with torch.no_grad():
        plain = model(images)[0]
    _, logits, _ = rollout_cams(model, images)
    assert torch.allclose(plain, logits, atol=1e-4)


def test_recorder_refuses_train_mode_attention_dropout():
    m = build_model(use_attention=False, pretrained=False, backbone_name=TINY, drop_rate=0.0)
    m.train()
    for blk in m.backbone.blocks:
        blk.attn.attn_drop.p = 0.1
    with pytest.raises(RuntimeError, match="eval"):
        with AttentionRecorder(m.backbone):
            pass


def test_rollout_matrix_matches_an_independent_numpy_implementation():
    maps = _softmax_maps()
    expected = None
    for a in maps:
        a_np = a.numpy().mean(axis=1)                                  # [B,N,N]
        layer = 0.5 * a_np + 0.5 * np.eye(a_np.shape[-1])[None]
        layer = layer / layer.sum(-1, keepdims=True)
        expected = layer if expected is None else np.einsum("bij,bjk->bik", layer, expected)
    got = rollout_matrix(maps).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_rollout_rows_sum_to_one_and_are_nonnegative():
    r = rollout_matrix(_softmax_maps(depth=6))
    assert torch.allclose(r.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert (r >= 0).all()


def test_identity_attention_rolls_out_to_identity():
    eye = torch.eye(5).expand(2, 2, 5, 5)
    assert torch.allclose(rollout_matrix([eye, eye, eye]), torch.eye(5).expand(2, 5, 5), atol=1e-6)


def test_later_blocks_multiply_on_the_left():
    a1, a2 = _softmax_maps(depth=2)
    r = rollout_matrix([a1, a2])
    layer = lambda a: 0.5 * a.mean(1) + 0.5 * torch.eye(5)
    assert torch.allclose(r, layer(a2) @ layer(a1), atol=1e-6)
    assert not torch.allclose(r, layer(a1) @ layer(a2), atol=1e-6)


def test_patch_scores_pick_the_right_rows_and_columns():
    n = 1 + 4                                                          # CLS + 2x2 patches
    r = torch.arange(n * n, dtype=torch.float32).reshape(1, n, n)
    cls_scores = rollout_patch_scores(r, 1, "cls")
    assert torch.equal(cls_scores, r[:, 0, 1:].reshape(1, 2, 2))
    mean_scores = rollout_patch_scores(r, 1, "patch_mean")
    assert torch.allclose(mean_scores, r[:, 1:, 1:].mean(1).reshape(1, 2, 2))


def test_patch_scores_reject_unknown_source_and_non_square_grids():
    r = torch.rand(1, 6, 6)
    with pytest.raises(ValueError, match="source"):
        rollout_patch_scores(r, 1, "max")
    with pytest.raises(ValueError, match="square"):
        rollout_patch_scores(torch.rand(1, 7, 7), 1, "cls")                # 6 patches


def test_scores_to_cam_is_normalised_and_full_size():
    cam = scores_to_cam(torch.rand(3, 14, 14) + 0.1, 224)
    assert cam.shape == (3, 1, 224, 224)
    assert torch.allclose(cam.flatten(1).amin(1), torch.zeros(3), atol=1e-6)
    assert torch.allclose(cam.flatten(1).amax(1), torch.ones(3), atol=1e-4)


def test_map_on_the_lung_scores_high_and_off_the_lung_scores_low():
    # Mask edges on the 16-px patch grid, so the 14x14 map can match it exactly.
    mask = torch.zeros(2, 1, 224, 224)
    mask[:, :, 48:176, 48:176] = 1.0
    on_lung = F.avg_pool2d(mask, 16).squeeze(1)                        # [2,14,14], 1 inside the lung
    high = energy_inside_lung(scores_to_cam(on_lung), mask)
    low = energy_inside_lung(scores_to_cam(1.0 - on_lung), mask)
    assert (high > 0.9).all()          # not 1.0: bilinear upsampling leaks across the boundary
    assert (low < 0.1).all()


def test_a_grid_unaligned_mask_caps_what_a_14x14_map_can_score():
    """Real lung masks do not sit on the patch grid, so even a perfect map is below 1."""
    mask = _center_mask(1).unsqueeze(1)                                # edges at 56/168, half a patch off-grid
    ideal = F.avg_pool2d(mask, 16).squeeze(1)
    score = energy_inside_lung(scores_to_cam(ideal), mask).item()
    assert 0.8 < score < 0.9


def test_random_maps_score_near_the_lung_fraction_chance_level():
    """The baseline eil_excess subtracts: an uninformative map scores ~ the lung's share of the image."""
    mask = torch.zeros(64, 1, 224, 224)
    mask[:, :, 48:176, 48:176] = 1.0
    g = torch.Generator().manual_seed(0)
    cams = scores_to_cam(torch.rand(64, 14, 14, generator=g), 224)
    mean_eil = energy_inside_lung(cams, mask).mean().item()
    assert mean_eil == pytest.approx((128 / 224) ** 2, abs=0.05)


def test_rollout_cams_output_contract(model, images):
    cams, logits, flat = rollout_cams(model, images)
    assert set(cams) == {"patch_mean", "cls"}
    for cam in cams.values():
        assert cam.shape == (2, 1, 224, 224)
        assert cam.min() >= 0 and cam.max() <= 1 + 1e-6
    assert logits.shape == (2, 4)
    assert flat["cls"].dtype == torch.bool and flat["cls"].shape == (2,)


def test_rollout_cams_requires_eval_mode(images):
    m = build_model(use_attention=False, pretrained=False, backbone_name=TINY)
    m.train()
    with pytest.raises(RuntimeError, match="eval"):
        rollout_cams(m, images)


def _loader(n_batches=2, bs=2):
    g = torch.Generator().manual_seed(1)
    return [
        (torch.randn(bs, 3, 224, 224, generator=g), torch.tensor([0, 1][:bs]), _center_mask(bs))
        for _ in range(n_batches)
    ]


def test_evaluate_rollout_eil_end_to_end(model, tmp_path):
    out = tmp_path / "per_image.csv"
    summary = evaluate_rollout_eil(
        model, _loader(), torch.device("cpu"), CLASSES, arm_name="A_test",
        subset_positions=[0, 3], per_image_csv=out,
    )
    per_image = __import__("pandas").read_csv(out)
    assert len(per_image) == 4
    assert per_image["test_position"].tolist() == [0, 1, 2, 3]
    assert per_image["in_cam_subset"].tolist() == [True, False, False, True]
    assert {"eil_patch_mean", "eil_cls", "lung_fraction"} <= set(per_image.columns)
    assert per_image["eil_patch_mean"].between(0, 1).all()
    assert np.allclose(per_image["lung_fraction"], 112 * 112 / 224 / 224)
    # one row per (source, scope)
    assert set(zip(summary["source"], summary["scope"])) == {
        (s, sc) for s in ("patch_mean", "cls") for sc in ("all", "cam_subset")
    }
    row = summary[(summary.source == "patch_mean") & (summary.scope == "all")].iloc[0]
    assert row.n_images == 4
    assert row.mean_eil == pytest.approx(per_image["eil_patch_mean"].mean())
    assert row.mean_eil_excess == pytest.approx((per_image["eil_patch_mean"] - per_image["lung_fraction"]).mean())
    sub = summary[(summary.source == "patch_mean") & (summary.scope == "cam_subset")].iloc[0]
    assert sub.n_images == 2
    assert sub.mean_eil == pytest.approx(per_image.loc[[0, 3], "eil_patch_mean"].mean())


def test_evaluate_rollout_eil_without_subset_has_only_the_all_scope(model):
    summary = evaluate_rollout_eil(model, _loader(1), torch.device("cpu"), CLASSES)
    assert set(summary["scope"]) == {"all"}


def test_evaluate_rollout_eil_max_images_truncates(model):
    summary = evaluate_rollout_eil(model, _loader(3), torch.device("cpu"), CLASSES, max_images=3)
    assert (summary["n_images"] == 3).all()
