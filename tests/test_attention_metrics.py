"""Tests for src/modules/attention_metrics.py.

Every metric is checked against a hand-computable synthetic case, not just
"does it run" -- these numbers feed T22's winner-selection decision and
T37's paired statistics, so a silently-wrong formula here is expensive.
"""
from __future__ import annotations

import math

import torch

from src.modules import (
    attention_dice,
    attention_entropy,
    attention_iou,
    background_attention,
    energy_inside_lung,
    ilar,
)

RECT = (slice(64, 160), slice(48, 176))  # a 96x128 rectangle inside a 224x224 canvas


def _rect_mask(batch: int = 1) -> torch.Tensor:
    m = torch.zeros(batch, 1, 224, 224)
    m[:, :, RECT[0], RECT[1]] = 1.0
    return m


# ---------------------------------------------------------------------------
# ilar
# ---------------------------------------------------------------------------

def test_ilar_perfect_map_is_one():
    mask = _rect_mask()
    assert abs(ilar(mask.clone(), mask).item() - 1.0) < 1e-4


def test_ilar_uniform_map_equals_lung_area_fraction():
    mask = _rect_mask()
    frac = mask.mean().item()
    uniform_att = torch.ones(1, 1, 7, 7)  # realistic: attention lives at 7x7 before upsampling
    assert abs(ilar(uniform_att, mask).item() - frac) < 0.02


def test_ilar_inverted_map_is_near_zero():
    mask = _rect_mask()
    assert ilar(1.0 - mask, mask).item() < 1e-4


def test_ilar_returns_per_image_not_averaged():
    mask = _rect_mask(batch=3)
    att = torch.stack([mask[0], torch.ones_like(mask[0]), 1.0 - mask[0]])
    out = ilar(att, mask)
    assert out.shape == (3,)
    assert out[0].item() > out[1].item() > out[2].item()


# ---------------------------------------------------------------------------
# attention_iou / attention_dice
# ---------------------------------------------------------------------------

def test_iou_and_dice_perfect_overlap():
    mask = _rect_mask()
    assert abs(attention_iou(mask.clone(), mask).item() - 1.0) < 1e-4
    assert abs(attention_dice(mask.clone(), mask).item() - 1.0) < 1e-4


def test_iou_and_dice_no_overlap():
    mask = _rect_mask()
    disjoint = torch.zeros_like(mask)
    disjoint[:, :, 0:20, 0:20] = 1.0  # a corner, entirely outside RECT
    assert attention_iou(disjoint, mask).item() < 1e-4
    assert attention_dice(disjoint, mask).item() < 1e-4


def test_iou_and_dice_known_partial_overlap():
    # Two equal-area squares overlapping by exactly half of each.
    a = torch.zeros(1, 1, 224, 224)
    a[:, :, 0:100, 0:100] = 1.0          # area 10000
    b = torch.zeros(1, 1, 224, 224)
    b[:, :, 50:150, 0:100] = 1.0         # area 10000, overlap rows 50:100 -> 5000
    inter, union = 5000, 15000
    expected_iou = inter / union
    expected_dice = 2 * inter / (10000 + 10000)
    assert abs(attention_iou(a, b).item() - expected_iou) < 1e-3
    assert abs(attention_dice(a, b).item() - expected_dice) < 1e-3


# ---------------------------------------------------------------------------
# attention_entropy
# ---------------------------------------------------------------------------

def test_entropy_uniform_attention_is_log2():
    att = torch.full((1, 1, 7, 7), 0.5)
    h = attention_entropy(att)
    assert abs(h.item() - math.log(2)) < 1e-4


def test_entropy_confident_attention_is_near_zero():
    att = torch.zeros(1, 1, 7, 7)
    att[:, :, 3, 3] = 1.0  # a mix of exact 0s and one exact 1 -- both confident
    h = attention_entropy(att)
    assert h.item() < 1e-2


def test_entropy_decreases_as_attention_sharpens():
    soft = torch.full((1, 1, 7, 7), 0.5)
    sharper = torch.full((1, 1, 7, 7), 0.9)
    assert attention_entropy(sharper).item() < attention_entropy(soft).item()


# ---------------------------------------------------------------------------
# background_attention
# ---------------------------------------------------------------------------

def test_background_attention_zero_when_attention_confined_to_mask():
    mask = _rect_mask()
    att_inside_only = mask.clone()
    assert background_attention(att_inside_only, mask).item() < 1e-4


def test_background_attention_uniform_attention_is_near_one():
    mask = _rect_mask()
    uniform_att = torch.ones(1, 1, 7, 7)
    # every background pixel has attention mass ~1 (before interpolation
    # smoothing at the boundary), so the ratio is close to 1.
    assert background_attention(uniform_att, mask).item() > 0.95


def test_background_attention_higher_when_attention_favors_background():
    mask = _rect_mask()
    inside_only = mask.clone()
    outside_only = 1.0 - mask
    assert background_attention(outside_only, mask).item() > background_attention(inside_only, mask).item()


# ---------------------------------------------------------------------------
# energy_inside_lung -- same formula as ilar, but this is the one T22
# actually uses to compare across candidates A/B/C, so it gets its own
# explicit coverage rather than relying on ilar's tests alone.
# ---------------------------------------------------------------------------

def test_energy_inside_lung_matches_ilar_semantics():
    mask = _rect_mask()
    perfect_cam = mask.clone()
    uniform_cam = torch.ones(1, 1, 7, 7)
    inverted_cam = 1.0 - mask

    assert abs(energy_inside_lung(perfect_cam, mask).item() - 1.0) < 1e-4
    assert abs(energy_inside_lung(uniform_cam, mask).item() - mask.mean().item()) < 0.02
    assert energy_inside_lung(inverted_cam, mask).item() < 1e-4


def test_energy_inside_lung_returns_per_image():
    mask = _rect_mask(batch=2)
    cam = torch.stack([mask[0], (1.0 - mask[0])])
    out = energy_inside_lung(cam, mask)
    assert out.shape == (2,)
    assert out[0].item() > out[1].item()
