"""Attention/faithfulness metrics for T18 (Candidate A).

Two families, per Claude Working Files/T18_Lung_Region_Attention_WBS.md
section 6:

- Module-specific (ilar, attention_iou, attention_dice, attention_entropy,
  background_attention): consume this module's own attention map. Undefined
  for arm A0 (no attention map) and only meaningful, not undefined, for
  arm A1 (unsupervised gate) -- report that explicitly rather than treat it
  as an error case.

- Module-agnostic (energy_inside_lung): consumes a Grad-CAM map instead, so
  the same function scores every arm/candidate identically. This is the
  number T22 (winner selection) actually compares across candidates A/B/C.

All functions operate on batches and return one value PER IMAGE (never a
pre-averaged scalar) -- callers average across a batch/dataset themselves,
so per-image values stay available for paired statistics (T37) and can be
written straight into per_image_predictions.csv.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def _prep(att: torch.Tensor, mask: torch.Tensor, size: int = 224):
    """Upsamples att to `size` (bilinear) and binarizes mask at `size` (nearest
    if it isn't already there). att stays in (0,1) -- callers threshold it
    themselves where a hard boundary is needed (IoU/Dice), not here.
    """
    if att.ndim == 3:
        att = att.unsqueeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    att = F.interpolate(att.float(), size=(size, size), mode="bilinear", align_corners=False)
    mask = (mask.float() > 0.5).float()
    if mask.shape[-2:] != (size, size):
        mask = F.interpolate(mask, size=(size, size), mode="nearest")
    return att, mask


def ilar(att: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inside-Lung Attention Ratio: fraction of attention MASS inside the lungs.
    Threshold-free -- the primary module-specific metric."""
    a, m = _prep(att, mask)
    return (a * m).flatten(1).sum(1) / (a.flatten(1).sum(1) + EPS)


def attention_iou(att: torch.Tensor, mask: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Shape agreement between the thresholded attention map and the lung mask."""
    a, m = _prep(att, mask)
    b = (a > thr).float()
    inter = (b * m).flatten(1).sum(1)
    union = ((b + m) > 0).float().flatten(1).sum(1)
    return inter / (union + EPS)


def attention_dice(att: torch.Tensor, mask: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Shape agreement, Dice form (comparable to AuxSeg/Candidate B's seg-Dice)."""
    a, m = _prep(att, mask)
    b = (a > thr).float()
    inter = (b * m).flatten(1).sum(1)
    return (2 * inter) / (b.flatten(1).sum(1) + m.flatten(1).sum(1) + EPS)


def attention_entropy(att: torch.Tensor) -> torch.Tensor:
    """Mean binary entropy of the attention map: is it confident or mush?
    Takes att alone -- no mask needed, this measures the map's own sharpness.

    Uses a larger clamp epsilon (1e-6) than the module's general EPS
    (1e-8): float32 near 1.0 has ~1.19e-7 precision, so `1 - 1e-8` rounds
    back to exactly 1.0, silently defeating the clamp and producing
    log(0) = -inf -> NaN for a fully-confident (att == 0 or 1) map.
    Verified empirically -- 1e-8 reproduces the NaN, 1e-6 does not.
    """
    entropy_eps = 1e-6
    a = att.clamp(entropy_eps, 1 - entropy_eps)
    h = -(a * a.log() + (1 - a) * (1 - a).log())
    return h.flatten(1).mean(1)


def background_attention(att: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean attention mass per background pixel -- direct "still looking at
    borders/text?" signal, independent of ilar (which is dominated by the
    much larger lung region)."""
    a, m = _prep(att, mask)
    bg = 1.0 - m
    return (a * bg).flatten(1).sum(1) / (bg.flatten(1).sum(1) + EPS)


def energy_inside_lung(cam: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Grad-CAM Energy-Inside-Lung (EIL): fraction of (ReLU'd, min-max
    normalised) CAM mass inside the lungs. Module-agnostic -- computed
    identically for every arm and every shortcut-suppression candidate, so
    this is the number that's actually comparable across A/B/C (WBS 6.3).
    `cam` must already be ReLU'd and normalised to [0,1] by the caller
    (see src/modules/gradcam.py::cam_for).
    """
    c, m = _prep(cam, mask)
    return (c * m).flatten(1).sum(1) / (c.flatten(1).sum(1) + EPS)
