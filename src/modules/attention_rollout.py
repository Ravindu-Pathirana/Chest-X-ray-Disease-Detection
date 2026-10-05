"""Attention rollout for the ViT-Base arms (P14 axis 3: "for ViT: attention rollout").

Abnar & Zuidema (2020). Per transformer block the post-softmax attention is
averaged over heads, mixed with the identity (the residual connection) and
the blocks are multiplied from the first to the last::

    A_hat_l = 0.5 * mean_heads(A_l) + 0.5 * I
    R       = A_hat_L @ ... @ A_hat_1          # R[i, j]: share of output
                                               # token i that came from input token j

Two ways to read a per-patch score out of ``R`` (both are reported):

- ``patch_mean`` (headline): mean over the *output patch* rows. T26's head
  average-pools the patch tokens and never reads the CLS token (see
  ``ViTLungAttention``), so this is the path the logits actually depend on.
- ``cls``: the CLS row, the textbook choice. For these checkpoints the CLS
  output feeds nothing, so treat it as a comparison to the literature, not as
  a statement about the deployed decision.

The 14x14 patch scores are bilinearly upsampled to the image size, normalised
with the *same* ``gradcam._normalize_cam`` and scored with the *same*
``energy_inside_lung`` as every Grad-CAM EIL in the project (T30), so the
numbers are on one scale.

Scope limits, stated here so nobody over-reads the result:

- Rollout only sees the transformer blocks. The lung gate in arms A1-A6 sits
  AFTER the backbone, so rollout cannot show the gate's effect directly; a
  change in rollout EIL between arms means the *backbone's attention* changed.
- Rollout is an attention-flow heuristic, not a class-specific attribution.
  It is the same map whichever class the model predicts.
- A constant score map has no range to normalise, so it scores EIL = 0; such
  images are counted in ``n_flat_maps`` rather than silently dropped.
- The map is 14x14 (16-px cells) and cannot follow a lung boundary that falls
  between cells. A perfect map on a lung-shaped mask scores about 0.86-0.94,
  never 1.0, and an uninformative map scores about the lung's share of the
  image (``eil_excess`` subtracts exactly that). Read EIL against those two
  reference points, not against 1.0.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_metrics import energy_inside_lung, lung_fraction
from .counterfactual import _dataset_image_paths
from .gradcam import _normalize_cam

SOURCES = ("patch_mean", "cls")
FLAT_TOL = 1e-10


class AttentionRecorder:
    """Context manager that captures the post-softmax attention of every block.

    timm's fused attention (``F.scaled_dot_product_attention``) never builds
    the attention matrix, so ``fused_attn`` is switched off for the duration
    and restored on exit. Maps are read from the output of each block's
    ``attn_drop`` (an identity in eval mode). Needs ``model.eval()``.
    """

    def __init__(self, backbone: nn.Module) -> None:
        self.blocks = list(backbone.blocks)
        self.maps: List[torch.Tensor] = []
        self._handles: list = []
        self._fused: list = []

    def __enter__(self) -> "AttentionRecorder":
        self.maps = []
        for blk in self.blocks:
            attn = blk.attn
            if attn.attn_drop.training and getattr(attn.attn_drop, "p", 0.0) > 0:
                raise RuntimeError("AttentionRecorder needs model.eval(): attention dropout is active")
            self._fused.append(getattr(attn, "fused_attn", None))
            if hasattr(attn, "fused_attn"):
                attn.fused_attn = False
            self._handles.append(attn.attn_drop.register_forward_hook(self._hook))
        return self

    def _hook(self, _module, _inputs, output) -> None:
        self.maps.append(output.detach())

    def __exit__(self, *exc) -> bool:
        for handle in self._handles:
            handle.remove()
        for blk, flag in zip(self.blocks, self._fused):
            if flag is not None:
                blk.attn.fused_attn = flag
        self._handles, self._fused = [], []
        return False


def rollout_matrix(attentions: Sequence[torch.Tensor], residual_weight: float = 0.5) -> torch.Tensor:
    """``attentions``: one ``[B,H,N,N]`` post-softmax map per block, first block first.
    Returns the rollout ``R`` as ``[B,N,N]`` (rows sum to 1)."""
    if len(attentions) == 0:
        raise ValueError("need at least one attention map")
    if not 0.0 <= residual_weight < 1.0:
        raise ValueError("residual_weight must be in [0, 1)")
    result: Optional[torch.Tensor] = None
    for a in attentions:
        if a.ndim != 4 or a.shape[-1] != a.shape[-2]:
            raise ValueError(f"expected [B,H,N,N] attention, got {tuple(a.shape)}")
        fused = a.float().mean(dim=1)                                   # [B,N,N], head fusion = mean
        eye = torch.eye(fused.shape[-1], device=fused.device, dtype=fused.dtype)
        layer = (1.0 - residual_weight) * fused + residual_weight * eye
        layer = layer / layer.sum(dim=-1, keepdim=True)
        result = layer if result is None else layer @ result
    return result


def rollout_patch_scores(rollout: torch.Tensor, num_prefix_tokens: int = 1,
                         source: str = "patch_mean") -> torch.Tensor:
    """Per-patch score map ``[B,side,side]`` read out of ``rollout`` ``[B,N,N]``."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}; got {source!r}")
    p = num_prefix_tokens
    if source == "cls":
        scores = rollout[:, 0, p:]
    else:
        scores = rollout[:, p:, p:].mean(dim=1)
    side = math.isqrt(scores.shape[1])
    if side * side != scores.shape[1]:
        raise ValueError(f"patch tokens do not form a square grid: {scores.shape[1]}")
    return scores.reshape(scores.shape[0], side, side)


def scores_to_cam(scores: torch.Tensor, size: int = 224) -> torch.Tensor:
    """``[B,h,w]`` scores -> ``[B,1,size,size]`` in [0,1] (bilinear upsample, then the shared normalisation)."""
    up = F.interpolate(scores.unsqueeze(1).float(), size=(size, size), mode="bilinear", align_corners=False)
    return _normalize_cam(up.squeeze(1)).unsqueeze(1)


@torch.no_grad()
def rollout_cams(model: nn.Module, images: torch.Tensor, sources: Sequence[str] = SOURCES,
                 residual_weight: float = 0.5):
    """Runs ``model`` (an eval-mode ``ViTLungAttention``) once and returns
    ``(cams, logits, flat)``: ``cams[source]`` is ``[B,1,H,W]`` in [0,1], ``logits``
    is ``[B,C]`` from the same forward pass, ``flat[source]`` is a bool ``[B]``
    marking constant score maps."""
    if model.training:
        raise RuntimeError("call model.eval() before computing rollout")
    with AttentionRecorder(model.backbone) as recorder:
        out = model(images)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    if len(recorder.maps) != len(recorder.blocks):
        raise RuntimeError(
            f"captured {len(recorder.maps)} attention maps for {len(recorder.blocks)} blocks; this timm "
            "version does not route attention through attn_drop in the non-fused path"
        )
    matrix = rollout_matrix(recorder.maps, residual_weight)
    prefix = getattr(model, "num_prefix_tokens", 1)
    cams, flat = {}, {}
    for source in sources:
        scores = rollout_patch_scores(matrix, prefix, source)
        flat[source] = (scores.flatten(1).amax(1) - scores.flatten(1).amin(1)) <= FLAT_TOL
        cams[source] = scores_to_cam(scores, images.shape[-1])
    return cams, logits, flat


@torch.no_grad()
def evaluate_rollout_eil(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: Sequence[str],
    arm_name: str = "",
    sources: Sequence[str] = SOURCES,
    subset_positions: Optional[Sequence[int]] = None,
    residual_weight: float = 0.5,
    per_image_csv: Optional[Union[str, Path]] = None,
    max_images: Optional[int] = None,
) -> pd.DataFrame:
    """Rollout EIL over a whole loader (batches of ``(image, label, mask)``).

    Returns one summary row per (source, scope): scope ``all`` is every image
    seen, ``cam_subset`` only ``subset_positions`` (the fixed 1,000 images the
    Grad-CAM EIL numbers use, so the two methods score the same images).
    ``eil_excess`` is EIL minus the image's lung fraction, the value a
    uniform map would get. The loader must not shuffle: positions are batch order.
    """
    model.eval()
    paths = _dataset_image_paths(getattr(loader, "dataset", None))
    subset = set(int(i) for i in subset_positions) if subset_positions is not None else None
    rows: List[dict] = []
    seen = 0
    for images, labels, masks in loader:
        images = images.to(device).float()
        masks = masks.to(device).float()
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        cams, logits, flat = rollout_cams(model, images, sources, residual_weight)
        preds = logits.argmax(1).cpu().numpy()
        fraction = lung_fraction(masks, size=images.shape[-1]).cpu().numpy()
        eil = {s: energy_inside_lung(cams[s], masks).cpu().numpy() for s in sources}
        flat_np = {s: flat[s].cpu().numpy() for s in sources}
        for j in range(images.shape[0]):
            position = seen + j
            row = {
                "test_position": position,
                "image_path": paths[position] if paths is not None else "",
                "true_label": class_names[int(labels[j])],
                "pred_label": class_names[int(preds[j])],
                "in_cam_subset": bool(subset is not None and position in subset),
                "lung_fraction": float(fraction[j]),
            }
            for s in sources:
                row[f"eil_{s}"] = float(eil[s][j])
                row[f"flat_{s}"] = bool(flat_np[s][j])
            rows.append(row)
        seen += images.shape[0]
        if max_images is not None and seen >= max_images:
            break

    per_image = pd.DataFrame(rows)
    if max_images is not None:
        per_image = per_image.head(max_images)
    if per_image_csv is not None:
        Path(per_image_csv).parent.mkdir(parents=True, exist_ok=True)
        per_image.to_csv(per_image_csv, index=False)

    summary_rows = []
    scopes = {"all": per_image}
    if subset is not None:
        scopes["cam_subset"] = per_image[per_image["in_cam_subset"]]
    for source in sources:
        for scope, frame in scopes.items():
            values = frame[f"eil_{source}"].to_numpy(dtype=float)
            fractions = frame["lung_fraction"].to_numpy(dtype=float)
            summary_rows.append({
                "arm": arm_name, "source": source, "scope": scope, "n_images": int(len(frame)),
                "mean_eil": float(np.mean(values)) if len(frame) else float("nan"),
                "median_eil": float(np.median(values)) if len(frame) else float("nan"),
                "std_eil": float(np.std(values, ddof=1)) if len(frame) > 1 else float("nan"),
                "mean_lung_fraction": float(np.mean(fractions)) if len(frame) else float("nan"),
                "mean_eil_excess": float(np.mean(values - fractions)) if len(frame) else float("nan"),
                "n_flat_maps": int(frame[f"flat_{source}"].sum()),
            })
    return pd.DataFrame(summary_rows)
