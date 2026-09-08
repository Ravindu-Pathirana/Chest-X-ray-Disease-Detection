"""Counterfactual background-perturbation robustness tests for T18.

Pure post-hoc evaluation: given an already-trained model (any arm) and a
loader of (image, label, mask) triples, measures how much the model's
predicted probabilities shift when only the BACKGROUND (non-lung) region
of the image is perturbed, with the lung region held pixel-identical.

Motivation: attention-Dice/ILAR/background_attention (attention_metrics.py)
only measure where the attention map POINTS -- they say nothing about
whether the classifier's actual decision still depends on background
content. A model could have a perfectly lung-shaped attention map and
still base its prediction partly on background artifacts the gate didn't
fully suppress. This module tests the thing those metrics can't measure:
does perturbing the background change the prediction?

No training, no gradients, and no dependency on this module's own
attention mechanism (works on a plain logits-only model exactly the same
way) -- so it doubles as a genuine A0-vs-A2 shortcut-reliance comparison,
not just a sanity check of the attention module itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def perturb_background(
    images: torch.Tensor,
    masks: torch.Tensor,
    mode: str = "zero",
    noise_std: float = 0.5,
) -> torch.Tensor:
    """Returns a copy of `images` with the background (mask <= 0.5) replaced;
    lung pixels (mask > 0.5) are left EXACTLY unchanged in every mode.

    `images` is normalized (ImageNet mean/std) as produced by JointTransform,
    so "zero" means "the per-channel dataset mean" (0 in normalized space),
    not raw black.

    mode:
    - "zero": background -> 0. The strongest test -- all background
      information is removed.
    - "noise": background += Gaussian noise (std=noise_std) -- background
      structure is buried under noise rather than removed.
    - "shuffle": background pixels randomly permuted among themselves, per
      image and per channel -- destroys spatial structure/text/artifacts
      while preserving the same background intensity distribution, so a
      model reacting to this specifically shows it depends on background
      SHAPE/STRUCTURE, not merely aggregate brightness.
    """
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    lung = (masks > 0.5).float()
    if lung.shape[-2:] != images.shape[-2:]:
        lung = F.interpolate(lung, size=images.shape[-2:], mode="nearest")
    bg = 1.0 - lung

    if mode == "zero":
        return images * lung
    if mode == "noise":
        noise = torch.randn_like(images) * noise_std
        return images * lung + (images + noise) * bg
    if mode == "shuffle":
        out = images.clone()
        batch, channels = images.shape[:2]
        for i in range(batch):
            bg_i = bg[i, 0] > 0.5
            if not bool(bg_i.any()):
                continue  # no background pixels in this image -- nothing to shuffle
            for ch in range(channels):
                vals = images[i, ch][bg_i]
                perm = torch.randperm(vals.numel(), device=vals.device)
                out[i, ch][bg_i] = vals[perm]
        return out
    raise ValueError(f"unknown perturbation mode: {mode}")


def _as_logits(model_output: Union[torch.Tensor, tuple]) -> torch.Tensor:
    """Accepts either a plain logits tensor or this repo's 3-tuple model
    output (logits, attention, attention_logits) -- so a caller can pass
    any T18 arm's model (including arm A0, which has no attention at all)
    without wrapping it in `LogitsOnly` first."""
    return model_output[0] if isinstance(model_output, tuple) else model_output


@torch.no_grad()
def counterfactual_stability(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    mode: str = "zero",
    noise_std: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Runs `model` on `images` and on a background-perturbed copy, and
    compares the predicted class distributions.

    Returns per-image tensors (never pre-averaged, matching this repo's
    attention_metrics.py convention -- callers average across a
    batch/dataset themselves):
    - "stability": 1 - total_variation_distance(p_orig, p_perturbed), in
      [0, 1]. 1.0 = the predicted distribution is completely unchanged by
      the background perturbation (no background reliance detected);
      0.0 = maximally different.
    - "pred_changed": 1.0 if argmax(p_orig) != argmax(p_perturbed), i.e.
      the perturbation flipped the predicted class outright; else 0.0.
    """
    perturbed = perturb_background(images, masks, mode=mode, noise_std=noise_std)

    p_orig = torch.softmax(_as_logits(model(images)), dim=1)
    p_pert = torch.softmax(_as_logits(model(perturbed)), dim=1)

    tv_distance = 0.5 * (p_orig - p_pert).abs().sum(dim=1)
    stability = 1.0 - tv_distance
    pred_changed = (p_orig.argmax(dim=1) != p_pert.argmax(dim=1)).float()

    return {"stability": stability, "pred_changed": pred_changed}


def evaluate_counterfactual_robustness(
    model: nn.Module,
    loader,
    device: torch.device,
    modes: Sequence[str] = ("zero", "shuffle", "noise"),
    noise_std: float = 0.5,
    seed: int = 42,
    arm_name: str = "",
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Runs `counterfactual_stability` over an entire loader for each mode
    in `modes`; returns one row per mode with the mean stability and the
    fraction of images whose predicted class flipped.

    `seed` is set once via `torch.manual_seed` before the loop (matching
    this repo's set-once-per-config convention, e.g. run_full_arm) so the
    "noise"/"shuffle" perturbations are reproducible across a re-run --
    exact reproducibility of the perturbation itself isn't load-bearing
    for the comparison (only the aggregate stability/flip-rate is), but
    there's no reason not to have it.

    If `output_csv` is given, appends to it (creating it if missing) --
    matches the append convention already used in this repo
    (notebooks/efficiency.py::benchmark_model), so this can be called once
    per arm (A0, A2, ...) across separate notebook cells and accumulate
    into one comparison table.
    """
    model.eval()
    torch.manual_seed(seed)

    rows = []
    for mode in modes:
        stabilities = []
        changed = []
        for images, _labels, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            result = counterfactual_stability(model, images, masks, mode=mode, noise_std=noise_std)
            stabilities.append(result["stability"].cpu())
            changed.append(result["pred_changed"].cpu())

        stabilities_t = torch.cat(stabilities)
        changed_t = torch.cat(changed)
        rows.append({
            "arm": arm_name,
            "mode": mode,
            "n_images": int(stabilities_t.numel()),
            "mean_stability": float(stabilities_t.mean()),
            "pred_flip_rate": float(changed_t.mean()),
        })

    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if output_csv.exists():
            df = pd.concat([pd.read_csv(output_csv), df], ignore_index=True)
        df.to_csv(output_csv, index=False)

    return df
