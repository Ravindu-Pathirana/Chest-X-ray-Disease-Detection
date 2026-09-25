"""Grad-CAM faithfulness harness (T18 section 6.3).

Computes the module-agnostic Energy-Inside-Lung (EIL) metric at two taps:

- post-gate (``model.post_attn``): "where does the deployed model's
  evidence actually live?" -- the headline number.
- pre-gate (``model.backbone.features.norm5``): answers H3 -- "did the
  backbone's own representation change, or did the module just multiply a
  lung-shaped map onto unchanged features?" For arm A0 (no gate at all)
  both taps are the same layer, so EIL_post == EIL_pre there -- a useful
  self-check that the harness is wired correctly.

Requires `pip install grad-cam` (the `pytorch_grad_cam` package) -- already
used successfully on Kaggle by the Candidate C notebook
(notebooks/candidate-c-grad-cam-shortcut-suppression-loss.ipynb).

WARNING: the exact EIL definition (ReLU + min-max normalisation here) has
not yet been confirmed against Member 5's T30 definition (see WBS section
6.3) -- swap the normalisation in `_normalize_cam` if theirs differs.
Cross-candidate numbers must be computed with one shared definition.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .lung_attention import LogitsOnly

EPS = 1e-8


class _IdentityTap(nn.Module):
    """Forces a distinct autograd node for Grad-CAM to hook, in case a plain
    nn.Identity doesn't produce gradients on some pytorch_grad_cam versions
    (WBS Appendix A.7 gotcha)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


def _normalize_cam(raw_cam: torch.Tensor) -> torch.Tensor:
    """ReLU, then per-image min-max normalise to [0,1]. `raw_cam` is [B,H,W]."""
    cam = F.relu(raw_cam)
    b = cam.shape[0]
    flat = cam.view(b, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + EPS)
    return flat.view_as(cam)


def cam_for(model: nn.Module, images: torch.Tensor, target_layer: nn.Module, device: torch.device):
    """Returns (cam, preds): cam is [B,1,224,224] in [0,1], preds is [B] on CPU.

    CAM is computed w.r.t. each image's PREDICTED class (not the true
    label) -- this measures where the model's actual decision came from,
    which is what "faithfulness" means here.

    Do not call this inside torch.no_grad()/inference_mode() -- Grad-CAM
    needs the backward pass. Runs in fp32 regardless of the caller's AMP
    setting: CAM normalisation in fp16 can produce NaNs.
    """
    wrapped = LogitsOnly(model).to(device).eval().float()
    images = images.to(device).float()

    with torch.no_grad():
        preds = wrapped(images).argmax(1)
    targets = [ClassifierOutputTarget(int(p)) for p in preds]

    cam_engine = GradCAM(model=wrapped, target_layers=[target_layer])
    try:
        grayscale = cam_engine(input_tensor=images, targets=targets)  # numpy [B,H,W]
    finally:
        # Not all pytorch_grad_cam versions support the `with ... as cam:`
        # context-manager form (WBS Appendix A.7 gotcha) -- release hooks
        # manually instead of relying on __exit__.
        if hasattr(cam_engine, "activations_and_grads"):
            cam_engine.activations_and_grads.release()

    raw = torch.from_numpy(grayscale).to(device)  # already ReLU'd + normalised by the library,
    cam = _normalize_cam(raw)  # but re-normalise ourselves so EIL's definition is ours, not the library's default.
    return cam.unsqueeze(1), preds.cpu()


_PRE_GATE_TAP = {
    # backbone_name prefix -> callable(backbone) -> pre-gate tap module
    "densenet": lambda bb: bb.features.norm5,
    "resnet": lambda bb: bb.layer4[-1],  # T23: ResNet50 has no features.norm5
}


def get_taps(model: nn.Module, backbone_name: str = "densenet121"):
    """Returns (tap_post, tap_pre) target-layer modules for cam_for().

    tap_post: model.post_attn (after the gate -- headline EIL_post), the
    same module for every backbone family.

    tap_pre: the last pre-gate feature-producing submodule -- family-specific,
    since different timm backbones expose it under different names
    (DenseNet's final norm layer vs. ResNet's final residual block). Looked
    up via `_PRE_GATE_TAP` by `backbone_name` prefix, the same
    family-registry pattern as `lung_attention.py`'s `_TAIL_MODULES`.
    `backbone_name` defaults to "densenet121" for backward compatibility
    with every existing T18 call site.

    For arm A0 (no gate at all) both taps are the same layer in effect
    (`post_attn` is `nn.Identity`), so EIL_post == EIL_pre there -- a useful
    self-check that the harness is wired correctly, on any backbone.
    """
    for prefix, tap_fn in _PRE_GATE_TAP.items():
        if backbone_name.startswith(prefix):
            return model.post_attn, tap_fn(model.backbone)
    raise ValueError(f"No pre-gate Grad-CAM tap registered for backbone '{backbone_name}'. Add one to _PRE_GATE_TAP.")
