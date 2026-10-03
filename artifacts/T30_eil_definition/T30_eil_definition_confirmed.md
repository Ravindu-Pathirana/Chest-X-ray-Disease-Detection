# T30: EIL Definition - Confirmed

**Task:** T30 (Member 5), confirming the Energy-Inside-Lung (EIL) definition used across all explainability evaluation (T18, T23, T25, T26).

## Status: Confirmed, no changes needed

The EIL definition already implemented in `src/modules/gradcam.py` and `src/modules/attention_metrics.py` is hereby confirmed as the single shared definition for all cross-backbone and cross-candidate comparisons. The warning comment previously in `gradcam.py` (awaiting this confirmation) has been removed.

## The definition

**Step 1 - normalise the raw Grad-CAM map** (`gradcam.py::_normalize_cam`):

```
C = ReLU(raw_cam)
C = (C - min(C)) / (max(C) - min(C) + eps)   # per-image min-max to [0, 1]
```

**Step 2 - score against the lung mask** (`attention_metrics.py::energy_inside_lung`):

```
EIL(C, M) = sum(C * M) / (sum(C) + eps)
```

Where `C` is the normalised CAM and `M` is the binary lung mask (from T07). EIL is the fraction of the model's visual evidence (heatmap energy) that falls inside the lungs - 0 means entirely outside, 1 means entirely inside.

## Why this definition

- **ReLU first**: only positive contributions to the predicted class count as "evidence." Negative Grad-CAM values indicate evidence against the predicted class and shouldn't inflate or deflate the inside-lung fraction.
- **Per-image min-max normalisation**: Grad-CAM's raw magnitude varies by image and by backbone; normalising each image to its own [0,1] range makes EIL comparable across images, arms, and architectures. This is a deliberate choice over the `pytorch_grad_cam` library's own internal normalisation, so the definition is ours and independently verifiable.
- **Two taps (pre-gate and post-gate)**: EIL is computed twice per image - once at the module's output (`post_attn`, the headline number) and once at the last pre-gate backbone feature layer. For arm A0 (no attention module), both taps collapse to the same layer, which doubles as a wiring self-check (EIL_pre == EIL_post for A0 on every backbone).

## Verified consistent usage across all four backbones

| Backbone | Task | Pre-gate tap |
|---|---|---|
| DenseNet121 | T18 | `features.norm5` |
| ResNet50 | T23 | `layer4[-1]` |
| EfficientNet-B0 | T25 | `bn2` |
| ViT-Base | T26 | `pre_attn` (patch tokens reshaped to a spatial map) |

All four use the identical `_normalize_cam` -> `energy_inside_lung` pipeline. No backbone-specific EIL variant exists, so no retroactive recomputation is needed for any already-reported EIL numbers (T18, T23, T25, T26 results stand as-is).

## What EIL does and does not show

Carried over from the explainability method doc (unchanged by this confirmation):

- EIL shows where the model's evidence sits. It does not show that the model relies less on non-lung information - a model could have high EIL and still be sensitive to background perturbations (observed on EfficientNet-B0: background-perturbation flip rates did not fall for any arm despite reasonable EIL).
- EIL depends on the normalisation choice. A nearly-flat heatmap gets stretched to the full [0,1] range by per-image min-max, which is why a single fixed definition (this one) must be used for every comparison - mixing normalisation schemes across models/arms would invalidate the comparison.

## Sources

`src/modules/gradcam.py`, `src/modules/attention_metrics.py`, `docs/explainability_method.md`, and the T18/T23/T25/T26 artifact folders this definition was already validated against.
