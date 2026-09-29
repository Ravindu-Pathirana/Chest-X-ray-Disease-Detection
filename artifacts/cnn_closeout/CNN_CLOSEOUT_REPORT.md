# Three-CNN Closeout Report

Generated entirely from committed per-image and efficiency artifacts; no model inference was performed.

## Three-seed headline results

| Backbone | Selected arm | Δ accuracy (pp) | Δ macro-F1 (pp) | Δ EIL (pp) |
|---|---|---:|---:|---:|
| DenseNet121 | A2_full | +0.02 ± 0.15 | -0.08 ± 0.39 | +7.05 ± 0.60 |
| ResNet50 | A3_multiply | -0.42 ± 0.05 | -0.52 ± 0.17 | +29.90 ± 0.21 |
| EfficientNet-B0 | A3 | +0.76 ± 0.08 | +0.72 ± 0.11 | +11.33 ± 2.62 |

Values are paired selected-arm minus baseline means and sample standard deviations across seeds 42, 123 and 2026.

## What this package closes

- Three-backbone classification and anatomical-localisation comparison.
- Three-seed repeatability for each selected CNN arm.
- Paired image-level McNemar and Wilcoxon results.
- Uncalibrated ECE/Brier values reconstructed from saved test probabilities.
- Parameter, FLOP and latency comparison from committed efficiency artifacts.
- Input-integrity checks for sample counts, image alignment and probability validity.

## Results that cannot be reconstructed from saved predictions

- DenseNet/ResNet counterfactual background predictions: modified-image inference is required.
- Temperature-scaled DenseNet/ResNet calibration: validation logits/probabilities or checkpoints are required.
- Lung-removal/occlusion results: modified-image inference is required.
- RSNA external performance: external-image inference is required.
- Additional saliency methods: model activations and gradients are required.

EfficientNet's already committed counterfactual outputs can be reported as a focused case study, but they do not constitute a standardized three-backbone counterfactual comparison.

## Interpretation boundary

The reconstructed evidence supports claims about anatomical evidence localisation. It must not be presented as proof that shortcut reliance was reduced across all three CNNs.
