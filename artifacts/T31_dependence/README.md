# T31: lung versus background dependence

Run locally 2026-10-03 (Apple M3 Pro, MPS, fp32) with the code of `notebooks/T31_dependence_all_models.ipynb`:
baseline and selected arm of each backbone at seeds 42/123/2026, plus DenseNet121 A3_multiply at seed 42
(25 checkpoints, each verified against its committed test accuracy first).

- `dependence_paired_summary.csv`: selected arm minus A0, paired by image, with bootstrap 95% intervals. Read this first.
- `dependence_verdict.csv`: the pre-set rule (LRG interval above zero in blur and swap on every seed).
- `dependence_summary_all.csv`, `counterfactual_summary_all.csv`, `occlusion_summary_all.csv`: per-checkpoint means.
- `eil_vs_background_dependence.csv`: per-image Spearman correlation of Grad-CAM EIL with background dP.
- `<backbone>/<arm>/seed_<seed>/`: per-image results (gzipped CSV).

## Reading the result

LRG rose in all 24 comparisons, so the pre-set rule is met on all four backbones. Most of that rise is higher lung
dependence. Background dependence itself fell consistently only on EfficientNet-B0, slightly on DenseNet121, and
was mixed on ResNet50 and ViT-Base. With the module, a background swap still flips 52-63% of predictions.

## Cautions

- The zero/shuffle/noise stress tests push most images into one class on most backbones (`top_class_share` in
  `counterfactual_summary_all.csv`), so their flip rates are not evidence about background reliance.
- The area-matched control perturbs scattered pixels, which moves predictions more than perturbing the whole
  background. It is reported but was not used in the rule.

The DenseNet121 A3_multiply and A5_cbam Grad-CAM results from the same run are in `artifacts/T18_lung_attention/`
(`runs/<arm>/per_image_predictions.csv`, `significance_tests_vs_A0.json`, `winner_selection.json`).
