# Explainability baseline: paired CNN localization

Retrospective analysis of already committed test predictions; no new inference was run.
The same 1,000 CAM-scored images are paired within each backbone and seed.
Intervals are percentile 95% paired-image bootstrap CIs (2,000 resamples, seed 42).
Wilcoxon p-values are two-sided and Holm-adjusted across the three backbones within each seed.
This family choice is descriptive; it was not preregistered before the original test results were seen.

| Backbone | Seed | Mean EIL gain | 95% CI | Fraction improved |
|---|---:|---:|---:|---:|
| DenseNet121 | 42 | +0.0773 | [+0.0737, +0.0813] | 91.9% |
| ResNet50 | 42 | +0.2989 | [+0.2932, +0.3047] | 99.8% |
| EfficientNet-B0 | 42 | +0.1428 | [+0.1380, +0.1476] | 97.9% |
| DenseNet121 | 123 | +0.0667 | [+0.0635, +0.0699] | 93.3% |
| ResNet50 | 123 | +0.2969 | [+0.2915, +0.3027] | 100.0% |
| EfficientNet-B0 | 123 | +0.0931 | [+0.0886, +0.0978] | 90.9% |
| DenseNet121 | 2026 | +0.0673 | [+0.0637, +0.0709] | 91.7% |
| ResNet50 | 2026 | +0.3011 | [+0.2957, +0.3068] | 99.9% |
| EfficientNet-B0 | 2026 | +0.1040 | [+0.0996, +0.1087] | 94.8% |

The selected arms increase Grad-CAM energy inside the lung mask. This is an
anatomical-localization result. Grad-CAM placement alone cannot establish
causal explanation faithfulness or reduced background shortcut reliance.
See `paired_eil_bootstrap.csv` for source files and exact statistics.
