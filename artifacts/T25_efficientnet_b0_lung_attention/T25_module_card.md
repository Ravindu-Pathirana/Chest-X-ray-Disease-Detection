# T25 -- Lung-Region Attention on EfficientNet-B0 -- Module Card

**Backbone:** EfficientNet-B0 (timm, ImageNet-pretrained)

**Evaluation seed:** 42

**Selected winner:** A3 (multiplicative lung attention)

## Selected hyperparameters

- `lambda_att* = 0.1`: largest tested value whose validation macro-F1 remained within 0.005 of `lambda_att=0.0`.
- `lambda_bg* = 1.0`: selected by the same validation-only tolerance rule.
- The attention module adds 102,561 parameters (+2.556%) and 0.00502 GFLOPs (+1.261%).
- Test metrics were not used for hyperparameter selection.

## Seed-42 comparison

| Arm | Gate | Accuracy | Macro-F1 | Attention Dice | EIL post | Delta EIL vs A0 |
|---|---|---:|---:|---:|---:|---:|
| A0 | none | 0.9109 | 0.9175 | n/a | 0.3610 | 0.0000 |
| A1 | residual, no guidance | 0.9112 | 0.9172 | 0.0000 | 0.3585 | -0.0025 |
| A2 | residual + guidance | 0.9061 | 0.9123 | 0.0095 | 0.3971 | +0.0361 |
| A3 | multiply + guidance | 0.9194 | 0.9234 | 0.6868 | 0.5038 | +0.1428 |
| A4 | guidance only | 0.9112 | 0.9177 | 0.7394 | 0.3616 | +0.0006 |
| A5 | CBAM | 0.9143 | 0.9202 | 0.4196 | 0.3700 | +0.0090 |
| A6 | residual + guidance + background loss | 0.9093 | 0.9160 | 0.0000 | 0.4009 | +0.0399 |

The authoritative full table is `T25_comparison_table.csv`.

## Acceptance criteria

The original headline arm for criteria A1-A4 is A2:

- **FAIL A1:** Attention Dice 0.0095; target >= 0.80.
- **FAIL A2:** EIL-post gain +0.0361; target >= +0.05.
- **PASS A3:** EIL-pre gain +0.0083; target > 0.
- **PASS A4:** macro-F1 delta -0.0051; target >= -0.01.
- **PASS A6:** parameter and GFLOP overheads are below 3% and 2%, respectively.

Failed criteria are reported findings, not reasons to continue tuning on the test set.

## Winner selection

A3 is the seed-42 winner. Candidate arms must show no statistically detectable accuracy degradation using a directional exact McNemar test, and must show a positive paired EIL-post gain using a one-sided Wilcoxon test. Candidates are then ranked by mean EIL-post gain.

A3 passes both gates and has the largest EIL-post gain (+0.1428). Its accuracy is 0.85 percentage points above A0. The two-sided McNemar p-value is 0.0217 because this improvement is statistically detectable; treating that improvement as an "accuracy cost" would reverse the meaning of the gate. The directional p-value for A3 being worse is 0.9933.

Exact results, including both two-sided and directional p-values, are in `significance_tests_vs_A0.json`; the decision is in `winner_selection.json`.

## Multi-seed confirmation

A0 and the selected A3 arm were evaluated at seeds 42, 123, and 2026.

| Metric | A0 mean +/- SD | A3 mean +/- SD | Paired A3-A0 mean +/- SD |
|---|---:|---:|---:|
| Accuracy | 0.9125 +/- 0.0015 | 0.9201 +/- 0.0007 | +0.0076 +/- 0.0008 |
| Macro-F1 | 0.9183 +/- 0.0013 | 0.9255 +/- 0.0020 | +0.0072 +/- 0.0011 |
| EIL post | 0.3204 +/- 0.0362 | 0.4338 +/- 0.0624 | +0.1133 +/- 0.0262 |

A3 improves accuracy, macro-F1, and EIL post in all three matched seeds. Its per-seed EIL gains are +0.1428, +0.0931, and +0.1040. The one-sided Wilcoxon EIL-gain p-values are below 6e-144 for every seed, while the one-sided exact McNemar p-values for A3 being worse are 0.9933, 0.9861, and 0.9900. This confirms A3 rather than indicating an accuracy trade-off.

Machine-readable aggregates are in `runs_multiseed/multiseed_summary.json`.

## Counterfactual result

Background perturbation was evaluated for all seven arms. A3 has the largest EIL improvement but is also more sensitive to background perturbation than A0, so localization improvement must not be described as demonstrated background robustness.

## Reproducibility

- Split: `artifacts/splits/split_manifest_v1.csv`.
- Grad-CAM comparisons use the same fixed 1,000-image subset for every arm.
- Classification comparisons use all 3,175 test images matched by `image_path`.
- Hyperparameter selection files are under `sweeps/lambda_att/` and `sweeps/lambda_bg/`.
- Per-image predictions are under `runs/<arm>/per_image_predictions.csv`.
- Multi-seed confirmation is intentionally not claimed in this seed-42 branch; it is supplied by the `A0-vs-A3-EfficientNet-B0` follow-up branch.
