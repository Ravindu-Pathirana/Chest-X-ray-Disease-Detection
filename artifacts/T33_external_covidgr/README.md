# T33: external test on COVIDGR-1.0 (COVID vs negative, one hospital in Granada)

Run locally 2026-10-03 (Apple M3 Pro, MPS, fp32), inference only, following `PROTOCOL.md`, which was written before
any model saw these images. COVIDGR-1.0: 426 RT-PCR-positive and 426 negative PA chest X-rays, same equipment.

## Checks
- No overlap with the primary dataset: 0 of 852 images match any of the 21,165 primary images (highest similarity
  0.959, threshold 0.97). See `overlap_check.json`.
- The released images are full chest views, not lung crops.

## Result (score = P(COVID), three-seed mean)

| Backbone | Arm | Internal AUROC (COVID vs Normal) | COVIDGR AUROC | Negatives called COVID |
|---|---|---:|---:|---:|
| DenseNet121 | A0 | 0.998 | 0.641 | 60% |
| DenseNet121 | A2_full | 0.998 | 0.628 | 65% |
| ResNet50 | A0 | 0.991 | 0.648 | 43% |
| ResNet50 | A3_multiply | 0.991 | 0.650 | 48% |
| EfficientNet-B0 | A0 | 0.995 | 0.649 | 44% |
| EfficientNet-B0 | A3 | 0.996 | 0.651 | 42% |
| ViT-Base | A0 | 0.997 | 0.638 | 58% |
| ViT-Base | A3_multiply | 0.997 | 0.613 | 58% |

- Every model drops from about 0.99 internally to 0.61-0.65 on this hospital, with or without the module.
- The module does not help under this shift. Selected minus A0 is inside the bootstrap interval on 11 of 12
  backbone-seed pairs and significantly worse on one (ViT-Base seed 2026, -0.049). The pre-set reading rule is not
  met on any backbone.
- The models do pick up disease signal: AUROC rises with severity, from about 0.52 for PCR-positive patients with a
  normal-looking X-ray to 0.71-0.78 for severe cases.
- 42-65% of the COVID-negative images are labelled COVID.

## Reading it
The drop mixes two things that this test cannot separate: reliance on source-specific cues in the training data,
and the difficulty of COVIDGR itself (76 positives have a normal-looking X-ray, 100 are mild). It is one hospital and
one binary contrast. It supports "the high internal scores do not transfer" and "the module does not change that";
it does not show why.

## Files
`covidgr_summary_all.csv` (all 44 checkpoints), `covidgr_core_3seed_mean.csv`, `covidgr_paired_selected_vs_baseline.csv`
(bootstrap intervals), `covidgr_per_image.csv.gz` (probabilities per image and model), `covidgr_image_list.csv`
(labels, severity, closest primary image), `overlap_check.json`, `PROTOCOL.md`.
The images are not in the repo: download release 1.0 of github.com/ari-dasci/OD-covidgr.
