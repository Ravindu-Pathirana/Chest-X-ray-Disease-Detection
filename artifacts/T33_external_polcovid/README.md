# T33: external test on POLCOVID (15 Polish hospitals; COVID-19 / other pneumonia / normal)

Run locally 2026-10-05 (Apple M3 Pro, MPS, fp32), inference only, following `PROTOCOL.md`, which was written before
any model saw these images. 4,809 original images were read straight from the Synapse zips and standardised to a
512x512 cache; 4,792 are used (16 marked bad quality by the authors, 1 just over the pixel-overlap threshold).

## Why the analysis is done inside hospitals
The classes are unevenly spread: two hospitals supply two thirds of the normal images and almost no COVID, four
supply almost only COVID (`hospital_class_counts.csv`). So the headline numbers compare a positive image only with
negative images from the same hospital. Hospital 2 (Wroclaw; 349 COVID, 332 normal, 229 pneumonia, one file format)
is the single-hospital panel.

## Result (three-seed mean AUROC; baseline / selected arm)

| Contrast | Internal test | DenseNet121 | ResNet50 | EfficientNet-B0 | ViT-Base |
|---|---|---|---|---|---|
| COVID vs normal, hospital 2 | 0.99 | 0.655 / 0.661 | 0.674 / 0.676 | 0.699 / 0.710 | 0.722 / 0.703 |
| COVID vs normal, within-hospital (7 hospitals) | 0.99 | 0.667 / 0.676 | 0.669 / 0.668 | 0.703 / 0.712 | 0.728 / 0.713 |
| Any pneumonia vs normal, within-hospital (9) | 0.97-0.99 | 0.818 / 0.811 | 0.775 / 0.782 | 0.804 / 0.813 | 0.781 / 0.778 |
| COVID vs other pneumonia, within-hospital (7) | 0.99 | 0.538 / 0.542 | 0.521 / 0.524 | 0.534 / 0.542 | 0.557 / 0.568 |

- COVID vs normal falls from about 0.99 to 0.66-0.73, in line with COVIDGR (0.61-0.65).
- COVID vs other pneumonia is close to chance (0.52-0.57) although it is 0.99 internally. In the training data COVID
  comes from sources no other class uses; on hospitals where both diseases share a source the models cannot tell
  them apart.
- "Something is wrong" transfers better: any pneumonia vs normal is 0.78-0.82.
- Mixing hospitals did not inflate the scores: naive pooled AUROC is slightly below the within-hospital value.

## The module
By the pre-set rules it neither helps under source shift nor reduces source sensitivity on any backbone
(`polcovid_paired_selected_vs_baseline.csv`). Seed-level differences for COVID vs normal are mostly inside the
bootstrap interval: within-hospital, 1 of 12 backbone-seed pairs is higher and 2 are lower. Two consistent but small
effects: EfficientNet-B0 A3 is higher on any-pneumonia-vs-normal on all three seeds (about +0.01 to +0.02), and
ViT-Base A3 is higher on pneumonia-vs-normal on all three seeds (+0.02 to +0.07).

## Source sensitivity (normal images only)
Images with the same label are scored very differently by hospital. Share of normal images called COVID, three-seed
mean over the eight core models: hospital 11, 53-70%; hospitals 1, 2 and 4, 18-36%; hospital 7, 3-20%; hospital 15,
3-29% (`polcovid_normals_by_hospital.csv`). The hospital explains 5-8% of the variance of P(COVID) among normal
images. The module changes this in both directions across seeds and meets the rule on no backbone. Within hospital
1, JPEG and DICOM normal images are not distinguishable by P(COVID) (AUROC 0.44-0.55).

## Limits
One country, 2020-21. "Other pneumonia" is not marked viral or bacterial, so it maps only loosely onto Lung_Opacity
and Viral Pneumonia. The drop mixes reliance on source cues with real difficulty (COVID vs other pneumonia is hard
for radiologists too) and with differences in image preparation. No lung masks fit the original images, so EIL and
the dependence test were not run here.

## Files
`polcovid_summary_all.csv` (all 44 checkpoints), `polcovid_core_3seed_mean.csv`,
`polcovid_paired_selected_vs_baseline.csv` (bootstrap intervals), `polcovid_normals_by_hospital.csv`,
`polcovid_per_image.csv.gz`, `polcovid_image_list.csv` (labels, hospital, conversion details, overlap),
`hospital_class_counts.csv`, `analysis_groups.json`, `overlap_check.json`, `PROTOCOL.md`.
The images are not in the repo (Synapse syn50877085). The conversion and scoring scripts are kept with the image
cache in `external_data/POLCOVID_cache/`, outside the repo.

---

# Part E: Grad-CAM EIL and lung-versus-background dependence on external images

Run locally 2026-10-05 following `PROTOCOL_part_E.md` (written before any model saw these inputs). Inputs are
POLCOVID's lung-cropped 512x512 images with the lung masks that fit them; 4,792 images, 25 checkpoints (baseline and
selected arm at three seeds, plus DenseNet121 A3_multiply at seed 42). These inputs are lung crops, so this is a
separate condition from the result above. The lungs fill 42% of a crop on average, which is the EIL a featureless
heatmap would score.

## Result (three-seed mean; baseline / selected arm)

| | DenseNet121 (A2) | ResNet50 (A3) | EfficientNet-B0 (A3) | ViT-Base (A3) |
|---|---|---|---|---|
| EIL, post-gate | 0.384 / 0.458 | 0.490 / 0.743 | 0.465 / 0.549 | 0.424 / 0.811 |
| EIL, pre-gate | 0.384 / 0.394 | 0.490 / 0.645 | 0.465 / 0.512 | 0.424 / 0.503 |
| LRG, blur | -0.071 / -0.094 | -0.068 / +0.044 | -0.007 / +0.049 | -0.131 / -0.089 |
| LRG, swap | +0.078 / +0.089 | +0.045 / +0.069 | +0.050 / +0.073 | +0.009 / +0.038 |
| Background dependence, blur | 0.375 / 0.393 | 0.453 / 0.362 | 0.383 / 0.345 | 0.413 / 0.428 |
| Background dependence, swap | 0.404 / 0.424 | 0.372 / 0.341 | 0.389 / 0.365 | 0.520 / 0.501 |
| COVID vs normal, within-hospital AUROC | 0.711 / 0.718 | 0.675 / 0.701 | 0.706 / 0.719 | 0.746 / 0.743 |

Against the pre-set rules (an interval on the right side of zero on all three seeds):
- **Evidence relocation transfers: met on all four backbones.** Post-gate EIL is higher on all 12 backbone-seed
  pairs. Pre-gate EIL is higher on 11 of 12 (lower on DenseNet121 seed 2026).
- **Lung reliance up: met on ResNet50 and ViT-Base.** EfficientNet-B0 misses on one seed in swap; DenseNet121's
  residual-gate arm is lower under blur on two seeds.
- **Background reliance down: met on ResNet50 and EfficientNet-B0.** Not on ViT-Base (mixed) or DenseNet121 A2
  (higher on most seeds).
- DenseNet121 A3_multiply (seed 42 only): EIL 0.695 post-gate but 0.366 pre-gate, below its A0; LRG +0.099 blur and
  +0.153 swap, from higher lung dependence, with background dependence not reduced.

## What it does not change
- Classification barely moves. On these crops COVID vs normal is higher with the module on 5 of 12 pairs and never
  lower; only ResNet50 improves on all three seeds (about +0.03). COVID vs other pneumonia stays at 0.51-0.56.
- Cropping to the lungs made the models call more normal images COVID: 36-58% on the crops against 21-30% on the
  original images. The framing of the image is itself a cue.

## Reading it
The module does on unseen hospitals what it does internally: Grad-CAM evidence moves into the lungs, and with the
multiplicative gate predictions lean more on the lungs, with less background dependence on two backbones. That does
not make the classifier recognise COVID on a new hospital. The masks come from the dataset authors' segmentation
model, not from manual annotation.

## Part E files
`PROTOCOL_part_E.md`, `partE_summary_all.csv`, `partE_core_3seed_mean.csv`, `partE_paired_selected_vs_baseline.csv`,
`partE_per_image.csv.gz`, `partE_normals_by_hospital.csv`, `partE_image_list.csv` (lung fraction, donor, CAM subset).
