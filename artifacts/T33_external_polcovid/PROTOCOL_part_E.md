# POLCOVID Part E: Grad-CAM EIL and lung-versus-background dependence on external images

Written 2026-10-05, after checking that POLCOVID's lung masks line up with its lung-cropped ("processed") images
on eight samples, and before any model was run on a processed image. It adds to `PROTOCOL.md`; nothing in that
protocol or its results is changed.

## Data
- POLCOVID "processed" images (512x512, cropped by the dataset's authors to the lung area) and the matching lung
  masks (produced by the authors' segmentation model, not drawn by hand). Read from inside the zips.
- Same exclusions as the main protocol (16 bad-quality images, 1 overlap image), plus any image without a mask and
  any image whose mask covers less than 5% or more than 85% of the image.
- Masks are binarised at half of their maximum value. Image and mask go through the repo's evaluation transform
  together (resize to 224x224; mask nearest-neighbour), exactly as the internal test set does.

## What is different from the internal test, stated in advance
- These inputs are lung crops; the models were trained on full chest images. Scores on them are a different
  condition from the main POLCOVID result and are reported separately.
- The lungs fill about 40% of a crop, against about 24% of an internal image. A uniform heatmap scores EIL equal to
  the lung fraction, so EIL levels are not comparable with internal EIL. EIL excess (EIL minus that image's lung
  fraction) is reported next to EIL.

## Models
Baseline (A0) and selected arm of each backbone at seeds 42, 123, 2026, plus DenseNet121 A3_multiply at seed 42
(the same 25 checkpoints as the internal dependence test).

## Measures
- **E1 Classification on the lung crops.** The four contrasts and the within-hospital / hospital-2 AUROC of the
  main protocol, and the share of normal images called COVID by hospital.
- **E2 Grad-CAM EIL.** Predicted class, post-gate and pre-gate taps, on a fixed subset of 1,000 images drawn at
  random within each class in proportion to its size (seed 42), the same images for every model.
- **E3 Dependence.** For every image: blur the lungs, blur the background, swap the lungs with a donor's, swap the
  background with a donor's (kernel and code as in the internal test). dP = total-variation change of the predicted
  distribution. LRG = dP_lung - dP_background. The donor is a different-class image chosen at random (seed 42) from
  the same hospital when that hospital has one, otherwise from any hospital; the same donor is used for every model.
  The area-matched control and the zero/shuffle/noise tests are not run: the internal run showed they are not
  informative.
- **Selected arm vs A0.** Same images, per backbone and seed; 95% interval from a paired bootstrap over images
  (2,000 resamples, stratified by hospital and class, seed 42).

## Reading rules
- "Evidence relocation transfers" on a backbone only if the selected arm's EIL (post-gate) is higher than A0's with
  an interval above zero on all three seeds. Pre-gate EIL is reported as the stricter check.
- "Lung reliance up" only if LRG is higher with an interval above zero in both blur and swap on all three seeds.
- "Background reliance down" only if dP_background is lower with an interval below zero in both blur and swap on
  all three seeds.
