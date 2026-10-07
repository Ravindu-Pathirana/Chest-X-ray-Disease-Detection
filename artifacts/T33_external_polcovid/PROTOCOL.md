# External test on POLCOVID: protocol

Written 2026-10-05, after reading POLCOVID's metadata and file listings and test-decoding four files, and before
any model was run on a POLCOVID image. Nothing below was chosen after seeing a prediction.

## Data
- POLCOVID (Suwalska et al., Scientific Data 2023; Synapse syn50877085, CC-BY): 4,809 chest X-rays from 15 Polish
  hospitals, 2020-21: 1,236 COVID-19, 1,147 other pneumonia, 2,426 normal. Original images only (DICOM and JPEG).
- Used for testing only. No training, tuning, threshold fitting or temperature fitting on these images.
- Excluded: the 16 images the authors marked `quality = Bad`. Any image that matches a primary-dataset image in the
  pixel overlap check (64x64 normalised fingerprint, threshold 0.97) is excluded too.
- The authors' `set` column (train / hold-out test) is their own split and is ignored: all images are test images here.

## What the metadata shows (known before the design was fixed)
Classes are unevenly spread over hospitals. Hospitals 1 and 4 supply 67% of the normal images and 3 COVID images
between them; hospitals 3, 8, 10 and 13 supply 355 COVID images and 3 normal ones. Pooling all hospitals would let a
model score well by telling hospitals apart. File type is also uneven (in hospital 15 every normal image is a JPEG).
The analysis below is built so that this cannot produce the headline numbers.

## Image preparation (identical for every image; fixed before inference)
1. Read from inside the zip, one file at a time. DICOM: pixel data, VOI LUT/window applied when present, inverted
   when PhotometricInterpretation is MONOCHROME1, then intensities clipped to the 0.5-99.5 percentile range and
   scaled to 8 bits. JPEG: converted to 8-bit grayscale.
2. Trim flat dark padding: keep the longest contiguous run of rows, and of columns, in which at least 10% of pixels
   are brighter than 20/255 (two passes). This is needed because some JPEGs are viewer screenshots with wide black
   margins. (Amended after the decoding pilot and before any inference: the first version, which trimmed from the
   edges at a 2% threshold, was defeated by a one-pixel frame line around the screenshots.)
3. Resize to 512x512 and save as PNG (the working cache). At inference the repo's evaluation transform resizes to
   224x224, 3 channels, ImageNet normalisation, exactly as for the internal test set and for COVIDGR.
A conversion log records format, original size, crop box, photometric interpretation, view position and
manufacturer for every image. A pilot of about 20 images per hospital is converted and looked at before the full
run, to check decoding only.

## Models
All 44 checkpoints are scored. Conclusions use the core grid: baseline (A0) and selected arm of each backbone at
seeds 42, 123, 2026 (DenseNet121 A2_full; ResNet50, EfficientNet-B0, ViT-Base A3).

## Contrasts and scores
| Contrast | Positives | Negatives | Score |
|---|---|---|---|
| C1 (primary) | COVID-19 | NORMAL | P(COVID) |
| C2 | COVID-19 + PNEUMONIA | NORMAL | 1 - P(Normal) |
| C3 | COVID-19 | PNEUMONIA | P(COVID) |
| C4 | PNEUMONIA | NORMAL | P(Lung_Opacity) + P(Viral Pneumonia) |
POLCOVID's PNEUMONIA is not marked viral or bacterial, so C4's score is the sum of the two non-COVID disease classes.

## Analyses
- **A. One hospital (primary).** Hospital 2 (Wroclaw): 349 COVID, 333 normal, 234 pneumonia, all DICOM. AUROC for
  C1-C4 inside this hospital only.
- **B. Within-hospital AUROC over all hospitals.** For each contrast, only pairs of a positive and a negative from the
  same hospital are counted (hospitals with at least 15 images of each class in the contrast). The pooled value is
  concordant pairs over total pairs.
- **C. Naive pooled AUROC**, all hospitals mixed. Reported only to show what ignoring the hospital does.
- **D. Source sensitivity (the direct shortcut test).** Normal images only, hospitals with at least 30 normal images.
  eta-squared = share of the variance of P(COVID) that is explained by the hospital. Images with the same label should
  not be scored differently by hospital; a high value means the model reads the source. Also reported: the share of
  normal images called diseased, per hospital. Secondary: inside hospital 1, AUROC of P(COVID) for telling its JPEG
  normal images from its DICOM normal images (0.5 = the file type leaves no trace).
- **Internal reference.** The same contrasts on the internal test split from each run's committed predictions (C1:
  COVID vs Normal; C2: all diseased vs Normal; C3: COVID vs Lung_Opacity + Viral Pneumonia; C4: those two vs Normal).
- **Selected arm vs A0.** Same images, per backbone and seed; 95% interval from a paired bootstrap over images
  (2,000 resamples, stratified by hospital and class, seed 42).

## Reading rules
- The module "helps under source shift" on a backbone only if the selected arm's AUROC for C1 in analysis A and in
  analysis B is higher than A0's with an interval above zero on all three seeds.
- The module "reduces source sensitivity" on a backbone only if eta-squared in analysis D is lower than A0's with an
  interval below zero on all three seeds.
- Analysis A is one hospital. Fifteen hospitals in one country are not all hospitals.

## Not done here
POLCOVID's lung masks are 512x512 and fit its separate lung-cropped images, not the originals, so EIL and the
lung-versus-background dependence test are not run in this protocol.
