# External test on COVIDGR-1.0: protocol

Written 2026-10-03, before any model was run on COVIDGR images. Nothing below was chosen after seeing predictions.

## Data
- COVIDGR-1.0 (Tabik et al., IEEE JBHI 2020), release 1.0 of github.com/ari-dasci/OD-covidgr: 852 PA chest X-rays from
  Hospital Universitario Clinico San Cecilio, Granada: 426 RT-PCR-positive (folder P) and 426 negative (folder N),
  all from the same equipment. `severity.csv` grades the positives: NORMAL-PCR+ (PCR+ with a normal-looking X-ray), MILD,
  MODERATE, SEVERE. One positive image has no row in that file; it is kept as a positive and left out of the
  per-severity figures (noted while loading the files, before any model was run).
- Used for testing only. No training, tuning, threshold fitting or temperature fitting on these images.
- Images are read as grayscale and passed through the repo's evaluation transform (resize to 224x224, 3 channels,
  ImageNet normalisation), as `scripts/run_rsna_external.py` does. No cropping. COVIDGR has no lung masks, so EIL
  and the dependence test are not run on it.
- Before inference, every COVIDGR image is compared by pixels with every primary image (64x64 normalised
  fingerprint, match threshold 0.97, the T33 method). Any matched image is excluded.

## Models
All 44 checkpoints are scored. Conclusions are drawn from the core grid only: baseline (A0) and selected arm of
each backbone at seeds 42, 123, 2026 (DenseNet121 A2_full; ResNet50, EfficientNet-B0, ViT-Base A3).

## Metrics
- Primary score: the model's softmax probability of the COVID class. Primary metric: AUROC, positives vs negatives.
- Secondary: average precision; AUROC with the 76 NORMAL-PCR+ positives removed (they carry no visible sign);
  AUROC per severity grade against all negatives; AUROC with the score 1 - P(Normal) ("any abnormality");
  sensitivity and specificity of the decision "predicted class is COVID"; which class each image is assigned to.
- Internal reference for the same contrast: AUROC of P(COVID) on the internal test images whose true class is COVID
  or Normal, from each run's committed `per_image_predictions.csv`. Drop = internal AUROC - COVIDGR AUROC.
- Comparison: selected arm minus A0, same images, per backbone and seed. 95% interval from a paired bootstrap over
  images (2,000 resamples, stratified by class, seed 42). Reported per seed and as the three-seed mean.

## Reading rule
The module "helps under source shift" on a backbone only if the selected arm's COVIDGR AUROC is higher than A0's
with an interval above zero on all three seeds. A smaller internal-to-external drop is reported but, on its own,
is not treated as evidence. One external hospital cannot show generalisation to all hospitals.
