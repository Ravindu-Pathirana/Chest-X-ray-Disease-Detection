# DenseNet121 A3_multiply at seeds 123 and 2026

Trained on Kaggle (Tesla T4) from `main` at commit `43c14ec`, same protocol, split manifest and 1,000-image Grad-CAM
subset as the seed-42 A3 run in `../runs/A3_multiply/`. Copied here on 2026-10-05 from the Kaggle export
`results (11).zip`; the two checkpoints are kept outside the repo (`checkpoints/densenet121/A3_multiply_seed<seed>/`).
With these runs the multiplicative gate has a three-seed repeat on all four backbones.

- `A3_multiply_seed<seed>/`: test results, per-image predictions, paired A0-vs-A3 classification/EIL/calibration
  files, lung-vs-background dependence (per image and summary) and `run_manifest.json` (checkpoint hash, source commit).
- `A3_three_seed_summary.csv`, `A3_three_seed_aggregate.json`: A0 vs A3 per seed and the three-seed mean and SD.

Three-seed mean, A3 minus A0: accuracy -0.09 pp (per seed -0.44, -0.41, +0.57), macro-F1 -0.06 pp,
Grad-CAM EIL post-gate +30.1 pp, pre-gate +6.5 pp.

## `addendum/` (run locally 2026-10-05, Apple M3 Pro, MPS, fp32, inference only)

Produced by `external_data/densenet121_A3_addendum.py` in the workspace (not in this repo), following the T33
protocols unchanged; image lists, exclusions and A0 probabilities are read from the committed T33 files.

- `checkpoint_verification.csv`: all three A3 checkpoints reproduce their committed test accuracy with 100% prediction
  agreement. Seed 42 was re-scored only as a check: the script reproduces the committed COVIDGR and POLCOVID
  per-image probabilities to within 1e-6.
- `covidgr_A0_vs_A3_*.csv`: COVIDGR AUROC, A3 minus A0: -0.024 (seed 42), -0.024 (seed 123), +0.025 (seed 2026), each
  interval excluding zero. Three-seed mean 0.634 against 0.641. No consistent effect; the reading rule is not met.
- `polcovid_A0_vs_A3_*.csv`: within-hospital COVID vs normal +0.035, +0.022 (both intervals above zero), -0.009
  (interval includes zero); mean 0.683 against 0.667. COVID vs other pneumonia 0.564 against 0.538, still near chance.
  The reading rule (interval above zero on all three seeds) is not met.
- `dependence_paired_A0_vs_A3.csv`: same paired bootstrap as `artifacts/T31_dependence/dependence_paired_summary.csv`.
  Background dependence falls and LRG rises in all six seed-by-mode comparisons, every interval excluding zero.
  A background swap still flips 45-46% of predictions (53-55% for A0).
- `external_per_image_A3.csv.gz`: A3 probabilities per external image and seed.

- `partE_A0_vs_A3_*.csv`: POLCOVID lung-crop analysis (Part E) for A3 at all three seeds, produced by
  `external_data/densenet121_A3_partE_addendum.py` with the committed image list, donors and Grad-CAM subset. The
  seed-42 re-run reproduces the cached Part E result exactly. Post-gate EIL is higher on all three seeds (+0.32,
  +0.25, +0.26). Pre-gate EIL is lower on all three (-0.01, -0.02, -0.03), so on external crops the relocation
  comes from the gate, not from changed backbone features. LRG rises in all six comparisons through higher lung
  dependence; background dependence is higher in five of six and unchanged in one, so the internal fall in
  background dependence does not carry over.

Not done for these two seeds: T28 output in the T28 folder layout (calibration is in each run's `calibration/`
folder instead).
