# T28: calibration with temperature scaling, all 44 checkpoints

Run locally 2026-10-03 (Apple M3 Pro, MPS, fp32) with the code of `notebooks/T28_calibration_all_models.ipynb`.
Temperature is fitted on the validation split and reported on the test split; ECE uses 15 bins.

- All 44 checkpoints reproduced their committed test accuracy before calibration (`checkpoint_verification.csv`).
- `calibration_summary_all.csv`: one row per checkpoint (ECE, Brier and NLL before and after scaling).
- `calibration_core_3seed.csv`: baseline and selected arm per backbone, mean and SD over seeds 42/123/2026.
- `<backbone>/<arm>/seed_<seed>/`: summary JSON, reliability diagrams, and validation/test logits (`logits.npz`).

The seed-42 values agree with the independent Kaggle run on branch `t28-calibration-all-models`
(`artifacts/calibration/`).
