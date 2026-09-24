# T25 EfficientNet-B0 Lung-Attention Runbook

T25 preserves the original A0-A6 design. Hyperparameters are selected from
validation only; test evaluation occurs only in the final stage.

## T16 baseline audit

Committed T16 artifacts and `notebooks/efficientnet-b0-baseline-model.ipynb`
confirm: phase-1 LR `3e-4`, phase-2 LR `3e-5`, AdamW, weight decay `1e-3`,
cosine scheduling, batch size `32`, one unfrozen block, image size `224`,
standard timm average pooling/head, and no dropout override. T16 used a seeded
70/15/15 split and image-only augmentation. T25 intentionally uses the current
official `split_manifest_v1.csv` and `joint_v1` transforms so masks remain
aligned; those are current project requirements, not T16-tuned values.

## Kaggle commands

```bash
pip install -q pytest scipy timm grad-cam fvcore PyYAML scikit-learn pandas matplotlib seaborn wandb
```

Set the dataset once:

```bash
DATA_DIR=/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset
```

Smoke test:

```bash
python scripts/kaggle_t25_efficientnet_b0_lung_attention.py --stage smoke --data-dir "$DATA_DIR" --arms A0 A2 --lambda-att 0.5 --lambda-bg 0.0
```

Validation-only sweeps:

```bash
python scripts/kaggle_t25_efficientnet_b0_lung_attention.py --stage lambda-att-sweep --data-dir "$DATA_DIR"
python scripts/kaggle_t25_efficientnet_b0_lung_attention.py --stage lambda-bg-sweep --data-dir "$DATA_DIR"
```

Final seed-42 A0-A6 run:

```bash
python scripts/kaggle_t25_efficientnet_b0_lung_attention.py --stage final --data-dir "$DATA_DIR" --arms A0 A1 A2 A3 A4 A5 A6
```

After seed 42, choose the winning arm from the official validation/ablation
rule. Example below assumes `A2`; replace it if another arm wins:

```bash
python scripts/kaggle_t25_efficientnet_b0_lung_attention.py --stage final --data-dir "$DATA_DIR" --seed 123 --arms A0 A2 --winner A2
python scripts/kaggle_t25_efficientnet_b0_lung_attention.py --stage final --data-dir "$DATA_DIR" --seed 2026 --arms A0 A2 --winner A2
```

## Artifact layout

```text
artifacts/T25_efficientnet_b0_lung_attention/
├── model_contract_audit.json
├── sweeps/
│   ├── lambda_att/{tuning_summary.csv,selected_config.json,lambda_att_*/...}
│   └── lambda_bg/{tuning_summary.csv,selected_config.json,lambda_bg_*/...}
├── runs/A0..A6/
│   ├── efficientnet_b0_<arm>.pt
│   ├── phase*_history.json
│   ├── phase*_test_results.json
│   ├── phase*_summary_metrics.csv
│   ├── phase*_confusion_matrix.csv
│   ├── validation_metrics.json
│   ├── per_image_predictions.csv
│   ├── calibration/
│   └── counterfactual_per_image.csv
├── T25_comparison_table.csv
├── T25_efficiency.csv
├── counterfactual_summary.csv
├── counterfactual_per_image.csv
├── cam_subset_indices.json
├── run_manifest.json
└── runs_multiseed/seed_{123,2026}/...
```

The project-wide T35 table still depends on checkpoints supplied by the other
model owners. T25 writes only real EfficientNet rows and does not fabricate
missing models.
