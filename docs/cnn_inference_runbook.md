# Running the remaining CNN closeout evaluations

The two checkpoint-only runners use the trained weights; they never retrain a
model. Install the project dependencies (`torch`, `torchvision`, `timm`,
`PyYAML`, `pandas`, `scikit-learn`, `Pillow`, `scipy`, `matplotlib`, and optional
`fvcore` for FLOPs) in the execution environment.

## Primary CNN evaluations

Place the primary dataset in the original `COVID-19_Radiography_Dataset`
layout: each class has `images/` and matching `masks/` PNGs. The runner uses
the committed `split_manifest_v1.csv`, validates the 3,175-image validation
and test sets, strictly loads the specified checkpoint, and scores at least
100 test images before writing full results. Each run writes a checkpoint
hash, source commit, config and split details to `run_manifest.json`.

Run this command once for each checkpoint, replacing paths, backbone, arm,
and seed:

```powershell
py -3.13 scripts/run_cnn_closeout_inference.py `
  --backbone densenet121 --arm A2_full --seed 42 `
  --checkpoint C:\weights\densenet_a2_seed42.pt `
  --data-dir C:\data\COVID-19_Radiography_Dataset `
  --tasks all
```

Valid pairs are:

| Backbone argument | Baseline arm | Selected arm |
|---|---|---|
| `densenet121` | `A0_vanilla` | `A2_full` |
| `resnet50` | `A0_vanilla` | `A3_multiply` |
| `efficientnet_b0` | `A0` | `A3` |

Use `--smoke-only` to test loading and the first 100 test images. `--tasks`
can select `calibration`, `counterfactual`, `occlusion`, `cam`, or
`efficiency`; the default is all five. Calibration fits temperature on
validation data and evaluates on test. The counterfactual and occlusion runs
write per-image and summary CSVs. CAM uses the same committed 1,000-image
subset for every arm and seed. The efficiency task uses the existing
batch-one benchmark; use the same hardware and options for every model.
The output defaults to `artifacts/cnn_closeout/inference/<backbone>/<arm>/seed_<seed>/`.

For the six-model seed-42 study, run the two listed arms for each of the
three backbones. To repeat new inference across all three seeds, supply the
matching seed-123 and seed-2026 weights too. Checkpoint metadata is checked
against the requested arm, seed, class order and backbone when available.

After all six runs for one seed finish with `--tasks all`, aggregate the new
calibration and efficiency rows and paired intervention results:

```powershell
py -3.13 scripts/summarize_cnn_inference.py --seed 42
```

This checks one-to-one image pairing and matching labels before computing
selected-minus-baseline differences, paired bootstrap intervals and Wilcoxon
tests. The existing `scripts/build_cnn_closeout.py` remains the authoritative
builder for the previously committed classification and EIL master table.

## RSNA external evaluation

The prepared RSNA manifest refers to PNGs under `/kaggle/working/rsna_png/`.
Pass the directory containing `Normal/` and `Pneumonia/` subdirectories as
`--rsna-dir`. All 26,684 image paths are checked before inference.

The mapping is a scientific choice: decide which primary classes count as
RSNA-positive **before examining predictions**, then record a protocol ID.
For example, if the team has adopted *Lung_Opacity as the pneumonia proxy*,
the command is:

```powershell
py -3.13 scripts/run_rsna_external.py `
  --backbone densenet121 --arm A2_full --seed 42 `
  --checkpoint C:\weights\densenet_a2_seed42.pt `
  --rsna-dir C:\data\rsna_png `
  --positive-classes Lung_Opacity --protocol-id opacity_proxy_v1
```

This example mapping is **not** an endorsed choice. The script reports
binary AUROC and average precision using the sum of selected class
probabilities as the positive score. It does not report four-class accuracy.
Use `--smoke-only` to check 100 images without producing performance metrics.
The output includes per-patient probabilities, summary, protocol ID, and
checkpoint and manifest hashes.

## What still has to be supplied

Saved CSV/JSON metrics do not contain model weights or original pixels. The
checkpoint files and image datasets must be made available locally or in a
GPU-capable workspace such as Kaggle. CPU execution is supported but can be
slow. A checkpoint name alone is insufficient: verify its arm and seed; the
runner performs a strict state-dict load before inference.
