# CNN closeout execution status

Audited on 30 September 2026 on branch `cnn-closeout`, starting from commit
`45e5bf2431413a109acae36d3dd8d3fafe73fc03`. The research brief is a
planning input; the results below are based on repository artifacts and the
local environment.

## Completed from committed evidence

- `scripts/build_cnn_closeout.py` rebuilds the 18-row, three-backbone,
  three-seed master results and paired significance package. The source is
  the saved per-image predictions, not the inconsistent legacy DenseNet
  summary. The builder's integrity audit checks all 18 files for 3,175
  unique test images, valid probability vectors, and matched arm image paths.
- The package includes per-image EIL/predictions, uncalibrated ECE and Brier,
  paired image tests, and prior efficiency results. The EfficientNet seed-42
  calibration and background counterfactual outputs are already committed
  under `artifacts/T25_efficientnet_b0_lung_attention/runs/`.
- `evaluate_region_occlusion` now supports zero/blur lung occlusion and zero
  background occlusion with per-image original-class probability drops. It
  can be run with the same deterministic mask-aware loader used for the
  counterfactual evaluation.
- `scripts/audit_cnn_closeout_readiness.py` records all 18 backbone/arm/seed
  slots, config paths, dataset manifest counts, candidate checkpoint files,
  and the execution environment. Its output is
  `artifacts/cnn_closeout/run_readiness_manifest.json`.
- `scripts/run_cnn_closeout_inference.py` now connects the trained-checkpoint
  loader to calibration, counterfactual, occlusion, CAM, and efficiency runs.
  `scripts/run_rsna_external.py` performs optional external inference after
  the team explicitly declares a binary class mapping. See the
  [inference runbook](cnn_inference_runbook.md).

## Blocked inference

The local checkout contains zero `.pt`, `.pth`, `.ckpt`, or `.safetensors`
checkpoint candidates under `artifacts/` and no `data/` image directory.
The split manifest has 14,815 train, 3,175 validation, and 3,175 test rows;
the image files themselves are absent. The RSNA manifest has 26,684 entries,
but its image paths point to `/kaggle/working/rsna_png/`, which is not here.
The current local PyTorch build is CPU only. Therefore no 100-image checkpoint
smoke test or new image inference was possible in this environment.

To finish the requested inference results, provide the trained weights for
DenseNet A0/A2, ResNet A0/A3, and EfficientNet A0/A3 at least for seed 42,
with their arm/seed identity, plus the primary images, lung masks, and RSNA
images in a GPU-capable environment. Three-seed inference requires the other
12 checkpoints too. Run the readiness inventory again with
`--checkpoint-root <weights-directory> --data-root <dataset-directory>`;
each candidate still needs a strict model load and 100-image smoke test.
Then run matched counterfactual, occlusion, validation-fitted calibration,
Grad-CAM verification, and controlled efficiency measurements. Store per-image
outputs with image identifiers before aggregating.

RSNA labels are `Normal` (20,672) and `Pneumonia` (6,012); the primary model
has four classes (`COVID`, `Lung_Opacity`, `Normal`, `Viral Pneumonia`). A
four-class external accuracy is undefined without a defensible label map.
The study team must predeclare the binary mapping and primary metric before
viewing predictions. No RSNA performance claim is made here. ViT explanation
comparability also remains unaudited and is outside this three-CNN closeout.

## Interpretation

The present results support improved anatomical evidence localization for
the selected CNN arms. They do not establish reduced shortcut reliance across
all backbones; the existing EfficientNet counterfactual case study shows why
that stronger claim needs the missing interventions.
