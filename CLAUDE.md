# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A research benchmark comparing four transfer-learning architectures (ResNet50, DenseNet121,
EfficientNet-B0, ViT-Base/16) on chest X-ray disease classification (COVID-19 Radiography
dataset: COVID / Normal / Lung Opacity / Viral Pneumonia). The point isn't just accuracy — models
are compared across five trustworthiness axes (accuracy, calibration, explainability/faithfulness,
robustness on external data, efficiency), because the primary dataset is known to contain
source bias (models can shortcut-learn hospital/scanner artifacts instead of disease features). A
second research thread develops and ablates shortcut-suppression techniques (lung-mask attention,
auxiliary segmentation head, Grad-CAM-penalty loss — see `notebooks/candidate-c-grad-cam-shortcut-suppression-loss.ipynb`)
using the dataset's supplied lung masks. Full methodology and the 17-stage pipeline plan (P01–P17)
are in `Project Documents/Each task description.md`; product framing is in `README.md`.

**Current state (2026-10-05):** training is finished on all four backbones. Each has an arm
ablation of the Lung-Region Attention Module (Candidate A), a selected arm and a three-seed repeat
(42/123/2026) of A0 and the selected arm: DenseNet121 `A2_full` (T18), ResNet50 `A3_multiply`
(T23), EfficientNet-B0 `A3` (T25), ViT-Base `A3_multiply` (T26). Arm names are `A0_vanilla`,
`A1_gate_only`, `A2_full`, `A3_multiply`, `A4_guidance_only`, `A5_cbam`, `A6_full_bg` on ResNet50
and ViT-Base; DenseNet121 has A0-A5 plus the separate follow-up arm `A2_bg` instead of A6; and
EfficientNet-B0 uses plain `A0`..`A6`. Code that walks arms needs to handle all three.
DenseNet121 `A3_multiply` also has seeds 123/2026 (`artifacts/T18_lung_attention/runs_multiseed_A3/`, added
2026-10-05 with their external-test and dependence results), so the multiplicative gate has a three-seed
repeat on every backbone. The T28/T31/T33 summary files and README tables still use `A2_full` for DenseNet121.
The 44 resulting checkpoints have been evaluated inference-only for calibration (T28), lung versus
background dependence (T31), external validity (T33) and efficiency (T35); current work is those
evaluations and the paper. `README.md`'s status and results tables stop at the three-CNN closeout
and do not yet cover T28/T31/T33, so read the `README.md` in each `artifacts/T*` folder for those.

**Findings that limit what may be claimed** (in code comments, docs, README or paper text):

- RSNA is **not** an external or OOD test set: 14,863 of the 21,165 primary images are
  pixel-identical to RSNA images, most of them in the training split
  (`artifacts/T33_rsna_overlap/`). External validity is tested on COVIDGR-1.0 and POLCOVID.
- On both external sets COVID-vs-normal AUROC falls from about 0.99 to 0.61-0.73, with or without
  the module (`artifacts/T33_external_covidgr/`, `artifacts/T33_external_polcovid/`).
- EIL is an anatomical-localisation metric, not evidence of faithfulness. The module raises EIL on
  every backbone, yet a background swap still flips 52-63% of predictions
  (`artifacts/T31_dependence/`). Say "moves evidence into the lungs", not "suppresses shortcuts".
- The dataset's per-file source metadata is wrong for the Normal class; use
  `artifacts/T33_rsna_overlap/primary_image_source_verified.csv` for any analysis by image source.

The repo has no `data/`, `models/` or `results/` directories. The dataset, the `.pt` checkpoints
and the external test sets sit outside the repo (locally, in the parent workspace folder; see its
`CLAUDE.md`) and are passed in as explicit paths.

## Commands

```bash
# Setup (local only — Kaggle/Colab ship a GPU-matched torch preinstalled, don't pip install torch there)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# requirements.txt covers only src/utils. The test suite and scripts also need:
pip install torch torchvision pandas scikit-learn Pillow timm grad-cam tqdm matplotlib scipy

# Run the full test suite (CPU-only, about 70 s; no dataset, weights or network needed)
pytest

# Run a single test file / test
pytest tests/test_infrastructure.py -v
pytest tests/test_infrastructure.py::test_load_config_reads_baseline_yaml -v
```

`requirements.txt` intentionally excludes `torch`/`torchvision` and model-specific deps — it only
covers the shared tracking/reproducibility layer used by `src/utils`. The authoritative list of
what the tests need is the install step in `.github/workflows/tests.yml` (CI runs `pytest -v` on
Python 3.10 for pushes and PRs to `main`); add a new test-time dependency there, not to
`requirements.txt`. `fvcore` (FLOPs) and `python-pptx` (`requirements-presentation.txt`, for
`scripts/build_t45_slides.py`) are optional. `conftest.py` puts the repo root on `sys.path` so
`from src.utils import ...` works under pytest with no packaging setup. There is no linter or
formatter configured. Tests build models with `pretrained=False`, so they never download weights.

Scripts are run from the repo root as `python scripts/<name>.py`. Checkpoint-only evaluation of one model:

```bash
python scripts/run_cnn_closeout_inference.py --backbone densenet121 --arm A2_full --seed 42 \
    --checkpoint <path>.pt --data-dir <COVID-19_Radiography_Dataset> --tasks all   # or --smoke-only
```

## Architecture

### `src/utils/` — shared infrastructure, not model-specific

Every team member's training notebook (any architecture) imports from here rather than
reimplementing config/seeding/tracking/checkpointing per-model:

- `config.py` — `load_config(path)` parses a YAML into a plain dict; `merge_overrides(config, {"training.batch_size": 64})` returns a modified copy via dotted-key paths (for one-off notebook tweaks without a new file).
- `reproducibility.py` — `set_seed(seed)` seeds Python/NumPy/PyTorch(CPU+CUDA) and sets `cudnn.deterministic=True` by default; `create_generator(seed)` + `seed_worker` are passed to any shuffling `DataLoader`. `strict=True` on `set_seed` enables `torch.use_deterministic_algorithms` — slower, only for chasing nondeterminism bugs, not standard runs.
- `experiment_tracking.py` — thin W&B wrapper: `initialize_wandb(config, ...)` (flattens config + logs git commit/python/torch/CUDA metadata), `log_metrics`, `log_summary_metrics`, `finish_run`, `generate_run_name(model, experiment, seed)` → `"{model}_{experiment}_seed{seed}"`.
- `checkpointing.py` — `BestCheckpointSaver` tracks one monitored metric across epochs and writes a checkpoint only on improvement; `load_checkpoint` reads it back. Architecture-agnostic (takes any `nn.Module`/optimizer).

All of it is re-exported from `src/utils/__init__.py`; import from there (`from src.utils import ...`), not from the submodules directly.

### `src/datasets/` — shared mask-aware CXR data pipeline

Built for T18, reusable by T23. `JointTransform` applies identical spatial augmentation to an
image and its lung mask together (not the image-only `build_transforms` most training notebooks
use); `CXRWithMaskDataset` returns `(image, label, mask)` triples. `build_dataloaders(...,
split_manifest_path=...)` loads the fixed split from `artifacts/splits/split_manifest_v1.csv`
rather than regenerating it (`load_split_indices_from_manifest` is the lower-level function this
calls) — per `docs/experiment_policy.md`, the split must never be regenerated per-notebook.
`compute_class_weights`/`stratified_split`/`only_images_folder`/`IMAGENET_MEAN`/`IMAGENET_STD`
match the values every other notebook already uses. Import from `src.datasets`, not
`src.datasets.covid_cxr` directly.

### `src/modules/` — shortcut-suppression modules

Mainly T18's Lung-Region Attention Module (Candidate A), designed backbone-agnostic for T23, split
by concern (one file per WBS build step) — plus `calibration.py` (T27), which is not T18-specific:
it's the shared harness every model owner uses for the calibration trustworthiness axis (T28), the
same role `notebooks/efficiency.py` plays for the efficiency axis.

- `lung_attention.py` — `LungRegionAttention` (the module itself), `CBAMSpatialAttention` (the
  published-method comparator), `DenseNetLungAttention`/`build_model()` (backbone-agnostic model
  wrapper — works for any timm architecture, not just DenseNet despite the filename; `drop_rate`
  matches Member 2's tuned baseline head, see T13 below), `freeze_backbone`/`unfreeze_final_blocks`
  (one small per-architecture-family registry covering densenet/resnet/efficientnet/vit),
  `LogitsOnly` (adapter for tools expecting `model(x) -> Tensor` — the model itself returns a
  3-tuple `(logits, attention, attention_logits)`), `compute_total_loss`/`attention_guidance_loss`
  (CE + λ_att·guidance) plus the opt-in `background_suppression_loss`/`lambda_bg` term (default
  `0.0` — not part of the frozen A0-A5 ablation, see `configs/densenet121_lung_attention.yaml`).
  This term now has a runnable follow-up experiment — `notebooks/T18_A2bg_background_suppression_followup.ipynb`
  trains a new "A2_bg" arm with `lambda_bg > 0`, as a standalone notebook that does not modify
  `T18_lung_region_attention.ipynb` or any frozen checkpoint. Run once (seed 42): improved ILAR/background
  attention but not Grad-CAM EIL (0.374→0.359) — see `T18_module_card.md` §10.
- `counterfactual.py` — `perturb_background`/`counterfactual_stability`/
  `evaluate_counterfactual_robustness`: pure post-hoc robustness tests (no training) that zero,
  shuffle, or noise a checkpoint's background pixels (lung held pixel-identical) and measure how
  much the prediction shifts — tests shortcut *reliance*, not attention *placement*, so it works
  identically on arm A0. See `artifacts/T18_lung_attention/T18_module_card.md` §10.
- `attention_metrics.py` — `ilar`, `attention_iou`, `attention_dice`, `attention_entropy`,
  `background_attention`, `energy_inside_lung` (the module-agnostic Grad-CAM faithfulness metric
  used to compare across shortcut-suppression candidates, not just within this one).
- `gradcam.py` — `cam_for`/`get_taps`: Grad-CAM at two taps (pre-gate and post-gate) to
  distinguish "the module reshaped the backbone's own evidence" from "the module just multiplies
  by a lung-shaped mask at the end."
- `training.py` — `run_epoch`/`train_phase`/`evaluate`/`run_full_arm`
  (build→freeze→train→unfreeze→train→evaluate→save, one function for every experimental arm, no
  per-arm branching)/`build_optimizer`/`build_scheduler`/`best_history_row`.
- `comparison.py` — per-image predictions, the cross-arm comparison table, acceptance-criteria
  checks, and (optional) multi-seed mean±std summaries.
- `efficiency_check.py` — reads `notebooks/efficiency.py`'s benchmark output and applies a
  pass/fail threshold; does not duplicate that harness.
- `figures.py` — heat-map overlay selection/rendering for the paper's figures.
- `calibration.py` — **T27, architecture-agnostic, used by every model owner (T28), not just
  T18.** `expected_calibration_error`/`brier_score`/`reliability_diagram_data` (hand-verified
  binning math, shared by all three) and `TemperatureScaler`/`fit_temperature` (a single learnable
  scalar `T`, fit via LBFGS). `calibration_report(model, val_loader, test_loader, ...)` is the
  one-call harness: fits `T` on `val_loader` only and reports ECE/Brier before vs. after scaling on
  `test_loader` — never the other way around, matching `docs/experiment_policy.md`'s rule that the
  test split is touched only for final numbers, never for fitting/selection. `model` must return
  plain logits — wrap T18-style `(logits, attention, attention_logits)` models with `LogitsOnly`
  first, same convention as `efficiency.py`/`gradcam.py`.

- `xai_dependence.py` — **T31.** `evaluate_xai_dependence`: per image, blurs or swaps (with a
  different-class donor image) the lungs, the background, and an area-matched background region,
  and records the total-variation change in the predicted distribution (`dP`) and
  `LRG = dP_lung - dP_background`. `build_swap_pairs` freezes the donor pairing
  (`artifacts/explainable_ai/swap_pairs_seed42.csv`) so it is identical for every arm. This is the
  dependence test the paper's claims rest on; `counterfactual.py`'s zero/shuffle/noise modes push
  most images into one class and are kept only as stress tests.
- `xai_statistics.py` — `paired_eil` (same-image paired bootstrap + Wilcoxon on saved EIL) and
  `holm_adjust`. Works on committed prediction CSVs, no model needed.
- `vit_attention.py` — **not exported and not the trained architecture.** A reconstructed ViT
  wrapper whose LayerNorm placement differs from `lung_attention.ViTLungAttention`, so T26
  checkpoints do not strict-load into it. Use `build_model(backbone_name="vit_...")` for anything
  that loads a checkpoint.

`attention_metrics.py` also exports `lung_fraction`, `eil_excess`, `eil_lift` and `pointing_game`
(EIL corrected for how much of the image the lungs occupy). The EIL definition itself (ReLU, then
per-image min-max, then energy fraction inside the mask) is fixed by
`artifacts/T30_eil_definition/T30_eil_definition_confirmed.md`; do not change the normalisation.

Import from `src.modules`, not the submodules directly. Full design rationale, formal equations,
and the acceptance-criteria targets: `artifacts/T18_lung_attention/T18_module_card.md`.

**Cross-backbone transfer (P13):** `build_model(backbone_name=...)` covers all four backbones.
ResNet50 (T23) and EfficientNet-B0 (T25) use `DenseNetLungAttention` unchanged. ViT-Base (T26) gets
`ViTLungAttention`, selected automatically for `backbone_name` starting with `vit`: timm's default
ViT head reads only the CLS token, which a spatial gate over the patch tokens never reaches, so the
wrapper reshapes the 196 patch tokens to a 14×14 map, gates it and **average-pools** into the head.
Arm A0 uses the same avg-pool head, so it is the control for A1–A6 and is not T17's CLS-token
baseline. The wrapper exposes `pre_attn`/`post_attn` identity modules as Grad-CAM taps
(`gradcam.py::get_taps` returns them for ViT, since the token→grid reshape happens in the wrapper,
not in a backbone submodule). Config: `configs/vit_base_lung_attention.yaml`; tests:
`tests/test_vit_lung_attention.py`; Kaggle notebook: `notebooks/T_26_Vit_Base_Model.ipynb`
(generated from the T23 notebook, same resumable structure and outputs). It was run on Kaggle on
2026-10-02; results, the executed notebook and a reviewed module card are in
`artifacts/T26_vit_base_lung_attention/` (winner A3_multiply; the 11 checkpoints are not committed).
`notebooks/archive/T_26_Vit_Base_Model_local_run_2026-10-01.ipynb` is the T26 owner's earlier self-contained notebook (own model
class, batch 8, early stopping on val macro-F1); its results are in `artifacts/vit_lung_attention/`,
are not comparable with the T26 protocol above, and are superseded by the Kaggle run. Design background:
`T22_T23_Cross_Backbone_Shortcut_Suppression.md` in the parent workspace's `Claude Working Files/`
(outside this repo).

### `artifacts/` — committed, reproducible run outputs

`.gitignore` ignores `artifacts/*` and re-includes each experiment folder by name, so **a new
`artifacts/<folder>/` is silently untracked until a `!artifacts/<folder>/` line is added**.
`*.pt`/`*.pth` stay ignored everywhere; never commit weights. What is committed:
`artifacts/splits/` holds the fixed split manifest that `src/datasets` loads (21,165 images; the
test split is 3,175), the per-backbone training folders (`T18_lung_attention/`,
`T23_resnet50_lung_attention/`, `T25_efficientnet_b0_lung_attention/`,
`T26_vit_base_lung_attention/`) hold `runs/<arm>/per_image_predictions.csv`, comparison tables and
a module card each, and the evaluation folders (`T28_calibration/`, `T31_dependence/`,
`T33_*`, `T35_efficiency/`, `cnn_closeout/`, `explainable_ai/`) each carry a `README.md` stating
how and when they were produced. `artifacts/calibration/` (Kaggle, seed 42) and
`artifacts/vit_lung_attention/` are earlier runs superseded by `T28_calibration/` and
`T26_vit_base_lung_attention/`. Treat anything here as a checked-in reproducibility artifact, not
scratch space: later analyses read these CSVs as inputs.

### Checkpoint evaluation — the inference-only path

Everything after training goes through one pattern, used by `scripts/run_cnn_closeout_inference.py`
and by the all-model notebooks (`notebooks/T28_calibration_all_models.ipynb`,
`T31_dependence_all_models.ipynb`, `T35_efficiency_all_models.ipynb`):

1. Read `config` from inside the checkpoint and rebuild the model with `build_model(...)` from its
   `model`/`module` blocks; load `model_state_dict` with `strict=True`; check `class_names` order
   (`COVID, Lung_Opacity, Normal, Viral Pneumonia`).
2. Recompute test predictions and confirm they reproduce the accuracy committed in that
   backbone's `artifacts/T18|T23|T25|T26_*` folder (`checkpoint_verification.csv`). No new number
   is produced from a checkpoint that fails this.
3. Fit anything fittable (temperature) on validation only; report on test.

Keep inputs fp32: fp16 inputs flip about 2% of predictions. The notebooks default to Kaggle paths
and run off Kaggle through `TRUST_DATA_ROOT`, `TRUST_CKPT_ROOT` (searched recursively for `*.pt`),
`TRUST_WORK`, `TRUST_REPO` (skips the `git clone`), `TRUST_DRY_RUN` (first two checkpoints only)
and `TRUST_WORKERS`. `docs/cnn_inference_runbook.md` documents the script route;
`docs/KAGGLE_RUN_HANDOFF.md` was written before the checkpoints were available locally and is
partly superseded.

External-test results follow a protocol written before any model saw the images
(`PROTOCOL.md` in each `artifacts/T33_external_*` folder), including a pixel-overlap check against
the primary dataset. The POLCOVID conversion and scoring scripts are not in this repo.

### `scripts/` — builders and runners

Three kinds, and the distinction matters for what may be re-run freely:

- **Builders from committed artifacts** (no model, no images, deterministic):
  `build_cnn_closeout.py` (authoritative for `artifacts/cnn_closeout/`),
  `build_explainability_baseline.py`, `build_xai_swap_pairs.py`, `build_t26_vit_tables.py`,
  `summarize_cnn_inference.py`, `build_t45_slides.py`, `create_research_execution_brief.py`.
  Regenerate their outputs by running them, not by editing the outputs.
- **Checkpoint runners** (need weights and the dataset): `run_cnn_closeout_inference.py`
  (`--tasks` from calibration, counterfactual, occlusion, cam, efficiency, dependence; its
  `SPECS`/`CLASS_NAMES`/`load_model` are imported by the other runners), `run_rsna_external.py`
  (binary RSNA scoring; see the RSNA finding above before using its output),
  `run_t35_efficiency.py`, `audit_checkpoint_archives.py`, `audit_cnn_closeout_readiness.py`.
- **Kaggle training entry points:** `kaggle_t25_efficientnet_b0_lung_attention.py` (the T25
  runner) and `make_t26_notebook.py`, which generates `notebooks/T_26_Vit_Base_Model.ipynb` from
  the T23 notebook by asserted cell rewrites. Change the T26 notebook through the generator.

`notebooks/efficiency.py` is the shared efficiency harness (`benchmark_model`), imported as a
module by scripts and notebooks despite living in `notebooks/`.

### `configs/` — YAML-driven experiments

`configs/baseline.yaml` is the reference config and is treated as **read-only** — copy it
(`resnet50_aug_v2.yaml`, `densenet121_baseline.yaml`, ...) for any new experiment rather than
editing it in place. Every important hyperparameter belongs in a config file, not hardcoded in a
notebook cell. See `configs/README.md` for the copy-naming convention and versioning fields
(`dataset.split_version`, `augmentation.version` must correspond to real, reproducible artifacts).
Never put secrets in a config file. Each `*_lung_attention.yaml` describes arm A2 of its
backbone; the other arms are produced at run time with `merge_overrides()`. Do not fork a YAML per
arm, since that lets the arms of an ablation drift apart.

### `docs/experiment_policy.md` — the cross-team contract

This is the shared standard every model owner follows regardless of architecture or training
platform (Kaggle vs Colab):

- Default seed is **42**; only deviate when deliberately testing seed sensitivity.
- All models train/eval on the *same* saved split manifest (never regenerate per-model) — split
  must be patient-level, not image-level, where patient IDs are available.
- The test set is touched only for final reported numbers, never for hyperparameter/model/threshold
  selection — use validation for all of that.
- Run naming is always `{model}_{experiment}_seed{seed}` via `generate_run_name()` — never `test`,
  `final`, `final2`.
- Multi-seed repeats (42, 123, 2026) are reserved for final/paper-reported comparisons, not every
  exploratory run.

`docs/wandb_setup.md` covers W&B auth (Colab/Kaggle secrets, never a pasted API key) and how to log
runs using the `src/utils` wrapper rather than calling `wandb` directly.

### Notebooks vs `src/`

Per-architecture training/HP-tuning still lives directly in `notebooks/` (ResNet50, DenseNet121,
EfficientNet-B0 baselines). T18's shortcut-suppression module is the one deliberate exception to
"heavy model code stays in the notebook": the module, training loop, and evaluation harness live
in `src/modules/`, and `notebooks/T18_lung_region_attention.ipynb` only orchestrates calls into
it — because T23 needs to reuse the exact same module code on ResNet50 later, and a notebook-local
implementation can't be imported. When adding reusable pipeline code (dataset/manifest loading,
preprocessing, splitting, evaluation) — or a model/module intended for reuse across
architectures — prefer putting it in `src/` so multiple model owners' notebooks can import it,
consistent with how `src/utils`, `src/datasets`, and `src/modules` are already used.
`notebooks/` used to be split across a second `kusal-notebooks/` directory (one member's
baseline-CNN/AuxSeg/EfficientNet-B0 work, including `efficiency.py`, the T34 efficiency-benchmark
harness); it has since been merged in — everything lives in `notebooks/` and `artifacts/` now.
