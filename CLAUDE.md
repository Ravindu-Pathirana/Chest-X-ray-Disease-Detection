# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A research benchmark comparing four transfer-learning architectures (ResNet50, DenseNet121,
EfficientNet-B0, ViT-Base/16) on chest X-ray disease classification (COVID-19 Radiography
dataset: COVID / Normal / Lung Opacity / Viral Pneumonia). The point isn't just accuracy — models
are compared across five trustworthiness axes (accuracy, calibration, explainability/faithfulness,
OOD robustness via the RSNA dataset, efficiency), because the primary dataset is known to contain
source bias (models can shortcut-learn hospital/scanner artifacts instead of disease features). A
second research thread develops and ablates shortcut-suppression techniques (lung-mask attention,
auxiliary segmentation head, Grad-CAM-penalty loss — see `notebooks/candidate-c-grad-cam-shortcut-suppression-loss.ipynb`)
using the dataset's supplied lung masks. Full methodology and the 17-stage pipeline plan (P01–P17)
are in `Project Documents/Each task description.md`; product framing is in `README.md`.

**Current state:** the shared experiment-tracking/reproducibility infrastructure (`src/utils/`) is
built and tested. The dataset split is fixed and committed (`artifacts/splits/split_manifest_v1.csv`,
verified against the real dataset and cross-checked against AuxSeg's committed test results).
Model training is happening per-architecture in `notebooks/` (ResNet50 baseline + HP tuning;
DenseNet121 baseline + AuxSeg candidate; EfficientNet-B0 baseline + tuning). T18's Lung-Region
Attention Module (Candidate A, `src/modules/`) is code-complete and locally verified (unit tests,
plus real diagnostic runs against the actual dataset for the smoke/overfit checks) but not yet
trained on Kaggle — see `artifacts/T18_lung_attention/T18_module_card.md` for status and handoff
notes. Don't assume `data/`, `models/`, or `results/` directories exist yet; they are `.gitignore`d
/ not-yet-created per the README's planned layout.

## Commands

```bash
# Setup (local only — Kaggle/Colab ship a GPU-matched torch preinstalled, don't pip install torch there)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full test suite
pytest

# Run a single test file / test
pytest tests/test_infrastructure.py -v
pytest tests/test_infrastructure.py::test_load_config_reads_baseline_yaml -v
```

`requirements.txt` intentionally excludes `torch`/`torchvision` and model-specific deps (`timm`,
`grad-cam`, `shap`, `scikit-learn`, ...) — it only covers the shared tracking/reproducibility layer
used by `src/utils`. `conftest.py` puts the repo root on `sys.path` so `from src.utils import ...`
works under pytest with no packaging setup.

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

T18's Lung-Region Attention Module (Candidate A), designed backbone-agnostic for T23. Split by
concern, one file per WBS build step:

- `lung_attention.py` — `LungRegionAttention` (the module itself), `CBAMSpatialAttention` (the
  published-method comparator), `DenseNetLungAttention`/`build_model()` (backbone-agnostic model
  wrapper — works for any timm architecture, not just DenseNet despite the filename),
  `freeze_backbone`/`unfreeze_final_blocks` (one small per-architecture-family registry covering
  densenet/resnet/efficientnet/vit), `LogitsOnly` (adapter for tools expecting `model(x) ->
  Tensor` — the model itself returns a 3-tuple `(logits, attention, attention_logits)`).
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
- `efficiency_check.py` — reads `kusal-notebooks/efficiency.py`'s benchmark output and applies a
  pass/fail threshold; does not duplicate that harness.
- `figures.py` — heat-map overlay selection/rendering for the paper's figures.

Import from `src.modules`, not the submodules directly. Full design rationale, formal equations,
and the acceptance-criteria targets: `Claude Working Files/T18_Lung_Region_Attention_WBS.md` and
`artifacts/T18_lung_attention/T18_module_card.md`.

### `configs/` — YAML-driven experiments

`configs/baseline.yaml` is the reference config and is treated as **read-only** — copy it
(`resnet50_aug_v2.yaml`, `densenet121_baseline.yaml`, ...) for any new experiment rather than
editing it in place. Every important hyperparameter belongs in a config file, not hardcoded in a
notebook cell. See `configs/README.md` for the copy-naming convention and versioning fields
(`dataset.split_version`, `augmentation.version` must correspond to real, reproducible artifacts).
Never put secrets in a config file.

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
