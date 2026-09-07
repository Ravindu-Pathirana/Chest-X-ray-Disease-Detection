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

**Current state:** the shared experiment-tracking/reproducibility infrastructure (`src/utils/`) and
the shared mask-paired data pipeline (`src/datasets/`) are built and tested. Model training is
happening per-architecture in `notebooks/` (ResNet50 baseline + HP tuning; DenseNet121 AuxSeg; the
T18 lung-region-attention candidate). `src/modules/` (shortcut-suppression modules, starting with
T18's Lung-Region Attention Module) is in progress. Don't assume `data/`, `models/`, or `results/`
directories exist yet — they are `.gitignore`d / not-yet-created per the README's planned layout;
committed run artifacts instead live under `artifacts/` (see below).

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

### `src/datasets/` — shared mask-paired data pipeline

`src/datasets/covid_cxr.py` (import via `from src.datasets import ...`, per the same
re-export convention as `src/utils`) is the one dataset pipeline every mask-aware model owner
should share, ported verbatim from the AuxSeg (Candidate B) notebook so T18 and future work stay
aligned:

- `JointTransform` applies identical spatial augmentation (resize/pad/crop/flip/rotate) to an
  (image, mask) pair from the same random draw; masks always use `NEAREST` interpolation and are
  zero-padded, never reflected.
- `CXRWithMaskDataset` returns `(image, label, mask)` triples, locating each mask by swapping
  `images/` for `masks/` one directory up (`COVID/images/COVID-1.png` → `COVID/masks/COVID-1.png`)
  and raising `FileNotFoundError` loudly on a missing mask rather than skipping it.
- `load_split_indices_from_manifest` reads the committed split manifest
  (`artifacts/splits/split_manifest_v1.csv`) so training loads a fixed split instead of
  recomputing one — this is what actually enforces `docs/experiment_policy.md`'s "same split for
  every model" rule. `stratified_split` (the 70/15/15 stratified fallback) exists mainly for
  regenerating the manifest itself.
- `compute_class_weights` (`w_c = N / (K * n_c)`) and `build_dataloaders` (wires all of the above
  into train/val/test `DataLoader`s, with reproducible worker seeding on the shuffling train
  loader only) round out the module.

### `src/modules/` — shortcut-suppression modules

Architecture-agnostic modules implementing the candidate shortcut-suppression techniques (see
README's "second research thread"), designed to be reused across backbones rather than tied to one
notebook — e.g. T18's Lung-Region Attention Module, built on DenseNet121 but intended for reuse on
ResNet50.

### `artifacts/` — committed, reproducible run outputs

Unlike `data/`/`models/`/`results/` (gitignored, not yet created), `artifacts/` **is** committed:
`artifacts/splits/` holds the fixed split manifest(s) that `src/datasets` loads, and
`artifacts/<experiment>/runs/<run_name>/` holds small per-run outputs tied to a specific
experiment. Treat anything here as a checked-in reproducibility artifact, not scratch space.

### `src/utils/` — shared infrastructure, not model-specific

Every team member's training notebook (any architecture) imports from here rather than
reimplementing config/seeding/tracking/checkpointing per-model:

- `config.py` — `load_config(path)` parses a YAML into a plain dict; `merge_overrides(config, {"training.batch_size": 64})` returns a modified copy via dotted-key paths (for one-off notebook tweaks without a new file).
- `reproducibility.py` — `set_seed(seed)` seeds Python/NumPy/PyTorch(CPU+CUDA) and sets `cudnn.deterministic=True` by default; `create_generator(seed)` + `seed_worker` are passed to any shuffling `DataLoader`. `strict=True` on `set_seed` enables `torch.use_deterministic_algorithms` — slower, only for chasing nondeterminism bugs, not standard runs.
- `experiment_tracking.py` — thin W&B wrapper: `initialize_wandb(config, ...)` (flattens config + logs git commit/python/torch/CUDA metadata), `log_metrics`, `log_summary_metrics`, `finish_run`, `generate_run_name(model, experiment, seed)` → `"{model}_{experiment}_seed{seed}"`.
- `checkpointing.py` — `BestCheckpointSaver` tracks one monitored metric across epochs and writes a checkpoint only on improvement; `load_checkpoint` reads it back. Architecture-agnostic (takes any `nn.Module`/optimizer).

All of it is re-exported from `src/utils/__init__.py`; import from there (`from src.utils import ...`), not from the submodules directly.

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

Heavy experiment/model code (per-architecture training loops, HP tuning, candidate
shortcut-suppression notebooks, the T34 efficiency-benchmark harness `notebooks/efficiency.py`)
lives in `notebooks/`, while `src/` holds infrastructure and pipeline code meant to be shared
across notebooks (`src/utils`, `src/datasets`, `src/modules`). When code originates in a notebook
but is reusable by more than one model owner (as `src/datasets/covid_cxr.py` was promoted from the
AuxSeg notebook), move it into `src/` rather than copy-pasting it into each new notebook.
`notebooks/` used to be split across a second `kusal-notebooks/` directory (one member's
baseline-CNN/AuxSeg/EfficientNet-B0 work); it has since been merged in — everything lives in
`notebooks/` and `artifacts/` now.
