# Experiment Reproducibility Policy

This document is the shared standard for training and tracking models in
this project. Every group member training a model (ResNet50, DenseNet121,
EfficientNet, ViT, or anything else) should follow it, regardless of whether
training happens on Kaggle or Google Colab.

The goal: any official run can be reproduced by another team member using
only the code and config in this GitHub repository, plus the dataset and
its split manifest. **W&B stores history, not the reproduction recipe.**

---

## Default Seed

The default baseline seed is:

```
42
```

All baseline model comparisons (e.g. "how does ResNet50 compare to
DenseNet121 under the same conditions?") should use seed 42 unless the
experiment is specifically investigating seed sensitivity.

## Randomness

Every experiment must seed:

- Python's `random` module
- NumPy
- PyTorch (CPU and CUDA, all devices)
- DataLoader workers

Use `src/utils/reproducibility.py` for all of this — call `set_seed(seed)`
once at the start of the script/notebook, and use `create_generator(seed)` +
`seed_worker` when constructing any `DataLoader` that shuffles data. Don't
seed manually in individual notebooks; that's how seeding gets missed.

Note that `cudnn.deterministic=True` (the baseline default) gives
reproducibility on the *same* hardware/software stack, not bit-identical
results across different GPUs or CUDA versions. A stricter mode
(`torch.use_deterministic_algorithms`) is available via `set_seed(seed,
strict=True)` but can slow down training and isn't required for standard
runs — see the docstring in `reproducibility.py` for when it's worth the
cost.

## Dataset Split

All models must be trained and evaluated on the **same** saved
train/validation/test split. Do not regenerate the split for each model —
that would silently change what's being compared.

Policy:

1. Dataset splitting is performed **once** per split version, using the
   project's fixed seed (42).
2. The resulting split is saved as a **manifest** — a CSV mapping each image
   to its assigned split — not regenerated at training time.
3. Every model's config references that manifest via `dataset.split_version`
   (e.g. `v1`), and every training run loads the same file.
4. Where patient identifiers are available, splitting must be done at the
   **patient level**, not the image level. Chest X-ray datasets often
   contain multiple images per patient; splitting by image risks the same
   patient appearing in both train and test, which leaks information and
   inflates reported performance. This is a well-documented failure mode in
   medical imaging ML.
5. The **test set must never be used for model selection or hyperparameter
   tuning** (see below).

A split manifest should look like:

```csv
image_path,patient_id,split,label
path/to/image1.png,P001,train,Pneumonia
path/to/image2.png,P002,val,Normal
path/to/image3.png,P003,test,Cardiomegaly
```

This repository does not yet implement dataset-specific splitting logic — the
actual dataset and patient-ID availability aren't finalized. What's fixed
here is the **policy and interface**: whoever implements the split loader
must produce a manifest in this shape, versioned via `dataset.split_version`
in `configs/baseline.yaml`, and every training notebook must load that
manifest rather than calling a random-split function inline.

## Test Set

The test set exists only to report final performance. It must not be used
to choose:

- hyperparameters (learning rate, batch size, optimizer, ...)
- number of epochs / early stopping point
- decision thresholds
- augmentation configuration
- model architecture

Use the **validation** split for all of these decisions. Touch the test set
only for the final reported numbers.

## Configuration

Every important hyperparameter must live in a YAML file under `configs/`,
not be hardcoded in a notebook cell. See `configs/README.md` for the
convention. The config is logged to W&B automatically by
`initialize_wandb()` — see `src/utils/experiment_tracking.py`.

## Run Tracking

Every official experiment (i.e. one whose results might go in the paper or
a comparison table) must be logged to the shared W&B project
(`chest-xray-disease-classification`). Scratch/debugging runs can use
`mode="disabled"` or `mode="offline"` to avoid cluttering the dashboard.

## Naming

Run names follow:

```
{model}_{experiment}_seed{seed}
```

Examples:

- `resnet50_baseline_seed42`
- `densenet121_baseline_seed42`
- `efficientnet_b0_aug_v1_seed42`

Use `generate_run_name()` from `src/utils/experiment_tracking.py` rather
than typing names by hand. Do not use names like `test`, `new_test`,
`final`, `final2`, or `final_final` — they carry no information and make the
W&B dashboard unusable for comparison.

## Required Information

Each official run should record (most of this happens automatically via
`initialize_wandb()`):

- model
- seed
- dataset version
- split version
- image size
- augmentation version
- batch size
- epochs
- learning rate
- optimizer
- weight decay
- scheduler
- training loss (per epoch)
- validation loss (per epoch)
- relevant validation metrics (accuracy, precision, recall, F1, AUROC)
- best epoch
- best validation metric

## Repeated Experiments

Seed 42 is the standard baseline seed for day-to-day development and
architecture comparisons.

For final, important model comparisons (the ones likely to appear in the
paper), repeat training using multiple predefined seeds if time and compute
allow, e.g.:

```
42
123
2026
```

Report the mean and standard deviation across seeds where appropriate,
rather than a single run's number.

Multi-seed training is **not** required for every exploratory experiment —
GPU time on Kaggle/Colab is limited, and requiring 3x the compute for every
quick check would slow the team down. Reserve it for results you intend to
report as final.
