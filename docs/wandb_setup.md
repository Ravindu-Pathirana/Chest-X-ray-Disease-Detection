# W&B Setup Guide

This guide explains how to use [Weights & Biases](https://wandb.ai) (W&B) —
the tool this project uses to track training runs — from scratch.

## What is W&B, and why are we using it?

W&B is a hosted dashboard for machine learning experiments. Every time
someone trains a model, the training script sends metrics (loss, accuracy,
F1, AUROC, ...) and configuration (model name, learning rate, seed, ...) to
W&B, which plots them and lets you compare runs side by side.

We're training on Kaggle and Colab, which are temporary environments — the
notebook session eventually shuts down and local files disappear. W&B gives
us a permanent, shared record of every run so the team can compare models
without anyone needing to keep their own spreadsheet of results.

**Important:** W&B stores run *history*, not the recipe to reproduce a run.
The actual reproduction recipe is the code + config in this GitHub repo plus
the dataset split manifest. See `docs/experiment_policy.md`.

## Joining the shared project

1. Create a free account at [wandb.ai](https://wandb.ai) if you don't have one.
2. Ask whoever created the team's W&B project (`chest-xray-disease-classification`)
   to invite your account, or share the team/entity name with you.
3. Once invited, runs logged with `project="chest-xray-disease-classification"`
   (the default in `configs/baseline.yaml`) will appear in the shared dashboard.

## Installing wandb

```bash
pip install wandb
```

This is already listed in `requirements.txt`.

## Authenticating

**Never put your W&B API key in a notebook cell, a config file, or commit it
to GitHub.** Anyone with the key can log runs (or worse, delete them) under
your account. `.gitignore` in this repo excludes `.env` and `wandb/` for
exactly this reason.

Get your personal API key from <https://wandb.ai/authorize> (you'll need to
log in to your own W&B account to see it).

### From Google Colab

The simplest option — this opens a login prompt/link and doesn't require you
to paste the key into a cell that gets saved:

```python
import wandb
wandb.login()
```

Alternatively, store the key as a Colab secret (the key icon in the left
sidebar) and read it into the environment without ever displaying it:

```python
from google.colab import userdata
import os
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

### From Kaggle

Use **Kaggle Secrets** (Add-ons → Secrets in the notebook editor) to store
your key, then load it into the environment at the start of the notebook:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

`wandb.init()` will then authenticate automatically using the environment
variable — no `wandb.login()` call needed.

### Running without any account (offline / disabled)

For quick local testing or CI, skip authentication entirely:

```python
initialize_wandb(config, mode="disabled")  # no network calls, no login needed
# or
initialize_wandb(config, mode="offline")   # logs to local files, sync later with `wandb sync`
```

## Starting a run

Use the project helper rather than calling `wandb.init` directly, so run
naming and config logging stay consistent across everyone's notebooks:

```python
from src.utils import initialize_wandb, generate_run_name

run_name = generate_run_name(config["model"]["name"], config["experiment"]["name"], config["experiment"]["seed"])
run = initialize_wandb(config, run_name=run_name)
```

## Logging metrics

Call this once per epoch (or per step) with whatever metrics are available:

```python
from src.utils import log_metrics

log_metrics({
    "epoch": epoch + 1,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
    "val_precision": val_precision,
    "val_recall": val_recall,
    "val_f1": val_f1,
    "val_auroc": val_auroc,
    "learning_rate": current_lr,
})
```

You can pass a partial set of metrics (e.g. only `train_loss` if validation
hasn't run yet) — `None` values are dropped automatically.

## Finishing a run

Always finish the run at the end of the notebook, including in the `except`
path if training fails, so the run doesn't hang in an unfinished state:

```python
from src.utils import finish_run

finish_run()
```

## Comparing runs and finding the best one

In the W&B dashboard:

1. Open the `chest-xray-disease-classification` project.
2. Use the **Runs table** to sort by a summary metric, e.g. `best_val_f1`
   or `best_val_auroc` (written by `log_summary_metrics()` at the end of
   training).
3. Use the **Charts** tab to overlay `val_loss` / `val_f1` curves across
   runs, filtered by tags or config fields (e.g. `model_name`, `seed`).
4. Group by `model_name` to compare architectures, or by `seed` to check
   seed sensitivity for a fixed architecture.

The run with the best `best_val_<metric>` summary value (matching
`checkpoint.monitor` in the config) is the current best model for that
metric — but always sanity-check it wasn't selected using the held-out test
set (see `docs/experiment_policy.md`).

## Why API keys must never be committed

A committed API key is visible to anyone with read access to the GitHub
repo (and to the entire internet, if the repo is public) forever — even if
you delete it in a later commit, it remains in the Git history. Anyone with
the key can impersonate you on W&B: log fake runs, or delete your real ones.
Always use `wandb.login()`, Colab secrets, or Kaggle secrets instead.
