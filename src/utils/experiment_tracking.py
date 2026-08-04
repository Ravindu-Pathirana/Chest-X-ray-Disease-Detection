"""Weights & Biases experiment tracking utilities.

Thin wrapper around the `wandb` API so training code doesn't duplicate
config-flattening, run-naming, or environment/git metadata logic across
notebooks. Nothing here is specific to any one model architecture -- the
same functions work for ResNet50, DenseNet121, EfficientNet, ViT, etc.

W&B holds experiment *history* (metrics, hyperparameters logged for a run).
It is not the source of truth for how to reproduce a run -- that's the
combination of this repository's code/config plus the dataset split
manifest. See docs/experiment_policy.md.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_git_commit_hash(repo_path: Optional[str] = None) -> Optional[str]:
    """Return the current Git commit hash, or None if it can't be determined.

    Fails gracefully: returns None (rather than raising) when not run inside
    a Git repository, when git isn't installed, or on any other lookup
    failure. Training must never crash just because commit metadata
    couldn't be captured.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_environment_metadata() -> Dict[str, Any]:
    """Collect Python/PyTorch/CUDA/GPU/Git metadata for reproducibility logging."""
    metadata: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "git_commit": get_git_commit_hash(),
    }
    try:
        import torch

        metadata["pytorch_version"] = torch.__version__
        cuda_available = torch.cuda.is_available()
        metadata["cuda_available"] = cuda_available
        metadata["cuda_version"] = torch.version.cuda if cuda_available else None
        metadata["gpu_name"] = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        metadata["pytorch_version"] = None
        metadata["cuda_available"] = False
        metadata["cuda_version"] = None
        metadata["gpu_name"] = None
    return metadata


def generate_run_name(model_name: str, experiment_name: str, seed: int, timestamp: bool = False) -> str:
    """Build a standard, human-readable run name: `{model}_{experiment}_seed{seed}`.

    Example:
        generate_run_name("resnet50", "baseline", 42) -> "resnet50_baseline_seed42"

    Avoid manually naming runs "test", "final", "final2", etc. -- this
    function exists so every official run has a name that says what it is.
    """
    parts = [str(model_name), str(experiment_name), f"seed{seed}"]
    name = "_".join(p for p in parts if p)
    if timestamp:
        name += "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    return name


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested baseline.yaml structure into the fields W&B should record."""
    experiment = config.get("experiment", {}) or {}
    model = config.get("model", {}) or {}
    dataset = config.get("dataset", {}) or {}
    training = config.get("training", {}) or {}
    scheduler = config.get("scheduler", {}) or {}
    augmentation = config.get("augmentation", {}) or {}
    checkpoint = config.get("checkpoint", {}) or {}

    return {
        "experiment_name": experiment.get("name"),
        "seed": experiment.get("seed"),
        "model_name": model.get("name"),
        "pretrained": model.get("pretrained"),
        "num_classes": model.get("num_classes"),
        "dataset_name": dataset.get("name"),
        "dataset_version": dataset.get("version"),
        "split_version": dataset.get("split_version"),
        "image_size": dataset.get("image_size"),
        "epochs": training.get("epochs"),
        "batch_size": training.get("batch_size"),
        "learning_rate": training.get("learning_rate"),
        "optimizer": training.get("optimizer"),
        "weight_decay": training.get("weight_decay"),
        "scheduler": scheduler.get("name"),
        "augmentation_version": augmentation.get("version"),
        "checkpoint_monitor": checkpoint.get("monitor"),
        "checkpoint_mode": checkpoint.get("mode"),
    }


def initialize_wandb(
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    mode: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    """Start a W&B run and log the standard config/environment metadata.

    Args:
        config: The parsed baseline.yaml (or a variant) dictionary.
        run_name: Explicit run name. If None, generated with
            `generate_run_name(model.name, experiment.name, experiment.seed)`.
        mode: Passed to `wandb.init(mode=...)`. Use "offline" or "disabled"
            for local testing / environments without network access or an
            API key (falls back to `config["tracking"]["mode"]` if not
            given, then to W&B's normal "online" default).
        tags: Optional list of W&B tags, useful for filtering runs in the
            dashboard (e.g. ["baseline", "cnn"]).

    Returns:
        The active `wandb.run` object.
    """
    import wandb

    tracking = config.get("tracking", {}) or {}
    model = config.get("model", {}) or {}
    experiment = config.get("experiment", {}) or {}

    if run_name is None:
        run_name = generate_run_name(
            model.get("name", "model"),
            experiment.get("name", "experiment"),
            experiment.get("seed", 0),
        )

    wandb_config = _flatten_config(config)
    wandb_config.update(get_environment_metadata())

    run = wandb.init(
        project=tracking.get("project", "chest-xray-disease-classification"),
        entity=tracking.get("entity"),
        name=run_name,
        config=wandb_config,
        mode=mode or tracking.get("mode"),
        tags=tags,
    )
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a (possibly partial) dict of metrics for the current epoch/step.

    Any subset of the standard metrics may be passed -- e.g. only
    `train_loss` before validation has run. `None` values are dropped
    rather than sent to W&B, so callers don't need to pre-filter.

    Supports per-class metric keys such as "val_auroc/Pneumonia" for
    multi-label disease classification -- W&B groups metrics by the part of
    the key before the slash in the dashboard.
    """
    import wandb

    if wandb.run is None:
        raise RuntimeError("No active W&B run. Call initialize_wandb() first.")
    clean_metrics = {k: v for k, v in metrics.items() if v is not None}
    wandb.log(clean_metrics, step=step)


def log_summary_metrics(metrics: Dict[str, Any]) -> None:
    """Write final/best-of-run values (e.g. best_epoch, best_val_f1) to the run summary.

    Summary fields are what W&B's run-comparison table shows by default, so
    this is how "which epoch produced the best model" gets surfaced there.
    """
    import wandb

    if wandb.run is None:
        raise RuntimeError("No active W&B run. Call initialize_wandb() first.")
    for key, value in metrics.items():
        if value is not None:
            wandb.run.summary[key] = value


def finish_run() -> None:
    """Finish the active W&B run, if any. Safe to call even if no run is active."""
    import wandb

    if wandb.run is not None:
        wandb.finish()
