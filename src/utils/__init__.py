"""Reusable experiment-tracking and reproducibility utilities.

Shared across model architectures (ResNet50, DenseNet121, EfficientNet, ViT,
...) so each group member's training notebook uses the same reproducibility
guarantees and the same W&B logging conventions. See docs/experiment_policy.md
for the policy these utilities implement.
"""

from .checkpointing import BestCheckpointSaver, load_checkpoint
from .config import load_config, merge_overrides
from .experiment_tracking import (
    finish_run,
    generate_run_name,
    get_environment_metadata,
    get_git_commit_hash,
    initialize_wandb,
    log_metrics,
    log_summary_metrics,
)
from .reproducibility import create_generator, seed_worker, set_seed

__all__ = [
    "load_config",
    "merge_overrides",
    "set_seed",
    "create_generator",
    "seed_worker",
    "initialize_wandb",
    "log_metrics",
    "log_summary_metrics",
    "finish_run",
    "generate_run_name",
    "get_git_commit_hash",
    "get_environment_metadata",
    "BestCheckpointSaver",
    "load_checkpoint",
]
