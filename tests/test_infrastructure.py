"""Lightweight tests proving the experiment-tracking infrastructure works.

These do NOT train a real chest X-ray model -- they use a tiny dummy PyTorch
model and synthetic tensors to check that config loading, seeding, run
naming, metric logging, and checkpointing all work together. W&B runs in
"disabled" mode throughout, so this suite needs no API key and makes no
network calls.

Run with:
    pytest tests/test_infrastructure.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import (
    BestCheckpointSaver,
    create_generator,
    finish_run,
    generate_run_name,
    initialize_wandb,
    load_checkpoint,
    load_config,
    log_metrics,
    log_summary_metrics,
    merge_overrides,
    seed_worker,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"


class TinyDummyModel(nn.Module):
    """Minimal stand-in for ResNet50/DenseNet121/etc. -- just enough to exercise
    the optimizer/checkpoint/logging plumbing without real image data."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# --- config loading -----------------------------------------------------


def test_load_config_reads_baseline_yaml():
    config = load_config(BASELINE_CONFIG)
    assert config["experiment"]["seed"] == 42
    assert config["tracking"]["provider"] == "wandb"
    assert config["checkpoint"]["monitor"] == "val_f1"


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(REPO_ROOT / "configs" / "does_not_exist.yaml")


def test_merge_overrides_does_not_mutate_input():
    config = {"training": {"batch_size": 32}}
    updated = merge_overrides(config, {"training.batch_size": 64})
    assert config["training"]["batch_size"] == 32
    assert updated["training"]["batch_size"] == 64


# --- reproducibility -----------------------------------------------------


def test_set_seed_gives_reproducible_torch_output():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_seeded_generator_gives_reproducible_shuffle():
    gen1 = create_generator(42)
    perm1 = torch.randperm(20, generator=gen1).tolist()

    gen2 = create_generator(42)
    perm2 = torch.randperm(20, generator=gen2).tolist()

    assert perm1 == perm2


def test_seed_worker_runs_without_error():
    torch.manual_seed(123)
    seed_worker(0)  # DataLoader calls this per worker; check it doesn't raise


# --- run naming -----------------------------------------------------


def test_generate_run_name_is_human_readable():
    assert generate_run_name("resnet50", "baseline", 42) == "resnet50_baseline_seed42"


def test_generate_run_name_with_timestamp_keeps_base_prefix():
    name = generate_run_name("densenet121", "aug_v1", 123, timestamp=True)
    assert name.startswith("densenet121_aug_v1_seed123_")


# --- W&B logging interface (disabled mode, no network/API key) -----------


def test_wandb_disabled_run_logs_metrics_and_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(BASELINE_CONFIG)

    run = initialize_wandb(config, run_name="test_infra_run", mode="disabled")
    assert run is not None

    log_metrics({"epoch": 1, "train_loss": 0.5, "val_loss": None})  # None dropped, no error
    log_summary_metrics({"best_epoch": 1, "best_val_f1": 0.9})
    finish_run()


# --- checkpointing -----------------------------------------------------


def test_best_checkpoint_saver_saves_only_on_improvement(tmp_path):
    saver = BestCheckpointSaver(
        run_name="tinymodel_test_seed42", monitor="val_f1", mode="max", checkpoint_dir=tmp_path
    )
    model = TinyDummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    improved = [
        saver.step(epoch=1, metric_value=0.70, model=model, optimizer=optimizer, config={"x": 1}),
        saver.step(epoch=2, metric_value=0.65, model=model, optimizer=optimizer, config={"x": 1}),
        saver.step(epoch=3, metric_value=0.80, model=model, optimizer=optimizer, config={"x": 1}),
    ]

    assert improved == [True, False, True]
    assert saver.best_epoch == 3
    assert saver.best_metric == 0.80

    checkpoint = load_checkpoint(saver.checkpoint_path)
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_metric"] == 0.80
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint

    assert saver.summary() == {"best_epoch": 3, "best_val_f1": 0.80}


def test_best_checkpoint_saver_min_mode(tmp_path):
    saver = BestCheckpointSaver(run_name="tinymodel_min_seed42", monitor="val_loss", mode="min", checkpoint_dir=tmp_path)
    model = TinyDummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    assert saver.step(epoch=1, metric_value=1.0, model=model, optimizer=optimizer) is True
    assert saver.step(epoch=2, metric_value=1.5, model=model, optimizer=optimizer) is False
    assert saver.step(epoch=3, metric_value=0.5, model=model, optimizer=optimizer) is True
    assert saver.best_metric == 0.5


def test_checkpoint_saver_rejects_invalid_mode(tmp_path):
    with pytest.raises(ValueError):
        BestCheckpointSaver(run_name="x", monitor="val_f1", mode="sideways", checkpoint_dir=tmp_path)


# --- end-to-end smoke test -----------------------------------------------


def test_end_to_end_infrastructure_smoke(tmp_path, monkeypatch):
    """Runs a tiny 2-epoch loop over synthetic tensors, touching every piece:
    config -> seed -> model/optimizer -> W&B (disabled) -> checkpointing.
    """
    monkeypatch.chdir(tmp_path)
    config = load_config(BASELINE_CONFIG)
    set_seed(config["experiment"]["seed"])

    model = TinyDummyModel(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    run_name = generate_run_name(
        config["model"]["name"], config["experiment"]["name"], config["experiment"]["seed"]
    )
    initialize_wandb(config, run_name=run_name, mode="disabled")

    saver = BestCheckpointSaver(
        run_name=run_name,
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
        checkpoint_dir=tmp_path / "checkpoints",
    )

    synthetic_val_f1_by_epoch = [0.55, 0.72]  # synthetic, not from a real evaluation
    for epoch, val_f1 in enumerate(synthetic_val_f1_by_epoch, start=1):
        inputs = torch.randn(4, 8)
        targets = torch.randint(0, 4, (4,))

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        log_metrics({"epoch": epoch, "train_loss": loss.item(), "val_f1": val_f1})
        saver.step(epoch=epoch, metric_value=val_f1, model=model, optimizer=optimizer, config=config)

    log_summary_metrics(saver.summary())
    finish_run()

    assert saver.best_epoch == 2
    assert saver.checkpoint_path.exists()
