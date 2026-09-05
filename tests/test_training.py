"""Tests for src/modules/training.py: run_epoch, train_phase, evaluate.

Uses a tiny synthetic, stratified dataset (no dependency on the real
~21k-image dataset) built with the same ImageFolder/CXRWithMaskDataset
pipeline as S2. wandb_enabled=False throughout -- these test the training
loop's own logic, not the W&B integration, and `wandb` isn't installed in
every environment these tests run in.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.datasets import CXRWithMaskDataset, JointTransform, compute_class_weights, only_images_folder
from src.modules import build_model, evaluate, freeze_backbone, train_phase

CLASSES = ["COVID", "Normal", "Lung_Opacity", "Viral_Pneumonia"]


def _synthetic_image_and_mask(size: int = 64):
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[:, : size // 2] = 220
    arr[:, size // 2 :] = 30
    mask_arr = np.zeros((size, size), dtype=np.uint8)
    mask_arr[:, : size // 2] = 255
    return Image.fromarray(arr).convert("L"), Image.fromarray(mask_arr).convert("L")


@pytest.fixture
def tiny_loaders(tmp_path):
    """15 images/class x 4 classes, stratified 70/30 split -- enough that
    every class appears in both splits (a naive non-stratified split on a
    dataset this small can leave a class entirely out of validation, which
    breaks sklearn's classification_report; caught for real while writing
    these tests, see the S6 commit message)."""
    for cls in CLASSES:
        (tmp_path / cls / "images").mkdir(parents=True)
        (tmp_path / cls / "masks").mkdir(parents=True)
        for i in range(15):
            img, mask = _synthetic_image_and_mask()
            img.save(tmp_path / cls / "images" / f"{cls}-{i}.png")
            mask.save(tmp_path / cls / "masks" / f"{cls}-{i}.png")

    base = ImageFolder(root=str(tmp_path), is_valid_file=only_images_folder)
    targets = np.array(base.targets)
    idx = np.arange(len(base))
    train_idx, val_idx = train_test_split(idx, test_size=0.3, stratify=targets, random_state=42)

    train_ds = CXRWithMaskDataset(base, train_idx, JointTransform(img_size=224, train=True))
    val_ds = CXRWithMaskDataset(base, val_idx, JointTransform(img_size=224, train=False))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(targets[train_idx], num_classes=4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return base, train_loader, val_loader, criterion


def _build_and_freeze(**kwargs):
    device = torch.device("cpu")
    model = build_model(num_classes=4, pretrained=False, **kwargs).to(device)
    freeze_backbone(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    return model, optimizer, device


def test_train_phase_runs_and_writes_history(tiny_loaders, tmp_path):
    _base, train_loader, val_loader, criterion = tiny_loaders
    model, optimizer, device = _build_and_freeze(use_attention=True, gate_mode="residual")
    out_dir = tmp_path / "run"

    model = train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler=None,
        device=device, epochs=2, patience=5, phase_name="phase1_frozen",
        output_dir=out_dir, lambda_att=0.5, wandb_enabled=False,
    )
    assert isinstance(model, nn.Module)

    hist_path = out_dir / "phase1_frozen_history.json"
    assert hist_path.exists()
    hist = json.loads(hist_path.read_text())
    assert len(hist) == 2
    expected_keys = {
        "epoch", "train_loss", "train_cls_loss", "train_att_loss", "train_acc", "train_ilar",
        "val_loss", "val_cls_loss", "val_att_loss", "val_acc", "val_ilar",
        "val_precision", "val_recall", "val_f1", "val_auroc",
    }
    assert expected_keys <= set(hist[0].keys())
    assert all(np.isfinite(row["train_loss"]) for row in hist)
    assert all(np.isfinite(row["val_f1"]) for row in hist)


def test_train_phase_att_loss_starts_near_log2_at_init(tiny_loaders, tmp_path):
    """Sanity-checks the WBS section 4.4a expectation from inside the real
    training loop, not just the loss function in isolation."""
    _base, train_loader, val_loader, criterion = tiny_loaders
    model, optimizer, device = _build_and_freeze(use_attention=True, gate_mode="residual")

    hist_dir = tmp_path / "run2"
    train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler=None,
        device=device, epochs=1, patience=5, phase_name="phase1_frozen",
        output_dir=hist_dir, lambda_att=0.5, wandb_enabled=False,
    )
    hist = json.loads((hist_dir / "phase1_frozen_history.json").read_text())
    assert abs(hist[0]["train_att_loss"] - 0.6931) < 0.05


def test_train_phase_arm_a0_has_nan_ilar(tiny_loaders, tmp_path):
    """Arm A0 (use_attention=False) has no attention map -- ilar must be
    NaN, not 0 (a 0 would silently read as 'all attention on background')."""
    _base, train_loader, val_loader, criterion = tiny_loaders
    model, optimizer, device = _build_and_freeze(use_attention=False)

    out_dir = tmp_path / "run_a0"
    train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler=None,
        device=device, epochs=1, patience=5, phase_name="phase1_frozen",
        output_dir=out_dir, lambda_att=0.0, wandb_enabled=False,
    )
    hist = json.loads((out_dir / "phase1_frozen_history.json").read_text())
    assert np.isnan(hist[0]["train_ilar"])
    assert np.isnan(hist[0]["val_ilar"])
    assert hist[0]["train_att_loss"] == 0.0  # lambda_att=0.0 -> total==cls, att detached to 0


def test_train_phase_early_stopping_fires(tiny_loaders, tmp_path):
    """With patience=1 and a model that can't improve (frozen everything,
    including the head, via lr=0), early stopping must fire well before
    the requested epoch count."""
    _base, train_loader, val_loader, criterion = tiny_loaders
    model, optimizer, device = _build_and_freeze(use_attention=True)
    for group in optimizer.param_groups:
        group["lr"] = 0.0  # no learning possible -> val_loss can only get (numerically) worse or equal

    out_dir = tmp_path / "run_stop"
    train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler=None,
        device=device, epochs=20, patience=1, phase_name="phase1_frozen",
        output_dir=out_dir, lambda_att=0.5, wandb_enabled=False,
    )
    hist = json.loads((out_dir / "phase1_frozen_history.json").read_text())
    assert len(hist) < 20, "early stopping should have fired well before the epoch cap"


def test_train_phase_best_val_f1_matches_best_epoch_not_global_max(tiny_loaders, tmp_path, monkeypatch):
    """best_val_f1 in the summary must be the F1 AT the val_loss-best epoch
    (the checkpoint actually kept), not the best F1 seen at any epoch --
    otherwise the summary would describe a model that wasn't saved."""
    _base, train_loader, val_loader, criterion = tiny_loaders
    model, optimizer, device = _build_and_freeze(use_attention=True)

    # Force a val_loss/val_f1 sequence where the best-loss epoch is NOT the
    # best-f1 epoch, and capture what train_phase actually reports via
    # log_summary_metrics's would-be payload (wandb_enabled=False skips the
    # real call, so patch run_epoch instead to control both metrics directly).
    import src.modules.training as training_module

    fake_val_metrics = [
        {"loss": 1.0, "cls_loss": 1.0, "att_loss": 0.0, "accuracy": 0.5, "ilar": 0.3,
         "precision": 0.5, "recall": 0.5, "f1": 0.9, "auroc": 0.6},  # best F1, worse loss
        {"loss": 0.5, "cls_loss": 0.5, "att_loss": 0.0, "accuracy": 0.6, "ilar": 0.4,
         "precision": 0.6, "recall": 0.6, "f1": 0.7, "auroc": 0.7},  # best loss, worse F1
    ]
    fake_train_metrics = {"loss": 0.1, "cls_loss": 0.1, "att_loss": 0.0, "accuracy": 0.9, "ilar": 0.5}

    calls = {"n": 0}

    def fake_run_epoch(model, loader, criterion, optimizer, device, train, **kwargs):
        if train:
            return dict(fake_train_metrics)
        idx = calls["n"]
        calls["n"] += 1
        return dict(fake_val_metrics[idx])

    monkeypatch.setattr(training_module, "run_epoch", fake_run_epoch)

    out_dir = tmp_path / "run_bestf1"
    train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler=None,
        device=device, epochs=2, patience=5, phase_name="phase1_frozen",
        output_dir=out_dir, lambda_att=0.5, wandb_enabled=False,
    )
    hist = json.loads((out_dir / "phase1_frozen_history.json").read_text())
    # epoch 2 has the better (lower) val_loss -- that's the kept checkpoint.
    assert hist[1]["val_loss"] == 0.5
    assert hist[1]["val_f1"] == 0.7  # confirms epoch 2's own f1 is what's in the history,
    # not epoch 1's higher 0.9 -- train_phase must not report the global-max f1 as "best".


def test_evaluate_writes_wbs_7_1_schema(tiny_loaders, tmp_path):
    _base, train_loader, val_loader, criterion = tiny_loaders
    model, _optimizer, device = _build_and_freeze(use_attention=True, gate_mode="residual")

    results = evaluate(model, val_loader, CLASSES, device, tmp_path, "phase2_finetune")

    for key in ("classification_report", "confusion_matrix", "roc_auc_macro", "roc_auc_per_class", "class_names"):
        assert key in results  # AuxSeg's original keys, unchanged
    for key in ("attention_ilar", "attention_dice", "attention_iou"):
        assert key in results  # T18 additions

    df = pd.read_csv(tmp_path / "phase2_finetune_summary_metrics.csv")
    assert list(df.columns) == [
        "accuracy", "macro_f1", "macro_recall", "macro_precision", "weighted_f1", "weighted_recall",
        "roc_auc_macro", "attention_ilar", "attention_dice", "attention_iou", "cam_eil_post", "cam_eil_pre",
    ]
    assert not pd.isna(df["attention_ilar"].iloc[0])  # model has an attention map
    assert pd.isna(df["cam_eil_post"].iloc[0])  # not computed here -- S11's job
    assert pd.isna(df["cam_eil_pre"].iloc[0])

    assert (tmp_path / "phase2_finetune_confusion_matrix.csv").exists()
    assert (tmp_path / "phase2_finetune_test_results.json").exists()


def test_evaluate_arm_a0_writes_nan_not_zero_for_attention_columns(tiny_loaders, tmp_path):
    _base, _train_loader, val_loader, _criterion = tiny_loaders
    model, _optimizer, device = _build_and_freeze(use_attention=False)

    results = evaluate(model, val_loader, CLASSES, device, tmp_path, "a0")
    assert results["attention_ilar"] is None

    df = pd.read_csv(tmp_path / "a0_summary_metrics.csv")
    assert pd.isna(df["attention_ilar"].iloc[0])
    assert pd.isna(df["attention_dice"].iloc[0])
    assert pd.isna(df["attention_iou"].iloc[0])
