"""Focused protocol tests for the T25 command-line runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/kaggle_t25_efficientnet_b0_lung_attention.py"
SPEC = importlib.util.spec_from_file_location("t25_runner", SCRIPT)
t25 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(t25)


def test_all_a0_a6_configs_build_with_expected_loss_weights():
    arms = t25.arm_overrides(lambda_att=0.3, lambda_bg=0.5)
    assert list(arms) == list(t25.ARM_ORDER)
    assert arms["A0"]["module.use_attention"] is False
    assert arms["A1"]["module.lambda_att"] == 0.0
    assert arms["A2"]["module.gate_mode"] == "residual"
    assert arms["A3"]["module.gate_mode"] == "multiply"
    assert arms["A4"]["module.gate_mode"] == "none"
    assert arms["A5"]["module.attention"] == "cbam"
    assert arms["A6"]["module.lambda_bg"] == 0.5


def test_all_a0_a6_model_variants_construct():
    base = {
        "model": {"name": "efficientnet_b0", "num_classes": 4, "pretrained": False, "drop_rate": 0.0},
        "module": {"reduction": 16},
    }
    for overrides in t25.arm_overrides(0.3, 0.5).values():
        cfg = t25.apply_overrides(base, overrides)
        model = t25._build_model(cfg, pretrained=False)
        assert model.backbone.num_features == 1280


def test_validation_selection_ignores_test_metrics():
    rows = [
        {"lambda_att": 0.0, "val_macro_f1": 0.90, "val_ilar": 0.3, "test_macro_f1": 0.99},
        {"lambda_att": 0.5, "val_macro_f1": 0.896, "val_ilar": 0.4, "test_macro_f1": 0.01},
        {"lambda_att": 1.0, "val_macro_f1": 0.894, "val_ilar": 0.9, "test_macro_f1": 1.00},
    ]
    selected = t25.select_largest_within_f1_tolerance(rows, "lambda_att", 0.005)
    assert selected["lambda_att"] == 0.5


def test_loader_builder_requires_fixed_manifest(tmp_path):
    args = type("Args", (), {"repo_root": tmp_path, "data_dir": tmp_path, "num_workers": 0})()
    cfg = {
        "experiment": {"seed": 42},
        "dataset": {"split_manifest": "missing.csv", "image_size": 224},
        "training": {"batch_size": 2},
    }
    with pytest.raises(FileNotFoundError, match="fixed split manifest"):
        t25._build_loaders(args, cfg)


def test_cli_requires_explicit_stage():
    with pytest.raises(SystemExit):
        t25.parse_args(["--data-dir", "dataset"])
