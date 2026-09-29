"""Regression checks for the offline three-CNN closeout package."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_cnn_closeout import expected_calibration_error, multiclass_brier


ROOT = Path(__file__).resolve().parents[1]
CLOSEOUT = ROOT / "artifacts" / "cnn_closeout"


def test_calibration_helpers_on_perfect_predictions():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)
    assert multiclass_brier(probs, labels) == pytest.approx(0.0)


def test_closeout_summary_contains_expected_three_seed_results():
    summary = pd.read_csv(CLOSEOUT / "cnn_multiseed_summary.csv").set_index("backbone")
    assert set(summary.index) == {"DenseNet121", "ResNet50", "EfficientNet-B0"}

    # Regression values reconstructed directly from the committed seed tables.
    assert summary.loc["DenseNet121", "delta_eil_mean"] == pytest.approx(0.0704550411, abs=1e-9)
    assert summary.loc["DenseNet121", "delta_eil_sample_sd"] == pytest.approx(0.0059793534, abs=1e-9)
    assert summary.loc["ResNet50", "delta_eil_mean"] == pytest.approx(0.2989704793, abs=1e-9)
    assert summary.loc["EfficientNet-B0", "delta_eil_mean"] == pytest.approx(0.1133025821, abs=1e-9)


def test_closeout_integrity_audit_passes():
    report = json.loads((CLOSEOUT / "cnn_data_quality_report.json").read_text(encoding="utf-8"))
    assert report["all_files_have_3175_rows"] is True
    assert report["all_files_have_unique_paths"] is True
    assert report["all_pairs_have_identical_paths"] is True
    assert report["all_probability_rows_valid"] is True
    assert len(report["files"]) == 18  # 3 backbones x 3 seeds x baseline/selected


def test_closeout_master_has_all_backbone_seed_arm_combinations():
    master = pd.read_csv(CLOSEOUT / "cnn_master_results.csv")
    assert len(master) == 18
    assert set(master["seed"]) == {42, 123, 2026}
    assert master.groupby(["backbone", "seed"])["role"].nunique().eq(2).all()
    assert master["n_test"].eq(3175).all()
    assert master["n_eil"].eq(1000).all()


def test_efficiency_closeout_has_one_pair_per_backbone():
    efficiency = pd.read_csv(CLOSEOUT / "cnn_efficiency_summary.csv")
    assert len(efficiency) == 6
    assert efficiency.groupby("backbone")["role"].nunique().eq(2).all()
