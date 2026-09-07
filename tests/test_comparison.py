"""Tests for src/modules/comparison.py: per-image predictions, the
cross-arm comparison table, and acceptance-criteria checking (S11).

Fast synthetic-data tests throughout (same category as every test since
S3) -- no real dataset, no GPU hours, no dependency on S8-S10's actual
(not-yet-run) checkpoints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.datasets import CXRWithMaskDataset, JointTransform, only_images_folder
from src.modules import (
    build_comparison_table,
    build_model,
    build_per_image_predictions,
    check_acceptance_criteria,
    stratified_cam_subset,
)

CLASSES = ["COVID", "Normal", "Lung_Opacity", "Viral_Pneumonia"]


def _synthetic_image_and_mask(size: int = 64):
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[:, : size // 2] = 220
    arr[:, size // 2 :] = 30
    mask_arr = np.zeros((size, size), dtype=np.uint8)
    mask_arr[:, : size // 2] = 255
    return Image.fromarray(arr).convert("L"), Image.fromarray(mask_arr).convert("L")


@pytest.fixture
def tiny_test_dataset(tmp_path):
    """5 images/class x 4 classes = 20 total -- a small but real
    CXRWithMaskDataset, same pipeline as every other dataset fixture."""
    for cls in CLASSES:
        (tmp_path / cls / "images").mkdir(parents=True)
        (tmp_path / cls / "masks").mkdir(parents=True)
        for i in range(5):
            img, mask = _synthetic_image_and_mask()
            img.save(tmp_path / cls / "images" / f"{cls}-{i}.png")
            mask.save(tmp_path / cls / "masks" / f"{cls}-{i}.png")

    base = ImageFolder(root=str(tmp_path), is_valid_file=only_images_folder)
    return CXRWithMaskDataset(base, np.arange(len(base)), JointTransform(img_size=224, train=False))


# ---------------------------------------------------------------------------
# stratified_cam_subset
# ---------------------------------------------------------------------------

def test_stratified_cam_subset_returns_all_when_n_exceeds_total(tiny_test_dataset):
    subset = stratified_cam_subset(tiny_test_dataset, n=1000, seed=42)
    assert len(subset) == len(tiny_test_dataset)
    assert set(subset.tolist()) == set(range(len(tiny_test_dataset)))


def test_stratified_cam_subset_correct_size_and_reproducible(tiny_test_dataset):
    subset_a = stratified_cam_subset(tiny_test_dataset, n=8, seed=42)
    subset_b = stratified_cam_subset(tiny_test_dataset, n=8, seed=42)
    assert len(subset_a) == 8
    np.testing.assert_array_equal(subset_a, subset_b)  # same seed -> identical subset


def test_stratified_cam_subset_different_seed_can_differ(tiny_test_dataset):
    subset_a = stratified_cam_subset(tiny_test_dataset, n=8, seed=42)
    subset_c = stratified_cam_subset(tiny_test_dataset, n=8, seed=999)
    assert not np.array_equal(subset_a, subset_c)


# ---------------------------------------------------------------------------
# build_per_image_predictions
# ---------------------------------------------------------------------------

def test_per_image_predictions_schema_classification_only(tiny_test_dataset):
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    df = build_per_image_predictions(model, tiny_test_dataset, CLASSES, torch.device("cpu"), cam_subset=None)

    assert len(df) == len(tiny_test_dataset)
    expected_cols = {"image_path", "true_label", "pred_label", "ilar", "eil_post", "eil_pre"}
    expected_cols |= {f"prob_{k}" for k in range(4)}
    assert expected_cols <= set(df.columns)
    assert df["eil_post"].isna().all()  # no cam_subset given -> no Grad-CAM computed at all
    assert df["eil_pre"].isna().all()
    assert not df["ilar"].isna().any()  # use_attention=True -> every row has a real ilar
    assert df["true_label"].isin(CLASSES).all()
    assert df["pred_label"].isin(CLASSES).all()
    probs = df[[f"prob_{k}" for k in range(4)]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)  # softmax rows sum to 1


def test_per_image_predictions_eil_only_for_subset_positions(tiny_test_dataset):
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    subset = np.array([0, 3, 7])  # 3 of the 20 positions
    df = build_per_image_predictions(model, tiny_test_dataset, CLASSES, torch.device("cpu"), cam_subset=subset)

    in_subset = df.index.isin(subset)
    assert df.loc[in_subset, "eil_post"].notna().all()
    assert df.loc[in_subset, "eil_pre"].notna().all()
    assert df.loc[~in_subset, "eil_post"].isna().all()
    assert df.loc[~in_subset, "eil_pre"].isna().all()
    assert df.loc[in_subset, "eil_post"].between(0, 1).all()
    assert df.loc[in_subset, "eil_pre"].between(0, 1).all()


def test_per_image_predictions_a0_style_has_nan_ilar(tiny_test_dataset):
    model = build_model(num_classes=4, use_attention=False, pretrained=False)
    df = build_per_image_predictions(model, tiny_test_dataset, CLASSES, torch.device("cpu"), cam_subset=None)
    assert df["ilar"].isna().all()  # no attention module -> NaN, not 0, for every row


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

def _fake_arm(macro_f1, eil_post_vals, eil_pre_vals, att_dice, gate_mode, lambda_att, ilar_val=0.5):
    per_image_df = pd.DataFrame({
        "eil_post": eil_post_vals,
        "eil_pre": eil_pre_vals,
    })
    return {
        "gate_mode": gate_mode,
        "lambda_att": lambda_att,
        "evaluate_results": {
            "classification_report": {"accuracy": macro_f1 + 0.01, "macro avg": {"f1-score": macro_f1}},
            "roc_auc_macro": 0.95,
            "attention_ilar": ilar_val,
            "attention_dice": att_dice,
        },
        "per_image_df": per_image_df,
    }


def test_build_comparison_table_deltas_relative_to_a0():
    arms = {
        "A0_vanilla": _fake_arm(0.90, [0.20, 0.22], [0.20, 0.22], None, "none", 0.0, ilar_val=None),
        "A2_full": _fake_arm(0.895, [0.30, 0.32], [0.24, 0.26], 0.85, "residual", 0.5),
    }
    df = build_comparison_table(arms)

    a0 = df[df["arm"] == "A0_vanilla"].iloc[0]
    a2 = df[df["arm"] == "A2_full"].iloc[0]

    assert a0["delta_macro_f1"] == pytest.approx(0.0)  # A0 relative to itself
    assert a0["delta_eil_post"] == pytest.approx(0.0)
    assert pd.isna(a0["ILAR"])  # A0 has no attention -- NaN, not 0

    assert a2["delta_macro_f1"] == pytest.approx(0.895 - 0.90)
    assert a2["delta_eil_post"] == pytest.approx(0.31 - 0.21, abs=1e-9)
    assert a2["delta_eil_pre"] == pytest.approx(0.25 - 0.21, abs=1e-9)


def test_build_comparison_table_explicit_reference_arm():
    arms = {
        "A1_gate_only": _fake_arm(0.88, [0.25], [0.21], 0.5, "residual", 0.0),
        "A2_full": _fake_arm(0.895, [0.30], [0.24], 0.85, "residual", 0.5),
    }
    # Neither arm has gate="none" -- must fall back sensibly when an
    # explicit reference_arm is given rather than silently picking arm 0
    # by accident.
    df = build_comparison_table(arms, reference_arm="A1_gate_only")
    a2 = df[df["arm"] == "A2_full"].iloc[0]
    assert a2["delta_macro_f1"] == pytest.approx(0.895 - 0.88)


def test_build_comparison_table_writes_csv(tmp_path):
    arms = {"A0_vanilla": _fake_arm(0.90, [0.2], [0.2], None, "none", 0.0, ilar_val=None)}
    out_csv = tmp_path / "T18_comparison_table.csv"
    build_comparison_table(arms, output_csv=out_csv)
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert "arm" in df.columns and "delta_macro_f1" in df.columns


# ---------------------------------------------------------------------------
# check_acceptance_criteria
# ---------------------------------------------------------------------------

def test_check_acceptance_criteria_all_pass():
    arms = {
        "A0_vanilla": _fake_arm(0.90, [0.20], [0.20], None, "none", 0.0, ilar_val=None),
        "A2_full": _fake_arm(0.895, [0.30], [0.25], 0.85, "residual", 0.5),  # dice 0.85>=0.80, dEIL_post=0.10>=0.05, dEIL_pre=0.05>0, dF1=-0.005>=-0.01
    }
    df = build_comparison_table(arms)
    verdicts = check_acceptance_criteria(df, headline_arm="A2_full")

    assert verdicts["A1_attention_matches_lung_mask"]["pass"] is True
    assert verdicts["A2_evidence_inside_lung_up"]["pass"] is True
    assert verdicts["A3_mechanism_is_real"]["pass"] is True
    assert verdicts["A4_no_accuracy_tax"]["pass"] is True
    for v in verdicts.values():
        assert "value" in v and "target" in v  # every verdict carries the actual number, not just pass/fail


def test_check_acceptance_criteria_reports_failures_honestly():
    """A2/A3/A4 targets are goals, not gates to tune against (WBS section
    1.3) -- a failing verdict must still be reported accurately, not
    silently coerced to pass."""
    arms = {
        "A0_vanilla": _fake_arm(0.90, [0.20], [0.20], None, "none", 0.0, ilar_val=None),
        "A2_full": _fake_arm(0.85, [0.21], [0.19], 0.60, "residual", 0.5),
        # dice 0.60 < 0.80 (fail A1), dEIL_post=0.01 < 0.05 (fail A2),
        # dEIL_pre=-0.01 <= 0 (fail A3), dF1=-0.05 < -0.01 (fail A4)
    }
    df = build_comparison_table(arms)
    verdicts = check_acceptance_criteria(df, headline_arm="A2_full")

    assert verdicts["A1_attention_matches_lung_mask"]["pass"] is False
    assert verdicts["A2_evidence_inside_lung_up"]["pass"] is False
    assert verdicts["A3_mechanism_is_real"]["pass"] is False
    assert verdicts["A4_no_accuracy_tax"]["pass"] is False
    assert verdicts["A4_no_accuracy_tax"]["value"] == pytest.approx(0.85 - 0.90)


def test_check_acceptance_criteria_missing_headline_arm_raises():
    arms = {"A0_vanilla": _fake_arm(0.90, [0.2], [0.2], None, "none", 0.0, ilar_val=None)}
    df = build_comparison_table(arms)
    with pytest.raises(KeyError):
        check_acceptance_criteria(df, headline_arm="A2_full")  # not in arms
