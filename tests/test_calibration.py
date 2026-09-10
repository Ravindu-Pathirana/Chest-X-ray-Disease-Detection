"""Tests for src/modules/calibration.py (T27).

Hand-computed cases for ECE/Brier/reliability-diagram binning, a
synthetic-but-realistic overconfidence case for temperature scaling
(fit on "validation", confirm it actually improves calibration on a held-
out "test" set -- not just that it runs), and one end-to-end
calibration_report() test using an Identity model so the harness itself
(loader iteration, file writes, JSON schema) is exercised without needing
a real trained checkpoint.
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn
import pytest

from src.modules import (
    TemperatureScaler,
    brier_score,
    calibration_report,
    collect_logits,
    expected_calibration_error,
    fit_temperature,
    reliability_diagram_data,
)


class _Identity(nn.Module):
    """Returns its input unchanged -- lets a test hand pre-built logits
    straight through collect_logits() as if they came from a real model,
    the same trick test_efficiency_check.py plays with synthetic CSVs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration_is_zero():
    probs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0, abs=1e-6)


def test_ece_hand_computed_two_bins():
    # confidences [0.9, 0.9, 0.6], predictions all class 0, labels [0, 1, 0]
    # -> accuracies [correct, wrong, correct] = [1, 0, 1]
    # bin (0.8,0.9]: 2 samples, acc=0.5, conf=0.9 -> weight 2/3, |0.9-0.5|=0.4
    # bin (0.5,0.6]: 1 sample,  acc=1.0, conf=0.6 -> weight 1/3, |0.6-1.0|=0.4
    # ECE = 2/3*0.4 + 1/3*0.4 = 0.4 exactly
    probs = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4]])
    labels = torch.tensor([0, 1, 0])
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-6)


def test_ece_empty_bins_dont_contribute():
    # All confidences land in one bin; every other bin is empty and must
    # contribute exactly 0, not NaN or an error.
    probs = torch.tensor([[0.55, 0.45]] * 4)
    labels = torch.tensor([0, 0, 1, 1])
    ece = expected_calibration_error(probs, labels, n_bins=20)
    # bin (0.5,0.55]: conf=0.55, acc=0.5 -> ECE = 1.0 * |0.55-0.5| = 0.05
    assert ece == pytest.approx(0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

def test_brier_score_perfect_prediction_is_zero():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    assert brier_score(probs, labels, num_classes=2) == pytest.approx(0.0, abs=1e-6)


def test_brier_score_hand_computed_uniform_binary():
    # p=[0.5,0.5], y=[1,0] -> (0.5-1)^2 + (0.5-0)^2 = 0.25+0.25 = 0.5
    probs = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([0])
    assert brier_score(probs, labels, num_classes=2) == pytest.approx(0.5, abs=1e-6)


def test_brier_score_hand_computed_three_class():
    # p=[0.7,0.2,0.1], y=class1 -> onehot=[0,1,0]
    # diffs: 0.7, -0.8, 0.1 -> squares: 0.49, 0.64, 0.01 -> sum 1.14
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    labels = torch.tensor([1])
    assert brier_score(probs, labels, num_classes=3) == pytest.approx(1.14, abs=1e-6)


def test_brier_score_averages_over_samples():
    probs = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    labels = torch.tensor([0, 0])
    # sample 0: brier=0, sample 1: brier=0.5 -> mean = 0.25
    assert brier_score(probs, labels, num_classes=2) == pytest.approx(0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# reliability_diagram_data
# ---------------------------------------------------------------------------

def test_reliability_diagram_data_matches_ece_binning():
    probs = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4]])
    labels = torch.tensor([0, 1, 0])
    rows = reliability_diagram_data(probs, labels, n_bins=10)
    assert len(rows) == 10

    bin_08_09 = next(r for r in rows if r["bin_lo"] == pytest.approx(0.8))
    assert bin_08_09["count"] == 2
    assert bin_08_09["confidence"] == pytest.approx(0.9, abs=1e-6)
    assert bin_08_09["accuracy"] == pytest.approx(0.5, abs=1e-6)

    bin_05_06 = next(r for r in rows if r["bin_lo"] == pytest.approx(0.5))
    assert bin_05_06["count"] == 1
    assert bin_05_06["accuracy"] == pytest.approx(1.0, abs=1e-6)


def test_reliability_diagram_data_empty_bin_is_none_not_zero():
    probs = torch.tensor([[0.55, 0.45]])
    labels = torch.tensor([0])
    rows = reliability_diagram_data(probs, labels, n_bins=20)
    empty_bin = next(r for r in rows if r["bin_lo"] == pytest.approx(0.0))
    assert empty_bin["count"] == 0
    assert empty_bin["confidence"] is None
    assert empty_bin["accuracy"] is None


# ---------------------------------------------------------------------------
# TemperatureScaler / fit_temperature
# ---------------------------------------------------------------------------

def test_temperature_scaler_init_is_identity():
    scaler = TemperatureScaler()
    assert scaler.temperature.item() == pytest.approx(1.0, abs=1e-6)
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    assert torch.allclose(scaler(logits), logits)


def test_fit_temperature_softens_an_overconfident_model():
    """8/10 confidently-correct + 2/10 confidently-WRONG -- the textbook
    overconfidence pattern temperature scaling exists to fix. Fitting T on
    this should push T above 1 (softening), and applying that T to a
    held-out set with the same miscalibration pattern should lower ECE,
    not just "run without crashing"."""
    torch.manual_seed(0)
    magnitude = 6.0

    def _make(n_wrong: int, n_total: int = 20):
        logits, labels = [], []
        for i in range(n_total):
            true_class = i % 2
            wrong = i < n_wrong
            predicted_class = 1 - true_class if wrong else true_class
            row = [-magnitude, -magnitude]
            row[predicted_class] = magnitude
            logits.append(row)
            labels.append(true_class)
        return torch.tensor(logits), torch.tensor(labels)

    val_logits, val_labels = _make(n_wrong=4)
    test_logits, test_labels = _make(n_wrong=4)

    temperature = fit_temperature(val_logits, val_labels)
    assert temperature > 1.5  # meaningfully softened, not left near 1.0

    raw_probs = torch.softmax(test_logits, dim=1)
    scaled_probs = torch.softmax(test_logits / temperature, dim=1)
    raw_ece = expected_calibration_error(raw_probs, test_labels)
    scaled_ece = expected_calibration_error(scaled_probs, test_labels)
    assert scaled_ece < raw_ece


# ---------------------------------------------------------------------------
# calibration_report (end-to-end harness)
# ---------------------------------------------------------------------------

def test_calibration_report_writes_expected_files_and_schema(tmp_path):
    torch.manual_seed(1)
    val_logits = torch.randn(16, 4) * 3
    val_labels = torch.randint(0, 4, (16,))
    test_logits = torch.randn(12, 4) * 3
    test_labels = torch.randint(0, 4, (12,))

    val_loader = [(val_logits, val_labels)]
    test_loader = [(test_logits, test_labels)]

    summary = calibration_report(
        _Identity(), val_loader, test_loader,
        class_names=["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
        device=torch.device("cpu"), output_dir=tmp_path, model_name="test_model",
    )

    assert summary["test_n"] == 12
    assert summary["val_n_used_for_temperature_fit"] == 16
    assert "ece" in summary["before_scaling"] and "brier" in summary["before_scaling"]
    assert "ece" in summary["after_scaling"] and "brier" in summary["after_scaling"]
    assert summary["temperature"] > 0

    assert (tmp_path / "calibration_summary.json").exists()
    assert (tmp_path / "reliability_before.png").exists()
    assert (tmp_path / "reliability_after.png").exists()

    on_disk = json.loads((tmp_path / "calibration_summary.json").read_text())
    assert on_disk == summary


def test_calibration_report_make_plots_false_skips_figures(tmp_path):
    val_loader = [(torch.randn(8, 2), torch.randint(0, 2, (8,)))]
    test_loader = [(torch.randn(8, 2), torch.randint(0, 2, (8,)))]
    calibration_report(
        _Identity(), val_loader, test_loader, class_names=["A", "B"],
        device=torch.device("cpu"), output_dir=tmp_path, make_plots=False,
    )
    assert (tmp_path / "calibration_summary.json").exists()
    assert not (tmp_path / "reliability_before.png").exists()


def test_collect_logits_rejects_tuple_returning_model():
    class _TupleModel(nn.Module):
        def forward(self, x):
            return x, None, None

    loader = [(torch.randn(4, 3), torch.randint(0, 3, (4,)))]
    with pytest.raises(TypeError):
        collect_logits(_TupleModel(), loader, torch.device("cpu"))


def test_collect_logits_tolerates_mask_aware_triple_batches():
    """Mask-aware loaders (CXRWithMaskDataset) yield (image, label, mask)
    triples -- collect_logits must ignore the third element, not crash."""
    images = torch.randn(4, 3)
    labels = torch.randint(0, 3, (4,))
    masks = torch.zeros(4, 1)
    loader = [(images, labels, masks)]
    logits, out_labels = collect_logits(_Identity(), loader, torch.device("cpu"))
    assert torch.equal(logits, images)
    assert torch.equal(out_labels, labels)
