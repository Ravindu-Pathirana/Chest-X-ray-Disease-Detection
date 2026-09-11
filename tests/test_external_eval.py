"""Tests for src/modules/external_eval.py -- the external-test protocol (T32).

All synthetic per-image prediction tables, matching the shape
comparison.py's build_per_image_predictions() produces. No real trained
model or dataset needed -- this tests the label-mapping and metric math,
which is deterministic and checkable by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modules.external_eval import (
    compute_ood_metrics,
    map_binary_collapse,
    map_class_matched,
    run_external_test_protocol,
)

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def _primary_row(image_path, true_label, probs):
    """probs: dict of class_name -> probability, must sum to ~1."""
    row = {"image_path": image_path, "true_label": true_label}
    for i, name in enumerate(CLASS_NAMES):
        row[f"prob_{i}"] = probs.get(name, 0.0)
    return row


def _rsna_row(image_path, true_label, probs):
    return _primary_row(image_path, true_label, probs)


# ---------------------------------------------------------------------------
# map_binary_collapse
# ---------------------------------------------------------------------------


def test_map_binary_collapse_primary_sums_three_classes():
    df = pd.DataFrame(
        [
            _primary_row("a.png", "Normal", {"Normal": 0.7, "COVID": 0.1, "Lung_Opacity": 0.1, "Viral Pneumonia": 0.1}),
            _primary_row("b.png", "COVID", {"Normal": 0.2, "COVID": 0.6, "Lung_Opacity": 0.1, "Viral Pneumonia": 0.1}),
        ]
    )
    result = map_binary_collapse(df, CLASS_NAMES, source="primary")

    assert list(result["true_binary"]) == ["Normal", "Abnormal"]
    assert result["prob_abnormal"].iloc[0] == pytest.approx(0.3)  # 0.1+0.1+0.1
    assert result["prob_abnormal"].iloc[1] == pytest.approx(0.8)  # 0.6+0.1+0.1


def test_map_binary_collapse_rsna_relabels_pneumonia_to_abnormal():
    df = pd.DataFrame(
        [
            _rsna_row("x.dcm", "Normal", {"Normal": 0.9, "COVID": 0.02, "Lung_Opacity": 0.03, "Viral Pneumonia": 0.05}),
            _rsna_row("y.dcm", "Pneumonia", {"Normal": 0.1, "COVID": 0.05, "Lung_Opacity": 0.15, "Viral Pneumonia": 0.7}),
        ]
    )
    result = map_binary_collapse(df, CLASS_NAMES, source="rsna")

    assert list(result["true_binary"]) == ["Normal", "Abnormal"]
    assert result["prob_abnormal"].iloc[1] == pytest.approx(0.9)  # 0.05+0.15+0.7


def test_map_binary_collapse_raises_on_unmapped_label():
    df = pd.DataFrame([_primary_row("a.png", "Something_Else", {"Normal": 1.0})])
    with pytest.raises(ValueError, match="Unmapped label"):
        map_binary_collapse(df, CLASS_NAMES, source="primary")


def test_map_binary_collapse_invalid_source_raises():
    df = pd.DataFrame([_primary_row("a.png", "Normal", {"Normal": 1.0})])
    with pytest.raises(ValueError, match="source must be"):
        map_binary_collapse(df, CLASS_NAMES, source="nonsense")


# ---------------------------------------------------------------------------
# map_class_matched
# ---------------------------------------------------------------------------


def test_map_class_matched_primary_drops_unmatched_classes():
    df = pd.DataFrame(
        [
            _primary_row("a.png", "Normal", {"Normal": 0.7, "Viral Pneumonia": 0.3}),
            _primary_row("b.png", "COVID", {"COVID": 0.9, "Normal": 0.1}),  # should be dropped
            _primary_row("c.png", "Viral Pneumonia", {"Normal": 0.2, "Viral Pneumonia": 0.6, "COVID": 0.2}),
        ]
    )
    result = map_class_matched(df, CLASS_NAMES, source="primary")

    assert len(result) == 2  # the COVID row is dropped
    assert set(result["true_matched"]) == {"Normal", "Viral Pneumonia"}


def test_map_class_matched_primary_renormalizes_over_kept_classes():
    df = pd.DataFrame(
        [_primary_row("c.png", "Viral Pneumonia", {"Normal": 0.2, "Viral Pneumonia": 0.6, "COVID": 0.2})]
    )
    result = map_class_matched(df, CLASS_NAMES, source="primary")

    # kept classes are Normal (0.2) + Viral Pneumonia (0.6) = 0.8 total
    # renormalized Viral Pneumonia prob = 0.6 / 0.8 = 0.75
    assert result["prob_pneumonia_equivalent"].iloc[0] == pytest.approx(0.75)


def test_map_class_matched_rsna_renames_and_renormalizes():
    df = pd.DataFrame(
        [_rsna_row("y.dcm", "Pneumonia", {"Normal": 0.1, "COVID": 0.05, "Lung_Opacity": 0.15, "Viral Pneumonia": 0.7})]
    )
    result = map_class_matched(df, CLASS_NAMES, source="rsna")

    assert result["true_matched"].iloc[0] == "Viral Pneumonia"
    # kept classes: Normal (0.1) + Viral Pneumonia (0.7) = 0.8
    assert result["prob_pneumonia_equivalent"].iloc[0] == pytest.approx(0.7 / 0.8)


def test_map_class_matched_zero_kept_probability_raises():
    df = pd.DataFrame([_primary_row("a.png", "Normal", {"COVID": 1.0})])  # Normal+ViralPneumonia both 0
    with pytest.raises(ValueError, match="zero total probability"):
        map_class_matched(df, CLASS_NAMES, source="primary")


# ---------------------------------------------------------------------------
# compute_ood_metrics
# ---------------------------------------------------------------------------


def test_compute_ood_metrics_perfect_classifier_gives_auc_one():
    internal_df = pd.DataFrame(
        {"true_binary": ["Normal", "Abnormal", "Normal", "Abnormal"], "prob_abnormal": [0.1, 0.9, 0.2, 0.8]}
    )
    external_df = pd.DataFrame(
        {"true_binary": ["Normal", "Abnormal", "Normal", "Abnormal"], "prob_abnormal": [0.1, 0.9, 0.2, 0.8]}
    )
    metrics = compute_ood_metrics(
        internal_df, external_df, true_column="true_binary", prob_column="prob_abnormal", positive_label="Abnormal"
    )

    assert metrics["internal_auc"] == pytest.approx(1.0)
    assert metrics["external_auc"] == pytest.approx(1.0)
    assert metrics["auc_drop"] == pytest.approx(0.0)
    assert metrics["internal_accuracy"] == pytest.approx(1.0)
    assert metrics["n_internal"] == 4
    assert metrics["n_external"] == 4


def test_compute_ood_metrics_detects_a_real_drop():
    # Internal: model separates classes perfectly.
    internal_df = pd.DataFrame(
        {"true_binary": ["Normal"] * 5 + ["Abnormal"] * 5, "prob_abnormal": [0.05] * 5 + [0.95] * 5}
    )
    # External: model is near-random (the shortcut-learning failure mode this whole project targets).
    external_df = pd.DataFrame(
        {"true_binary": ["Normal"] * 5 + ["Abnormal"] * 5, "prob_abnormal": [0.5, 0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.6, 0.4, 0.5]}
    )
    metrics = compute_ood_metrics(
        internal_df, external_df, true_column="true_binary", prob_column="prob_abnormal", positive_label="Abnormal"
    )

    assert metrics["internal_auc"] == pytest.approx(1.0)
    assert metrics["external_auc"] < 0.7  # meaningfully worse, not a strict random-AUC=0.5 assertion (small-n noise)
    assert metrics["auc_drop"] > 0.3  # a real, detectable robustness failure


# ---------------------------------------------------------------------------
# run_external_test_protocol (end-to-end)
# ---------------------------------------------------------------------------


def _synthetic_internal_df(n_per_class=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for class_name in CLASS_NAMES:
        for i in range(n_per_class):
            # A confident, mostly-correct model: true class gets high prob, rest split the remainder.
            probs = {c: 0.05 for c in CLASS_NAMES}
            probs[class_name] = 0.85
            # renormalize to sum to 1 exactly
            total = sum(probs.values())
            probs = {c: v / total for c, v in probs.items()}
            rows.append(_primary_row(f"{class_name}_{i}.png", class_name, probs))
    return pd.DataFrame(rows)


def _synthetic_external_df(n_per_class=10, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for rsna_label in ["Normal", "Pneumonia"]:
        for i in range(n_per_class):
            # A shakier, less-confident model on external data.
            probs = {c: 0.2 for c in CLASS_NAMES}
            mapped_class = "Normal" if rsna_label == "Normal" else "Viral Pneumonia"
            probs[mapped_class] = 0.4
            total = sum(probs.values())
            probs = {c: v / total for c, v in probs.items()}
            rows.append(_rsna_row(f"{rsna_label}_{i}.dcm", rsna_label, probs))
    return pd.DataFrame(rows)


def test_run_external_test_protocol_binary_collapse_end_to_end():
    internal_df = _synthetic_internal_df()
    external_df = _synthetic_external_df()

    result = run_external_test_protocol(internal_df, external_df, CLASS_NAMES, mapping="binary_collapse")

    assert result["mapping"] == "binary_collapse"
    assert result["n_internal"] == 40  # 4 classes x 10
    assert result["n_external"] == 20  # 2 classes x 10
    assert 0.0 <= result["internal_auc"] <= 1.0
    assert 0.0 <= result["external_auc"] <= 1.0


def test_run_external_test_protocol_class_matched_end_to_end():
    internal_df = _synthetic_internal_df()
    external_df = _synthetic_external_df()

    result = run_external_test_protocol(internal_df, external_df, CLASS_NAMES, mapping="class_matched")

    assert result["mapping"] == "class_matched"
    assert result["n_internal"] == 20  # only Normal + Viral Pneumonia rows kept (2 classes x 10)
    assert result["n_external"] == 20  # RSNA already only has 2 classes, none dropped


def test_run_external_test_protocol_invalid_mapping_raises():
    internal_df = _synthetic_internal_df(n_per_class=2)
    external_df = _synthetic_external_df(n_per_class=2)
    with pytest.raises(ValueError, match="mapping must be"):
        run_external_test_protocol(internal_df, external_df, CLASS_NAMES, mapping="nonsense")
