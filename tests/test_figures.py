"""Tests for src/modules/figures.py: image selection and heat-map rendering
(S12). Synthetic numpy arrays and synthetic per-image DataFrames throughout
-- fast (no model, no GPU, no dataset loading), same category as every
test since S3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modules import plot_attention_grid, plot_heatmap_row, select_heatmap_images

CLASSES = ["COVID", "Normal", "Lung_Opacity", "Viral_Pneumonia"]


# ---------------------------------------------------------------------------
# select_heatmap_images
# ---------------------------------------------------------------------------

def _per_image_df(rows):
    """rows: list of (image_path, true_label, a0_correct, a2_correct, ilar).
    Matches build_per_image_predictions's real schema: EVERY arm's per-image
    df has an "ilar" column, NaN for arms with no attention module (A0) --
    so it's present in both dataframes and gets suffixed to ilar_a0/ilar_a2
    by the merge, same as in production."""
    a0_rows, a2_rows = [], []
    for path, true_label, a0_correct, a2_correct, ilar in rows:
        a0_pred = true_label if a0_correct else _other_label(true_label)
        a2_pred = true_label if a2_correct else _other_label(true_label)
        a0_rows.append({"image_path": path, "true_label": true_label, "pred_label": a0_pred, "ilar": float("nan")})
        a2_rows.append({"image_path": path, "true_label": true_label, "pred_label": a2_pred, "ilar": ilar})
    return pd.DataFrame(a0_rows), pd.DataFrame(a2_rows)


def _other_label(label):
    others = [c for c in CLASSES if c != label]
    return others[0]


def test_select_heatmap_images_finds_a0_wrong_a2_right_case():
    rows = [
        ("covid_1.png", "COVID", True, True, 0.6),
        ("covid_2.png", "COVID", False, True, 0.7),   # A0 wrong, A2 right
        ("covid_3.png", "COVID", True, True, 0.2),    # lowest ilar among correct -> artifact proxy
    ]
    a0_df, a2_df = _per_image_df(rows)
    picks = select_heatmap_images(a0_df, a2_df, ["COVID"], seed=0)

    reasons = set(picks["reason"])
    assert "correctly_classified" in reasons
    assert "a0_wrong_a2_right" in reasons
    assert "shortcut_artifact_proxy_VERIFY_VISUALLY" in reasons

    artifact_row = picks[picks["reason"] == "shortcut_artifact_proxy_VERIFY_VISUALLY"].iloc[0]
    assert artifact_row["image_path"] == "covid_3.png"  # the lowest-ilar correct example

    a0_right_row = picks[picks["reason"] == "a0_wrong_a2_right"].iloc[0]
    assert a0_right_row["image_path"] == "covid_2.png"


def test_select_heatmap_images_falls_back_when_no_a0_wrong_a2_right_case():
    """WBS section S12 explicitly allows this case to be absent for a class."""
    rows = [
        ("covid_1.png", "COVID", True, True, 0.6),
        ("covid_2.png", "COVID", True, True, 0.4),
        # no row where A0 is wrong and A2 is right
    ]
    a0_df, a2_df = _per_image_df(rows)
    picks = select_heatmap_images(a0_df, a2_df, ["COVID"], seed=0)

    reasons = list(picks["reason"])
    assert "a0_wrong_a2_right" not in reasons
    assert "correctly_classified_fallback" in reasons  # a second correct example used instead
    assert len(picks) <= 3


def test_select_heatmap_images_handles_missing_class_gracefully():
    rows = [("covid_1.png", "COVID", True, True, 0.5)]
    a0_df, a2_df = _per_image_df(rows)
    # "Normal" has zero rows -- must not crash, just contribute nothing
    picks = select_heatmap_images(a0_df, a2_df, ["COVID", "Normal"], seed=0)
    assert (picks["class_name"] == "Normal").sum() == 0
    assert (picks["class_name"] == "COVID").sum() >= 1


def test_select_heatmap_images_returns_expected_columns_even_when_empty():
    a0_df, a2_df = _per_image_df([])
    picks = select_heatmap_images(a0_df, a2_df, CLASSES, seed=0)
    assert list(picks.columns) == ["class_name", "reason", "image_path"]
    assert len(picks) == 0


def test_select_heatmap_images_artifact_proxy_falls_back_to_unsuffixed_ilar():
    """If only the A2 dataframe carries an ilar column (no ilar column on
    the A0 side at all, so the merge can't suffix it to ilar_a2), the
    artifact-proxy step must still work via the bare "ilar" column rather
    than silently finding nothing."""
    a0_df = pd.DataFrame([
        {"image_path": "covid_1.png", "true_label": "COVID", "pred_label": "COVID"},
        {"image_path": "covid_2.png", "true_label": "COVID", "pred_label": "COVID"},
    ])
    a2_df = pd.DataFrame([
        {"image_path": "covid_1.png", "true_label": "COVID", "pred_label": "COVID", "ilar": 0.6},
        {"image_path": "covid_2.png", "true_label": "COVID", "pred_label": "COVID", "ilar": 0.1},
    ])
    picks = select_heatmap_images(a0_df, a2_df, ["COVID"], seed=0)
    artifact_rows = picks[picks["reason"] == "shortcut_artifact_proxy_VERIFY_VISUALLY"]
    assert len(artifact_rows) == 1
    assert artifact_rows.iloc[0]["image_path"] == "covid_2.png"  # lower ilar


# ---------------------------------------------------------------------------
# plot_heatmap_row / plot_attention_grid -- rendering only, synthetic arrays
# ---------------------------------------------------------------------------

def test_plot_heatmap_row_writes_file(tmp_path):
    size = 32
    image = np.random.default_rng(0).random((size, size))
    mask = (image > 0.5).astype(float)
    out_path = tmp_path / "heatmaps" / "COVID_1_a0_a2.png"

    plot_heatmap_row(image, mask, mask, mask, mask, "COVID test image", out_path, dpi=50)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_heatmap_row_creates_parent_dirs(tmp_path):
    size = 16
    image = np.zeros((size, size))
    out_path = tmp_path / "a" / "b" / "c" / "row.png"
    plot_heatmap_row(image, image, image, image, image, "t", out_path, dpi=50)
    assert out_path.exists()


def test_plot_attention_grid_writes_file(tmp_path):
    size = 16
    rng = np.random.default_rng(0)
    images = [rng.random((size, size)) for _ in range(8)]
    attentions = [rng.random((size, size)) for _ in range(8)]
    labels = (["COVID"] * 4) + (["Normal"] * 4)
    out_path = tmp_path / "attention_grid.png"

    plot_attention_grid(images, attentions, labels, ["COVID", "Normal"], out_path, n_per_class=4, dpi=50)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_attention_grid_handles_fewer_samples_than_n_per_class(tmp_path):
    """A class with only 1 available sample must not crash -- remaining
    panels in that row are simply left blank."""
    size = 16
    images = [np.zeros((size, size))]
    attentions = [np.zeros((size, size))]
    labels = ["COVID"]
    out_path = tmp_path / "sparse_grid.png"

    plot_attention_grid(images, attentions, labels, ["COVID", "Normal"], out_path, n_per_class=4, dpi=50)
    assert out_path.exists()
