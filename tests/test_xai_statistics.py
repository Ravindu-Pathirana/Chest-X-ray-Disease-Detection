"""Checks that localization statistics preserve image pairing."""
from __future__ import annotations

import pandas as pd
import pytest

from src.modules.xai_statistics import holm_adjust, paired_eil


def _frame(values, *, labels=(0, 1, 0, 1)):
    return pd.DataFrame({
        "image_path": ["a", "b", "c", "d"],
        "true_label": list(labels),
        "eil_post": values,
    })


def test_paired_eil_aligns_paths_and_keeps_unscored_rows_missing():
    baseline = _frame([0.2, None, 0.3, None])
    selected = _frame([0.4, None, 0.4, None]).iloc[::-1].reset_index(drop=True)
    result = paired_eil(baseline, selected, expected_n=2, expected_test_n=4, n_boot=200)
    assert result["n_paired"] == 2
    assert result["delta_eil_mean"] == pytest.approx(0.15)
    assert result["fraction_images_improved"] == 1.0
    assert result["ci_95_low"] <= result["delta_eil_mean"] <= result["ci_95_high"]


def test_paired_eil_rejects_different_cam_subset_and_labels():
    baseline = _frame([0.2, None, 0.3, None])
    with pytest.raises(ValueError, match="CAM subsets differ"):
        paired_eil(baseline, _frame([0.4, 0.4, None, None]),
                   expected_n=2, expected_test_n=4)
    with pytest.raises(ValueError, match="true labels differ"):
        paired_eil(baseline, _frame([0.4, None, 0.4, None], labels=(1, 1, 0, 1)),
                   expected_n=2, expected_test_n=4)


def test_holm_adjust_preserves_original_order():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(ValueError):
        holm_adjust([0.1, 1.1])
