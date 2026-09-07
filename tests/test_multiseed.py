"""Tests for src/modules/comparison.py::summarize_multiseed (S14, optional
multi-seed repeat). Fast synthetic-data tests -- no dependency on the
actual (not-yet-run, GPU-hour) seed-123/2026 repeats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modules import summarize_multiseed


def _comparison_df(rows):
    """rows: list of (arm, test_macro_f1, EIL_post)."""
    return pd.DataFrame([{"arm": arm, "test_macro_f1": f1, "EIL_post": eil} for arm, f1, eil in rows])


def test_summarize_multiseed_computes_mean_and_std():
    dfs = {
        42: _comparison_df([("A0_vanilla", 0.90, 0.20), ("A2_full", 0.895, 0.30)]),
        123: _comparison_df([("A0_vanilla", 0.91, 0.21), ("A2_full", 0.90, 0.31)]),
        2026: _comparison_df([("A0_vanilla", 0.895, 0.19), ("A2_full", 0.89, 0.29)]),
    }
    summary = summarize_multiseed(dfs)

    a2_f1 = summary["A2_full"]["test_macro_f1"]
    expected_values = [0.895, 0.90, 0.89]
    assert a2_f1["values"] == expected_values
    assert a2_f1["mean"] == pytest.approx(np.mean(expected_values))
    assert a2_f1["std"] == pytest.approx(np.std(expected_values, ddof=1))
    assert a2_f1["n_seeds"] == 3
    assert a2_f1["seeds"] == [42, 123, 2026]


def test_summarize_multiseed_headline_delta_reads_naturally():
    """The WBS section S14 motivating example: 'A2 beat A0 by 0.06 EIL' ->
    '0.06 +/- 0.01'. Sanity-check the arithmetic reads the way a reviewer
    would actually use it (mean delta with a real spread, not 0)."""
    dfs = {
        42: _comparison_df([("A0_vanilla", 0.90, 0.20), ("A2_full", 0.90, 0.26)]),
        123: _comparison_df([("A0_vanilla", 0.90, 0.21), ("A2_full", 0.90, 0.27)]),
        2026: _comparison_df([("A0_vanilla", 0.90, 0.19), ("A2_full", 0.90, 0.25)]),
    }
    summary = summarize_multiseed(dfs)
    a0_eil = summary["A0_vanilla"]["EIL_post"]["mean"]
    a2_eil = summary["A2_full"]["EIL_post"]["mean"]
    assert (a2_eil - a0_eil) == pytest.approx(0.06, abs=1e-9)
    assert summary["A2_full"]["EIL_post"]["std"] > 0  # a real spread, not a fabricated one


def test_summarize_multiseed_rejects_single_seed():
    """WBS section S14: 'do not imply repeats you didn't run' -- a mean
    +/- std computed from one seed would silently claim a repeat that
    never happened. Must raise, not return std=0 that looks like a real
    (if tiny) measured spread."""
    dfs = {42: _comparison_df([("A0_vanilla", 0.90, 0.20)])}
    with pytest.raises(ValueError):
        summarize_multiseed(dfs)


def test_summarize_multiseed_rejects_zero_seeds():
    with pytest.raises(ValueError):
        summarize_multiseed({})


def test_summarize_multiseed_missing_arm_in_one_seed_raises():
    dfs = {
        42: _comparison_df([("A0_vanilla", 0.90, 0.20), ("A2_full", 0.895, 0.30)]),
        123: _comparison_df([("A0_vanilla", 0.91, 0.21)]),  # A2_full missing for this seed
    }
    with pytest.raises(KeyError):
        summarize_multiseed(dfs)


def test_summarize_multiseed_explicit_arms_and_metrics_subset():
    dfs = {
        42: _comparison_df([("A0_vanilla", 0.90, 0.20), ("A2_full", 0.895, 0.30), ("A1_gate_only", 0.88, 0.25)]),
        123: _comparison_df([("A0_vanilla", 0.91, 0.21), ("A2_full", 0.90, 0.31), ("A1_gate_only", 0.87, 0.24)]),
    }
    summary = summarize_multiseed(dfs, arms=["A0_vanilla"], metrics=["test_macro_f1"])
    assert list(summary.keys()) == ["A0_vanilla"]
    assert list(summary["A0_vanilla"].keys()) == ["test_macro_f1"]
