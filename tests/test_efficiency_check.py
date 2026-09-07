"""Tests for src/modules/efficiency_check.py (S13, acceptance criterion A6).

Fast, synthetic-CSV tests, plus one test that replays the real measured
numbers from an actual local run of kusal-notebooks/efficiency.py's
benchmark_model against build_model(use_attention=False/True) --
confirming this function's math against real data, not just synthetic
rows (architecture-level profiling of a freshly-initialized model needs
no training, no dataset, and no GPU -- it was run for real while writing
this task, see the S13 commit message).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.modules import check_module_efficiency


def _df(without_params, with_params, without_gflops, with_gflops, without_module="without_module", with_module="with_module"):
    return pd.DataFrame([
        {"model": "DenseNet121", "module": without_module, "params_total": without_params, "gflops": without_gflops},
        {"model": "DenseNet121", "module": with_module, "params_total": with_params, "gflops": with_gflops},
    ])


def test_check_module_efficiency_pass_case():
    # +1.89% params, +0.22% gflops -- both comfortably under the 3%/2% targets
    df = _df(6_957_956, 7_089_285, 2.864527, 2.870956)
    verdicts = check_module_efficiency(df)
    v = verdicts["A6_module_is_cheap"]
    assert v["pass"] is True
    assert v["params_delta_absolute"] == 131329
    assert v["params_delta_pct"] == pytest.approx(1.8874, abs=1e-3)
    assert v["gflops_delta_pct"] == pytest.approx(0.2245, abs=1e-3)


def test_check_module_efficiency_fail_case():
    df = _df(1_000_000, 1_050_000, 1.0, 1.03)  # +5% params, +3% gflops -- both over target
    verdicts = check_module_efficiency(df)
    v = verdicts["A6_module_is_cheap"]
    assert v["pass"] is False
    assert v["params_delta_pct"] == pytest.approx(5.0)
    assert v["gflops_delta_pct"] == pytest.approx(3.0)


def test_check_module_efficiency_reads_from_csv_path(tmp_path):
    df = _df(1_000_000, 1_010_000, 1.0, 1.01)
    csv_path = tmp_path / "T18_efficiency.csv"
    df.to_csv(csv_path, index=False)
    verdicts = check_module_efficiency(str(csv_path))
    assert verdicts["A6_module_is_cheap"]["pass"] is True


def test_check_module_efficiency_missing_module_row_raises():
    df = pd.DataFrame([{"model": "DenseNet121", "module": "without_module", "params_total": 100, "gflops": 1.0}])
    with pytest.raises(ValueError):
        check_module_efficiency(df)  # no "with_module" row


def test_check_module_efficiency_uses_last_row_when_appended_multiple_times():
    """benchmark_model's own output_csv appends, it never overwrites -- if a
    session reran S13, the file could have stale earlier rows before the
    real ones. The most recent measurement must win, not the first."""
    df = pd.DataFrame([
        {"model": "DenseNet121", "module": "without_module", "params_total": 999, "gflops": 9.9},  # stale
        {"model": "DenseNet121", "module": "with_module", "params_total": 999, "gflops": 9.9},       # stale
        {"model": "DenseNet121", "module": "without_module", "params_total": 6_957_956, "gflops": 2.864527},
        {"model": "DenseNet121", "module": "with_module", "params_total": 7_089_285, "gflops": 2.870956},
    ])
    verdicts = check_module_efficiency(df)
    assert verdicts["A6_module_is_cheap"]["params_delta_absolute"] == 131329


def test_check_module_efficiency_matches_real_measured_run():
    """Replays the actual numbers from a real local run of
    benchmark_model(LogitsOnly(build_model(use_attention=False/True,
    pretrained=False)), ...) -- params_total 6957956 -> 7089285,
    gflops 2.864526848 -> 2.870955648. Confirms both the exact params
    delta (131329, matching WBS section 3.3's analytic estimate exactly)
    and that A6 passes on the real numbers, not just constructed ones."""
    df = _df(6_957_956, 7_089_285, 2.864526848, 2.870955648)
    verdicts = check_module_efficiency(df)
    v = verdicts["A6_module_is_cheap"]
    assert v["params_delta_absolute"] == 131329
    assert v["gflops_delta_absolute"] == pytest.approx(0.006429, abs=1e-6)
    assert v["pass"] is True
