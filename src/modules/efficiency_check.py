"""T18 section S13: acceptance criterion A6 (module is cheap).

Deliberately does NOT reimplement or duplicate benchmark_model/results_to_table
from kusal-notebooks/efficiency.py (Member 5's T34/T35 harness, shared
across every architecture) -- that stays the single source of truth for
how efficiency is measured. This file only reads its output and applies
T18's specific pass/fail threshold on top.

Named efficiency_check.py, not efficiency.py, so it can never collide on
`import efficiency` with the top-level module the WBS's own S13 snippet
imports directly from kusal-notebooks/ (see the notebook's S13 cell).
"""
from __future__ import annotations

from typing import Any, Dict, Union

import pandas as pd


def check_module_efficiency(
    efficiency_results: Union[str, "pd.DataFrame"],
    without_module_label: str = "without_module",
    with_module_label: str = "with_module",
    max_params_pct: float = 3.0,
    max_gflops_pct: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """WBS section 1.3, criterion A6 -- computed here in S13.

    `efficiency_results` is either a path to a CSV or an already-loaded
    DataFrame in kusal-notebooks/efficiency.py::benchmark_model's own
    output schema (model, module, input_shape, params_total,
    params_trainable, flops, gflops, ...) -- i.e. what benchmark_model's
    own `output_csv` parameter writes, NOT results_to_table's narrower
    convenience schema (which drops params/flops precision the delta
    calculation here needs and doesn't match efficiency_results.csv's
    existing columns -- see the notebook's S13 cell for why
    benchmark_model's own output_csv is used instead of the WBS's literal
    results_to_table snippet).

    If `module` has more than one row per label (e.g. re-run across
    sessions, appended each time), the LAST matching row is used --
    benchmark_model appends, it never overwrites.
    """
    df = efficiency_results if isinstance(efficiency_results, pd.DataFrame) else pd.read_csv(efficiency_results)

    without_matches = df[df["module"] == without_module_label]
    with_matches = df[df["module"] == with_module_label]
    if without_matches.empty or with_matches.empty:
        raise ValueError(
            f"efficiency_results must contain rows for both '{without_module_label}' and "
            f"'{with_module_label}' in the 'module' column; found: {sorted(df['module'].unique())}"
        )
    without_row = without_matches.iloc[-1]
    with_row = with_matches.iloc[-1]

    params_delta_abs = int(with_row["params_total"] - without_row["params_total"])
    gflops_delta_abs = float(with_row["gflops"] - without_row["gflops"])
    params_delta_pct = 100.0 * params_delta_abs / float(without_row["params_total"])
    gflops_delta_pct = 100.0 * gflops_delta_abs / float(without_row["gflops"])

    return {
        "A6_module_is_cheap": {
            "target": f"params delta <= {max_params_pct}%, GFLOPs delta <= {max_gflops_pct}%",
            "params_delta_absolute": params_delta_abs,
            "params_delta_pct": params_delta_pct,
            "gflops_delta_absolute": gflops_delta_abs,
            "gflops_delta_pct": gflops_delta_pct,
            "pass": bool(params_delta_pct <= max_params_pct and gflops_delta_pct <= max_gflops_pct),
        }
    }
