"""Paired statistics for saved anatomical-localization measurements.

These functions describe Grad-CAM energy placement. They do not measure
causal explanation faithfulness or dependence on lung/background pixels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_eil(reference: pd.DataFrame, selected: pd.DataFrame,
               *, expected_n: int = 1000, expected_test_n: int = 3175,
               n_boot: int = 2000,
               bootstrap_seed: int = 42) -> dict:
    """Compare EIL on exactly the same scored images in two arm files."""
    required = {"image_path", "true_label", "eil_post"}
    for name, frame in (("baseline", reference), ("selected", selected)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} prediction file lacks {sorted(missing)}")
        if frame["image_path"].duplicated().any():
            raise ValueError(f"{name} has duplicate image paths")
        if len(frame) != expected_test_n:
            raise ValueError(f"{name} must contain {expected_test_n} test images")
    if set(reference["image_path"]) != set(selected["image_path"]):
        raise ValueError("baseline and selected test image sets differ")
    ref = reference.loc[reference["eil_post"].notna(), list(required)]
    sel = selected.loc[selected["eil_post"].notna(), list(required)]
    if len(ref) != expected_n or len(sel) != expected_n:
        raise ValueError(f"expected {expected_n} scored CAM images per arm")
    if set(ref["image_path"]) != set(sel["image_path"]):
        raise ValueError("baseline and selected CAM subsets differ")
    joined = ref.merge(sel, on="image_path", suffixes=("_baseline", "_selected"),
                       validate="one_to_one", sort=True)
    if not (joined["true_label_baseline"] == joined["true_label_selected"]).all():
        raise ValueError("true labels differ between paired arms")
    diffs = (joined["eil_post_selected"] - joined["eil_post_baseline"]).to_numpy(float)
    if not np.isfinite(diffs).all():
        raise ValueError("paired EIL differences must be finite")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[sample_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(means, (0.025, 0.975))
    return {
        "n_paired": len(diffs),
        "baseline_eil_mean": float(joined["eil_post_baseline"].mean()),
        "selected_eil_mean": float(joined["eil_post_selected"].mean()),
        "delta_eil_mean": float(diffs.mean()),
        "delta_eil_median": float(np.median(diffs)),
        "fraction_images_improved": float(np.mean(diffs > 0)),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "wilcoxon_two_sided_p": float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0,
        "bootstrap_seed": bootstrap_seed,
        "n_boot": n_boot,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm adjusted p-values in original input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional array in [0, 1]")
    order = np.argsort(values)
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    output = np.empty_like(values)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()
