"""Aggregate six completed checkpoint-only CNN inference runs for one seed.

Requires the directory layout written by run_cnn_closeout_inference.py.
Pairs images by path and intervention, then bootstraps paired mean differences.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cnn_closeout_inference import SPECS


def paired_result(reference: pd.DataFrame, selected: pd.DataFrame,
                  keys: list[str], value: str, expected_n: int = 3175) -> dict:
    for name, frame in (("baseline", reference), ("selected", selected)):
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate {name} rows for {keys}")
        if not frame[value].notna().all():
            raise ValueError(f"missing {name} values for {value}")
    merged = reference.merge(selected, on=keys, suffixes=("_baseline", "_selected"),
                             validate="one_to_one")
    if len(merged) != expected_n or len(reference) != expected_n or len(selected) != expected_n:
        raise ValueError(f"paired {value} has {len(merged)} shared rows; expected {expected_n}")
    if "true_label_baseline" in merged and not (
        merged["true_label_baseline"] == merged["true_label_selected"]
    ).all():
        raise ValueError("paired image labels differ between arms")
    diff = (merged[f"{value}_selected"] - merged[f"{value}_baseline"]).to_numpy(float)
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(2000)])
    return {
        "n_images": len(diff),
        "selected_minus_baseline_mean": float(diff.mean()),
        "paired_bootstrap_95ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "wilcoxon_two_sided_p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
    }


def read_run(root: Path, backbone: str, arm: str, seed: int) -> dict:
    folder = root / backbone / arm / f"seed_{seed}"
    required = ("run_manifest.json", "counterfactual_per_image.csv",
                "occlusion_per_image.csv", "calibration/calibration_summary.json",
                "efficiency.csv")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {missing}")
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    if (manifest["backbone"], manifest["arm"], manifest["seed"]) != (backbone, arm, seed):
        raise ValueError(f"run manifest identity mismatch in {folder}")
    calibration = json.loads((folder / "calibration/calibration_summary.json").read_text(encoding="utf-8"))
    if calibration["test_n"] != 3175 or calibration["val_n_used_for_temperature_fit"] != 3175:
        raise ValueError(f"calibration split counts differ from fixed manifest in {folder}")
    return {
        "manifest": manifest,
        "counterfactual": pd.read_csv(folder / "counterfactual_per_image.csv"),
        "occlusion": pd.read_csv(folder / "occlusion_per_image.csv"),
        "calibration": calibration,
        "efficiency": pd.read_csv(folder / "efficiency.csv"),
    }


def summarize(root: Path, seed: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    paired = {}
    for backbone, spec in SPECS.items():
        baseline_arm, selected_arm = spec["arms"]
        baseline = read_run(root, backbone, baseline_arm, seed)
        selected = read_run(root, backbone, selected_arm, seed)
        if baseline["manifest"]["split_manifest_sha256"] != selected["manifest"]["split_manifest_sha256"]:
            raise ValueError(f"{backbone} arms use different split manifests")
        paired[backbone] = {}
        for key in ("counterfactual", "occlusion"):
            ref = baseline[key]
            sel = selected[key]
            grouping = ["mode"] if key == "counterfactual" else ["region", "mode"]
            if set(map(tuple, ref[grouping].drop_duplicates().to_numpy())) != set(
                map(tuple, sel[grouping].drop_duplicates().to_numpy())
            ):
                raise ValueError(f"{backbone} {key} modes differ between arms")
            paired[backbone][key] = {}
            for group_values, ref_group in ref.groupby(grouping, sort=False):
                values = group_values if isinstance(group_values, tuple) else (group_values,)
                sel_group = sel
                for column, value in zip(grouping, values):
                    sel_group = sel_group[sel_group[column] == value]
                label = "/".join(str(value) for value in values)
                columns = (["stability", "prediction_flipped"] if key == "counterfactual"
                           else ["probability_drop", "prediction_flipped"])
                paired[backbone][key][label] = {
                    metric: paired_result(ref_group, sel_group, ["image_path"], metric)
                    for metric in columns
                }
        for arm, role, result in ((baseline_arm, "baseline", baseline),
                                  (selected_arm, "selected", selected)):
            cal = result["calibration"]
            eff = result["efficiency"]
            if len(eff) != 1:
                raise ValueError(f"expected one efficiency row for {backbone}/{arm}")
            rows.append({
                "backbone": backbone, "arm": arm, "role": role, "seed": seed,
                "checkpoint_sha256": result["manifest"]["checkpoint_sha256"],
                "ece_uncalibrated": cal["before_scaling"]["ece"],
                "ece_calibrated": cal["after_scaling"]["ece"],
                "brier_uncalibrated": cal["before_scaling"]["brier"],
                "brier_calibrated": cal["after_scaling"]["brier"],
                "temperature": cal["temperature"],
                "params_total": eff.iloc[0]["params_total"],
                "gflops": eff.iloc[0]["gflops"],
                "cpu_latency_ms": eff.iloc[0]["cpu_latency_ms"],
                "gpu_latency_ms": eff.iloc[0]["gpu_latency_ms"],
            })
    return pd.DataFrame(rows), paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path("artifacts/cnn_closeout/inference"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    table, paired = summarize(args.input_root, args.seed)
    output = args.output_dir or args.input_root / f"seed_{args.seed}_summary"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "cnn_inference_summary.csv", index=False)
    (output / "cnn_paired_interventions.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote six-model inference summary to {output}")


if __name__ == "__main__":
    main()
