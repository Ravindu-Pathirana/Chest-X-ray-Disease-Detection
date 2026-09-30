"""Build the final three-CNN closeout package from committed result artifacts.

This script performs no model inference and never invents missing results.  It
recomputes classification, uncalibrated calibration, anatomical-localisation,
paired significance, multi-seed, efficiency, and data-integrity summaries from
the per-image CSV/JSON artifacts already committed to the repository.

Checkpoint-dependent experiments (new counterfactual interventions,
temperature fitting, external inference, and new saliency methods) are listed
as unavailable in the generated report rather than represented by placeholders.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "cnn_closeout"
SEEDS = (42, 123, 2026)
CLASS_NAMES = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")
PROB_COLS = tuple(f"prob_{i}" for i in range(4))


@dataclass(frozen=True)
class BackboneSpec:
    backbone: str
    artifact_dir: str
    baseline_arm: str
    selected_arm: str
    seed42_baseline_dir: str
    seed42_selected_dir: str
    multiseed_baseline_pattern: str
    multiseed_selected_pattern: str
    comparison_seed42: str
    comparison_multiseed_pattern: str
    efficiency_file: str


SPECS = (
    BackboneSpec(
        "DenseNet121",
        "artifacts/T18_lung_attention",
        "A0_vanilla",
        "A2_full",
        "runs/A0_vanilla",
        "runs/A2_full",
        "runs_multiseed/A0_vanilla_seed{seed}",
        "runs_multiseed/A2_full_seed{seed}",
        "T18_comparison_table.csv",
        "runs_multiseed/T18_comparison_table_seed{seed}.csv",
        "T18_efficiency.csv",
    ),
    BackboneSpec(
        "ResNet50",
        "artifacts/T23_resnet50_lung_attention",
        "A0_vanilla",
        "A3_multiply",
        "runs/A0_vanilla",
        "runs/A3_multiply",
        "runs_multiseed/A0_vanilla_seed{seed}",
        "runs_multiseed/A3_multiply_seed{seed}",
        "T23_comparison_table.csv",
        "runs_multiseed/T23_comparison_table_seed{seed}.csv",
        "T23_efficiency.csv",
    ),
    BackboneSpec(
        "EfficientNet-B0",
        "artifacts/T25_efficientnet_b0_lung_attention",
        "A0",
        "A3",
        "runs/A0",
        "runs/A3",
        "runs_multiseed/seed_{seed}/runs/A0",
        "runs_multiseed/seed_{seed}/runs/A3",
        "T25_comparison_table.csv",
        "runs_multiseed/seed_{seed}/T25_comparison_table.csv",
        "T25_efficiency.csv",
    ),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        chosen = (confidence > lo) & (confidence <= hi)
        if chosen.any():
            ece += chosen.mean() * abs(correct[chosen].mean() - confidence[chosen].mean())
    return float(ece)


def multiclass_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    # Kept explicit to match src/modules/calibration.py's summed multiclass definition.
    one_hot = np.eye(probs.shape[1], dtype=float)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def prediction_dir(spec: BackboneSpec, seed: int, selected: bool) -> Path:
    base = ROOT / spec.artifact_dir
    if seed == 42:
        rel = spec.seed42_selected_dir if selected else spec.seed42_baseline_dir
    else:
        pattern = spec.multiseed_selected_pattern if selected else spec.multiseed_baseline_pattern
        rel = pattern.format(seed=seed)
    return base / rel


def comparison_file(spec: BackboneSpec, seed: int) -> Path:
    base = ROOT / spec.artifact_dir
    rel = spec.comparison_seed42 if seed == 42 else spec.comparison_multiseed_pattern.format(seed=seed)
    return base / rel


def find_comparison_row(spec: BackboneSpec, seed: int, arm: str) -> pd.Series:
    table = pd.read_csv(comparison_file(spec, seed))
    match = table.loc[table["arm"] == arm]
    if match.empty:
        raise ValueError(f"{comparison_file(spec, seed)} has no row for {arm}")
    return match.iloc[0]


def label_mapping(df: pd.DataFrame) -> dict[str, int]:
    observed = set(df["true_label"].unique())
    if observed != set(CLASS_NAMES):
        raise ValueError(f"unexpected labels: {sorted(observed)}")
    return {name: i for i, name in enumerate(CLASS_NAMES)}


def validate_predictions(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    missing = {"image_path", "true_label", "pred_label", *PROB_COLS, "ilar", "eil_post", "eil_pre"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    probs = df.loc[:, PROB_COLS].to_numpy(float)
    sums = probs.sum(axis=1)
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(df)),
        "unique_image_paths": int(df["image_path"].nunique()),
        "duplicate_image_paths": int(df["image_path"].duplicated().sum()),
        "probability_sum_max_abs_error": float(np.max(np.abs(sums - 1.0))),
        "probabilities_in_unit_interval": bool(((probs >= 0) & (probs <= 1)).all()),
        "eil_rows": int(df["eil_post"].notna().sum()),
        "ilar_rows": int(df["ilar"].notna().sum()),
    }


def compute_metrics(df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mapping = label_mapping(df)
    y_true = df["true_label"].map(mapping).to_numpy(int)
    y_pred = df["pred_label"].map(mapping).to_numpy(int)
    probs = df.loc[:, PROB_COLS].to_numpy(float)
    report = classification_report(
        y_true, y_pred, labels=np.arange(4), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    try:
        auc = float(roc_auc_score(y_true, probs, labels=np.arange(4), multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    overall = {
        "n_test": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_auc": auc,
        "ece_15bin_uncalibrated": expected_calibration_error(probs, y_true),
        "brier_uncalibrated": multiclass_brier(probs, y_true),
        "ilar": float(df["ilar"].mean()) if df["ilar"].notna().any() else float("nan"),
        "eil_post": float(df["eil_post"].mean()),
        "eil_pre": float(df["eil_pre"].mean()),
        "n_eil": int(df["eil_post"].notna().sum()),
    }
    per_class = []
    for name in CLASS_NAMES:
        values = report[name]
        per_class.append({
            "class": name,
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1": float(values["f1-score"]),
            "support": int(values["support"]),
        })
    return overall, per_class


def paired_significance(reference: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    merged = reference.merge(selected, on="image_path", suffixes=("_baseline", "_selected"), validate="one_to_one")
    ref_correct = merged["true_label_baseline"] == merged["pred_label_baseline"]
    sel_correct = merged["true_label_selected"] == merged["pred_label_selected"]
    b = int((ref_correct & ~sel_correct).sum())
    c = int((~ref_correct & sel_correct).sum())
    discordant = b + c
    mcnemar_two = 1.0 if discordant == 0 else float(binomtest(min(b, c), discordant, 0.5).pvalue)
    # H1: selected model is worse, which means b > c.
    mcnemar_worse = 1.0 if discordant == 0 else float(binomtest(b, discordant, 0.5, alternative="greater").pvalue)

    eil = merged.loc[
        merged["eil_post_baseline"].notna() & merged["eil_post_selected"].notna(),
        ["eil_post_baseline", "eil_post_selected"],
    ]
    diffs = eil["eil_post_selected"].to_numpy() - eil["eil_post_baseline"].to_numpy()
    if len(diffs) and not np.allclose(diffs, 0):
        w_two = wilcoxon(diffs, alternative="two-sided")
        w_gain = wilcoxon(diffs, alternative="greater")
        w_two_stat, w_two_p = float(w_two.statistic), float(w_two.pvalue)
        w_gain_stat, w_gain_p = float(w_gain.statistic), float(w_gain.pvalue)
    else:
        w_two_stat = w_two_p = w_gain_stat = w_gain_p = float("nan")
    return {
        "n_test_images": int(len(merged)),
        "mcnemar_baseline_right_selected_wrong": b,
        "mcnemar_baseline_wrong_selected_right": c,
        "mcnemar_two_sided_p": mcnemar_two,
        "mcnemar_one_sided_selected_worse_p": mcnemar_worse,
        "n_paired_eil": int(len(diffs)),
        "mean_eil_gain": float(np.mean(diffs)),
        "median_eil_gain": float(np.median(diffs)),
        "fraction_eil_improved": float(np.mean(diffs > 0)),
        "wilcoxon_two_sided_stat": w_two_stat,
        "wilcoxon_two_sided_p": w_two_p,
        "wilcoxon_gain_stat": w_gain_stat,
        "wilcoxon_gain_one_sided_p": w_gain_p,
    }


def sample_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sample_sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "n_seeds": int(len(arr)),
        "values": [float(v) for v in arr],
        "seeds": list(SEEDS),
    }


def load_efficiency(spec: BackboneSpec) -> pd.DataFrame:
    df = pd.read_csv(ROOT / spec.artifact_dir / spec.efficiency_file)
    if "arm" in df.columns:  # EfficientNet table already carries A0-A6.
        selected = df[df["arm"].isin([spec.baseline_arm, spec.selected_arm])].copy()
        selected["role"] = np.where(selected["arm"] == spec.baseline_arm, "baseline", "selected")
    elif "module" in df.columns and {spec.baseline_arm, spec.selected_arm} <= set(df["module"].astype(str)):
        # T25 calls the arm column "module"; unlike T18/T23 its values are
        # A0..A6 rather than without_module/with_module.
        selected = df[df["module"].isin([spec.baseline_arm, spec.selected_arm])].copy()
        selected["arm"] = selected["module"]
        selected["role"] = np.where(selected["arm"] == spec.baseline_arm, "baseline", "selected")
    else:
        selected = df.copy()
        selected["role"] = np.where(selected["module"] == "without_module", "baseline", "selected")
        selected["arm"] = np.where(selected["role"] == "baseline", spec.baseline_arm, spec.selected_arm)
    selected.insert(0, "backbone", spec.backbone)
    keep = [
        "backbone", "role", "arm", "params_total", "gflops", "state_dict_size_mb",
        "checkpoint_size_mb", "cpu_latency_ms", "cpu_throughput_img_s",
        "gpu_latency_ms", "gpu_throughput_img_s",
    ]
    for col in keep:
        if col not in selected:
            selected[col] = np.nan
    return selected[keep]


def build_figures(summary_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"DenseNet121": "#4472C4", "ResNet50": "#ED7D31", "EfficientNet-B0": "#70AD47"}

    order = [spec.backbone for spec in SPECS]
    paired = summary_df.set_index("backbone").loc[order]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(order, paired["delta_eil_mean"] * 100, yerr=paired["delta_eil_sample_sd"] * 100,
           color=[colors[x] for x in order], capsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Selected arm − baseline EIL (percentage points)")
    ax.set_title("Anatomical evidence relocation across three seeds")
    fig.tight_layout()
    fig.savefig(figures / "eil_gain_by_backbone.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for _, row in summary_df.iterrows():
        ax.scatter(row["delta_macro_f1_mean"] * 100, row["delta_eil_mean"] * 100,
                   s=100, color=colors[row["backbone"]], label=row["backbone"])
        ax.annotate(row["backbone"], (row["delta_macro_f1_mean"] * 100, row["delta_eil_mean"] * 100),
                    xytext=(6, 5), textcoords="offset points")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Selected arm − baseline macro-F1 (percentage points)")
    ax.set_ylabel("Selected arm − baseline EIL (percentage points)")
    ax.set_title("Classification–localisation trade-off")
    fig.tight_layout()
    fig.savefig(figures / "classification_vs_eil.png", dpi=200)
    plt.close(fig)

    selected_eff = efficiency_df[efficiency_df["role"] == "selected"].copy()
    selected_eff = selected_eff.merge(summary_df[["backbone", "selected_macro_f1_mean"]], on="backbone")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for _, row in selected_eff.iterrows():
        ax.scatter(row["params_total"] / 1e6, row["selected_macro_f1_mean"] * 100,
                   s=100, color=colors[row["backbone"]])
        ax.annotate(row["backbone"], (row["params_total"] / 1e6, row["selected_macro_f1_mean"] * 100),
                    xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Three-seed mean macro-F1 (%)")
    ax.set_title("Selected CNN performance versus model size")
    fig.tight_layout()
    fig.savefig(figures / "efficiency_tradeoff.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {"expected_test_rows_per_file": 3175, "files": [], "pair_checks": []}
    significance: dict[str, Any] = {}
    multiseed: dict[str, Any] = {}

    for spec in SPECS:
        significance[spec.backbone] = {}
        by_role: dict[str, dict[str, list[float]]] = {
            "baseline": {"accuracy": [], "macro_f1": [], "eil_post": [], "ece": [], "brier": []},
            "selected": {"accuracy": [], "macro_f1": [], "eil_post": [], "ece": [], "brier": []},
        }
        deltas = {"accuracy": [], "macro_f1": [], "eil_post": [], "ece": [], "brier": []}
        for seed in SEEDS:
            frames: dict[str, pd.DataFrame] = {}
            metrics_for_seed: dict[str, dict[str, Any]] = {}
            for role, selected, arm in (
                ("baseline", False, spec.baseline_arm),
                ("selected", True, spec.selected_arm),
            ):
                path = prediction_dir(spec, seed, selected) / "per_image_predictions.csv"
                df = pd.read_csv(path)
                frames[role] = df
                integrity["files"].append(validate_predictions(df, path))
                metrics, per_class = compute_metrics(df)
                metrics_for_seed[role] = metrics
                comparison = find_comparison_row(spec, seed, arm)
                row = {
                    "backbone": spec.backbone,
                    "seed": seed,
                    "role": role,
                    "arm": arm,
                    **metrics,
                    "attention_dice": float(comparison.get("att_dice", comparison.get("attention_dice", np.nan))),
                    "attention_iou": float(comparison.get("attention_iou", np.nan)),
                }
                master_rows.append(row)
                for class_row in per_class:
                    class_rows.append({"backbone": spec.backbone, "seed": seed, "role": role, "arm": arm, **class_row})
                for metric, source in (
                    ("accuracy", "accuracy"), ("macro_f1", "macro_f1"), ("eil_post", "eil_post"),
                    ("ece", "ece_15bin_uncalibrated"), ("brier", "brier_uncalibrated"),
                ):
                    by_role[role][metric].append(float(metrics[source]))

            baseline_paths = set(frames["baseline"]["image_path"])
            selected_paths = set(frames["selected"]["image_path"])
            integrity["pair_checks"].append({
                "backbone": spec.backbone,
                "seed": seed,
                "baseline_rows": len(frames["baseline"]),
                "selected_rows": len(frames["selected"]),
                "common_paths": len(baseline_paths & selected_paths),
                "path_sets_identical": baseline_paths == selected_paths,
            })
            significance[spec.backbone][str(seed)] = paired_significance(frames["baseline"], frames["selected"])
            for metric, source in (
                ("accuracy", "accuracy"), ("macro_f1", "macro_f1"), ("eil_post", "eil_post"),
                ("ece", "ece_15bin_uncalibrated"), ("brier", "brier_uncalibrated"),
            ):
                deltas[metric].append(metrics_for_seed["selected"][source] - metrics_for_seed["baseline"][source])

        multiseed[spec.backbone] = {
            "baseline_arm": spec.baseline_arm,
            "selected_arm": spec.selected_arm,
            "baseline": {metric: sample_summary(values) for metric, values in by_role["baseline"].items()},
            "selected": {metric: sample_summary(values) for metric, values in by_role["selected"].items()},
            "paired_selected_minus_baseline": {metric: sample_summary(values) for metric, values in deltas.items()},
        }

    master = pd.DataFrame(master_rows)
    per_class_df = pd.DataFrame(class_rows)
    master.to_csv(OUTPUT / "cnn_master_results.csv", index=False)
    per_class_df.to_csv(OUTPUT / "cnn_per_class_results.csv", index=False)

    summary_rows = []
    for spec in SPECS:
        item = multiseed[spec.backbone]
        summary_rows.append({
            "backbone": spec.backbone,
            "baseline_arm": spec.baseline_arm,
            "selected_arm": spec.selected_arm,
            "baseline_accuracy_mean": item["baseline"]["accuracy"]["mean"],
            "selected_accuracy_mean": item["selected"]["accuracy"]["mean"],
            "delta_accuracy_mean": item["paired_selected_minus_baseline"]["accuracy"]["mean"],
            "delta_accuracy_sample_sd": item["paired_selected_minus_baseline"]["accuracy"]["sample_sd"],
            "baseline_macro_f1_mean": item["baseline"]["macro_f1"]["mean"],
            "selected_macro_f1_mean": item["selected"]["macro_f1"]["mean"],
            "delta_macro_f1_mean": item["paired_selected_minus_baseline"]["macro_f1"]["mean"],
            "delta_macro_f1_sample_sd": item["paired_selected_minus_baseline"]["macro_f1"]["sample_sd"],
            "baseline_eil_mean": item["baseline"]["eil_post"]["mean"],
            "selected_eil_mean": item["selected"]["eil_post"]["mean"],
            "delta_eil_mean": item["paired_selected_minus_baseline"]["eil_post"]["mean"],
            "delta_eil_sample_sd": item["paired_selected_minus_baseline"]["eil_post"]["sample_sd"],
            "baseline_ece_mean": item["baseline"]["ece"]["mean"],
            "selected_ece_mean": item["selected"]["ece"]["mean"],
            "delta_ece_mean": item["paired_selected_minus_baseline"]["ece"]["mean"],
            "baseline_brier_mean": item["baseline"]["brier"]["mean"],
            "selected_brier_mean": item["selected"]["brier"]["mean"],
            "delta_brier_mean": item["paired_selected_minus_baseline"]["brier"]["mean"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT / "cnn_multiseed_summary.csv", index=False)
    write_json(OUTPUT / "cnn_multiseed_summary.json", multiseed)
    write_json(OUTPUT / "cnn_paired_significance.json", significance)

    calibration_cols = [
        "backbone", "seed", "role", "arm", "n_test", "ece_15bin_uncalibrated", "brier_uncalibrated"
    ]
    master[calibration_cols].to_csv(OUTPUT / "cnn_uncalibrated_calibration.csv", index=False)

    efficiency = pd.concat([load_efficiency(spec) for spec in SPECS], ignore_index=True)
    efficiency.to_csv(OUTPUT / "cnn_efficiency_summary.csv", index=False)

    integrity["all_files_have_3175_rows"] = all(f["rows"] == 3175 for f in integrity["files"])
    integrity["all_files_have_unique_paths"] = all(f["duplicate_image_paths"] == 0 for f in integrity["files"])
    integrity["all_pairs_have_identical_paths"] = all(p["path_sets_identical"] for p in integrity["pair_checks"])
    integrity["all_probability_rows_valid"] = all(
        f["probabilities_in_unit_interval"] and f["probability_sum_max_abs_error"] < 1e-5
        for f in integrity["files"]
    )
    write_json(OUTPUT / "cnn_data_quality_report.json", integrity)
    build_figures(summary_df, efficiency)

    report_lines = [
        "# Three-CNN Closeout Report",
        "",
        "Generated entirely from committed per-image and efficiency artifacts; no model inference was performed.",
        "",
        "## Three-seed headline results",
        "",
        "| Backbone | Selected arm | Δ accuracy (pp) | Δ macro-F1 (pp) | Δ EIL (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['backbone']} | {row['selected_arm']} | "
            f"{100 * row['delta_accuracy_mean']:+.2f} ± {100 * row['delta_accuracy_sample_sd']:.2f} | "
            f"{100 * row['delta_macro_f1_mean']:+.2f} ± {100 * row['delta_macro_f1_sample_sd']:.2f} | "
            f"{100 * row['delta_eil_mean']:+.2f} ± {100 * row['delta_eil_sample_sd']:.2f} |"
        )
    report_lines += [
        "",
        "Values are paired selected-arm minus baseline means and sample standard deviations across seeds 42, 123 and 2026.",
        "",
        "## What this package closes",
        "",
        "- Three-backbone classification and anatomical-localisation comparison.",
        "- Three-seed repeatability for each selected CNN arm.",
        "- Paired image-level McNemar and Wilcoxon results.",
        "- Uncalibrated ECE/Brier values reconstructed from saved test probabilities.",
        "- Parameter, FLOP and latency comparison from committed efficiency artifacts.",
        "- Input-integrity checks for sample counts, image alignment and probability validity.",
        "",
        "## Results that cannot be reconstructed from saved predictions",
        "",
        "- DenseNet/ResNet counterfactual background predictions: modified-image inference is required.",
        "- Temperature-scaled DenseNet/ResNet calibration: validation logits/probabilities or checkpoints are required.",
        "- Lung-removal/occlusion results: modified-image inference is required.",
        "- RSNA external performance: external-image inference is required.",
        "- Additional saliency methods: model activations and gradients are required.",
        "",
        "EfficientNet's already committed counterfactual outputs can be reported as a focused case study, but they do not constitute a standardized three-backbone counterfactual comparison.",
        "",
        "## Interpretation boundary",
        "",
        "The reconstructed evidence supports claims about anatomical evidence localisation. It must not be presented as proof that shortcut reliance was reduced across all three CNNs.",
    ]
    (OUTPUT / "CNN_CLOSEOUT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    estimates = """# Remaining Checkpoint-Dependent Run Estimates

These are planning ranges, not guarantees. They assume one Kaggle-class GPU
(T4/P100 or similar), 224x224 inputs, cached/preprocessed data, and working
checkpoints. Data download, queue time, notebook recovery, and debugging are
not included unless stated.

| Remaining work | Minimum scope | Rough elapsed GPU time | All three seeds | Repository readiness |
|---|---|---:|---:|---|
| DenseNet + ResNet background counterfactuals | A0/selected, seed 42, zero/shuffle/noise | 1–3 h total | 3–8 h | Evaluation code exists; checkpoints required. |
| DenseNet + ResNet temperature scaling | Validation + test for A0/selected | 0.5–1.5 h total | 1.5–4 h | Harness exists; checkpoints or saved validation logits required. |
| Lung-versus-background occlusion | Six CNNs, seed 42 | 1–3 h | 3–8 h | Small extension to counterfactual code; checkpoints required. |
| RSNA external inference | Six CNNs, seed 42, about 26.7k studies | 3–8 h | 8–24 h | Dataset prepared; label mapping and final runner still required. |
| Additional Grad-CAM-style method | Six CNNs, fixed 1,000-image subset | 2–6 h | 6–18 h | New method integration and checkpoints required. |

## Important schedule risk

No `.pt`, `.pth`, or `.ckpt` model files are present in the repository. If the
trained weights cannot be recovered, the inference estimates above do not
apply. Retraining the selected baseline/attention pairs across three seeds can
reasonably require roughly 24–72 aggregate GPU-hours, depending on GPU type,
early stopping, caching, and notebook stability.

## Recommended stopping rule

The existing-data closeout is sufficient for the CNN anatomical-localisation
study. Run checkpoint-dependent work only if the weights are recovered quickly.
Prioritize, in order: (1) DenseNet/ResNet counterfactuals, (2) calibrated ECE,
(3) one occlusion experiment, and (4) RSNA. Do not allow ViT adaptation or a
new saliency family to delay the CNN result freeze.
"""
    (OUTPUT / "REMAINING_RUN_ESTIMATES.md").write_text(estimates, encoding="utf-8")
    print(f"Wrote CNN closeout package to {OUTPUT}")


if __name__ == "__main__":
    main()
