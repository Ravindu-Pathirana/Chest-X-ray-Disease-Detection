"""Build retrospective paired EIL confidence intervals from saved CNN outputs.

No checkpoints, images, model inference, or selection decisions are used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_cnn_closeout import ROOT, SEEDS, SPECS, prediction_dir
from src.modules.xai_statistics import holm_adjust, paired_eil


OUTPUT = ROOT / "artifacts" / "explainable_ai"


def build() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        seed_rows = []
        for spec in SPECS:
            baseline_path = prediction_dir(spec, seed, False) / "per_image_predictions.csv"
            selected_path = prediction_dir(spec, seed, True) / "per_image_predictions.csv"
            metrics = paired_eil(pd.read_csv(baseline_path), pd.read_csv(selected_path))
            seed_rows.append({
                "backbone": spec.backbone, "seed": seed,
                "baseline_arm": spec.baseline_arm, "selected_arm": spec.selected_arm,
                "baseline_source": baseline_path.relative_to(ROOT).as_posix(),
                "selected_source": selected_path.relative_to(ROOT).as_posix(),
                **metrics,
            })
        adjusted = holm_adjust([row["wilcoxon_two_sided_p"] for row in seed_rows])
        for row, p_adjusted in zip(seed_rows, adjusted):
            row["wilcoxon_holm_p_within_seed_three_backbones"] = p_adjusted
        rows.extend(seed_rows)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = build()
    table.to_csv(OUTPUT / "paired_eil_bootstrap.csv", index=False)
    lines = [
        "# Explainability baseline: paired CNN localization",
        "",
        "Retrospective analysis of already committed test predictions; no new inference was run.",
        "The same 1,000 CAM-scored images are paired within each backbone and seed.",
        "Intervals are percentile 95% paired-image bootstrap CIs (2,000 resamples, seed 42).",
        "Wilcoxon p-values are two-sided and Holm-adjusted across the three backbones within each seed.",
        "This family choice is descriptive; it was not preregistered before the original test results were seen.",
        "",
        "| Backbone | Seed | Mean EIL gain | 95% CI | Fraction improved |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.backbone} | {row.seed} | {row.delta_eil_mean:+.4f} | "
            f"[{row.ci_95_low:+.4f}, {row.ci_95_high:+.4f}] | "
            f"{row.fraction_images_improved:.1%} |"
        )
    lines += [
        "",
        "The selected arms increase Grad-CAM energy inside the lung mask. This is an",
        "anatomical-localization result. Grad-CAM placement alone cannot establish",
        "causal explanation faithfulness or reduced background shortcut reliance.",
        "See `paired_eil_bootstrap.csv` for source files and exact statistics.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(table)} paired EIL comparisons to {OUTPUT}")


if __name__ == "__main__":
    main()
