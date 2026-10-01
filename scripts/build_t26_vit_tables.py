"""Convert the T26 ViT-Base seven-arm delivery into the team's shared artifact schema.

Reads artifacts/vit_lung_attention/seven_arm/<arm>/{test_predictions_finetuned.csv,arm_result.json}
and writes, without touching the delivered files:

- seven_arm/<arm>/per_image_predictions.csv -- same columns as T18/T23/T25
  (image_path, true_label, pred_label, prob_0..prob_3, ilar, eil_post, eil_pre).
  image_path is made relative to the dataset root ("COVID/images/COVID-1.png"),
  matching artifacts/splits/split_manifest_v1.csv. ilar/eil_post/eil_pre are NaN:
  the delivery has no per-image attention or Grad-CAM values (NaN, not 0, per
  the per-image convention).
- T26_comparison_table.csv -- same columns as T23_comparison_table.csv, with
  EIL columns NaN until Grad-CAM is run on the ViT checkpoints.

Usage: python scripts/build_t26_vit_tables.py   (from the repo root)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("artifacts/vit_lung_attention")
ARMS = ["A0_vanilla", "A1_gate_only", "A2_full", "A3_multiply_gate", "A4_guidance_only", "A5_cbam", "A6_full_bg"]
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
MANIFEST = Path("artifacts/splits/split_manifest_v1.csv")


def relative_image_path(path: str) -> str:
    return "/".join(path.replace("\\", "/").split("/")[-3:])


def main() -> None:
    manifest_test = set(pd.read_csv(MANIFEST).query("split == 'test'").image_path)
    rows = []
    for arm in ARMS:
        arm_dir = ROOT / "seven_arm" / arm
        pred = pd.read_csv(arm_dir / "test_predictions_finetuned.csv")
        out = pd.DataFrame({
            "image_path": pred.image_path.map(relative_image_path),
            "true_label": pred.y_true.map(dict(enumerate(CLASS_NAMES))),
            "pred_label": pred.y_pred.map(dict(enumerate(CLASS_NAMES))),
        })
        for i, name in enumerate(CLASS_NAMES):
            out[f"prob_{i}"] = pred[f"prob_{name}"]
        out["ilar"] = np.nan
        out["eil_post"] = np.nan
        out["eil_pre"] = np.nan
        if set(out.image_path) != manifest_test:
            raise ValueError(f"{arm}: test images differ from {MANIFEST}")
        out.to_csv(arm_dir / "per_image_predictions.csv", index=False)

        r = json.loads((arm_dir / "arm_result.json").read_text())
        rows.append({
            "arm": arm,
            "gate": "none" if not r["use_attention"] or r["gate_mode"] == "none" else r["gate_mode"],
            "attention": r["attention"],
            "lambda": r["lambda_att"],
            "lambda_bg": r.get("lambda_bg", 0.0),
            "test_acc": r["test_accuracy"],
            "test_macro_f1": r["test_macro_f1"],
            "test_auc_macro": r["test_macro_roc_auc"],
            "ILAR": r.get("test_ilar") if r["use_attention"] else np.nan,
            "att_dice": r.get("test_dice") if r["use_attention"] else np.nan,
            "background_attention": r.get("test_background_attention") if r["use_attention"] else np.nan,
            "EIL_post": np.nan,
            "EIL_pre": np.nan,
        })

    table = pd.DataFrame(rows)
    table["delta_macro_f1"] = table.test_macro_f1 - table.test_macro_f1.iloc[0]
    table["delta_eil_post"] = np.nan
    table["delta_eil_pre"] = np.nan
    table.to_csv(ROOT / "T26_comparison_table.csv", index=False)
    print(table[["arm", "lambda", "lambda_bg", "test_acc", "test_macro_f1", "att_dice", "delta_macro_f1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
