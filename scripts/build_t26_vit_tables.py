"""Convert the T26 ViT-Base seven-arm delivery into the team's shared artifact schema.

Reads artifacts/vit_lung_attention/runs/vit_base_lung_attention/seven_arm/<arm>/
{test_predictions_finetuned.csv,arm_result.json}
and writes, without touching the delivered files:

- seven_arm/<arm>/per_image_predictions.csv -- same columns as T18/T23/T25
  (image_path, true_label, pred_label, prob_0..prob_3, ilar, eil_post, eil_pre).
  image_path is made relative to the dataset root ("COVID/images/COVID-1.png"),
  matching artifacts/splits/split_manifest_v1.csv. eil_post/eil_pre come from the
  delivery's gradcam_per_image.csv (a 200-image subsample) and are NaN elsewhere;
  ilar is NaN throughout (only arm means were delivered). NaN, not 0, per the
  per-image convention.
- T26_comparison_table.csv -- same columns as T23_comparison_table.csv, plus
  eil_n_images (EIL here is over 200 images, not the 1,000 the CNN tables use).

Usage: python scripts/build_t26_vit_tables.py   (from the repo root)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("artifacts/vit_lung_attention")
RUN = ROOT / "runs" / "vit_base_lung_attention"   # the T26 owner's delivery (PR #38 layout)
ARMS = ["A0_vanilla", "A1_gate_only", "A2_full", "A3_multiply_gate", "A4_guidance_only", "A5_cbam", "A6_full_bg"]
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
MANIFEST = Path("artifacts/splits/split_manifest_v1.csv")


def relative_image_path(path: str) -> str:
    return "/".join(path.replace("\\", "/").split("/")[-3:])


def main() -> None:
    manifest_test = set(pd.read_csv(MANIFEST).query("split == 'test'").image_path)
    rows = []
    for arm in ARMS:
        arm_dir = RUN / "seven_arm" / arm
        pred = pd.read_csv(arm_dir / "test_predictions_finetuned.csv")
        out = pd.DataFrame({
            "image_path": pred.image_path.map(relative_image_path),
            "true_label": pred.y_true.map(dict(enumerate(CLASS_NAMES))),
            "pred_label": pred.y_pred.map(dict(enumerate(CLASS_NAMES))),
        })
        for i, name in enumerate(CLASS_NAMES):
            out[f"prob_{i}"] = pred[f"prob_{name}"]
        out["ilar"] = np.nan                      # only arm means were delivered
        out["eil_post"] = np.nan
        out["eil_pre"] = np.nan
        cam_path = arm_dir / "gradcam_per_image.csv"   # T26 owner's Grad-CAM, 200-image subsample
        if cam_path.exists():
            cam = pd.read_csv(cam_path)
            cam["image_path"] = cam.image_path.map(relative_image_path)
            cam = cam.set_index("image_path")
            hit = out.image_path.isin(cam.index)
            out.loc[hit, "eil_post"] = out.loc[hit, "image_path"].map(cam.eil_post)
            out.loc[hit, "eil_pre"] = out.loc[hit, "image_path"].map(cam.eil_pre)
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
            "EIL_post": float(out.eil_post.mean()) if out.eil_post.notna().any() else np.nan,
            "EIL_pre": float(out.eil_pre.mean()) if out.eil_pre.notna().any() else np.nan,
            "eil_n_images": int(out.eil_post.notna().sum()),
        })

    table = pd.DataFrame(rows)
    table["delta_macro_f1"] = table.test_macro_f1 - table.test_macro_f1.iloc[0]
    table["delta_eil_post"] = table.EIL_post - table.EIL_post.iloc[0]
    table["delta_eil_pre"] = table.EIL_pre - table.EIL_pre.iloc[0]
    table.to_csv(ROOT / "T26_comparison_table.csv", index=False)
    print(table[["arm", "test_acc", "test_macro_f1", "att_dice", "EIL_pre", "EIL_post", "delta_eil_post", "eil_n_images"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
