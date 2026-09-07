"""T18 sections S11 and S14: per-image predictions, the cross-arm
comparison table, the acceptance-criteria (section 1.3, A1-A4) check,
and (S14, optional) multi-seed mean +/- std summaries.

Three real dependencies this module has on prior work:
- evaluate() (training.py, S6) already writes attention_ilar/dice/iou
  (mean, not per-image) into each arm's test_results.json/summary_metrics.csv.
- cam_for()/get_taps()/energy_inside_lung() (gradcam.py, attention_metrics.py,
  S5) compute the module-agnostic Grad-CAM faithfulness metric this file
  aggregates across arms.
- ilar() (attention_metrics.py, S5) is reused per-image here, not just as
  the mean evaluate() already reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .attention_metrics import energy_inside_lung, ilar
from .gradcam import cam_for, get_taps


def stratified_cam_subset(test_dataset, n: int = 1000, seed: int = 42) -> np.ndarray:
    """A fixed, stratified subsample of TEST DATASET positions (0..len-1),
    the same across every arm (WBS section S11 budget note: "same images
    for every arm is non-negotiable; the number of images is negotiable").
    Returns all positions if n >= len(test_dataset) (no subsampling needed).
    """
    n_total = len(test_dataset)
    if n >= n_total:
        return np.arange(n_total)

    base = test_dataset.base_dataset
    targets = np.array([base.targets[i] for i in test_dataset.indices])
    positions = np.arange(n_total)
    subset, _ = train_test_split(positions, train_size=n, stratify=targets, random_state=seed)
    return np.sort(subset)


def build_per_image_predictions(
    model: torch.nn.Module,
    test_dataset,
    class_names: List[str],
    device: torch.device,
    cam_subset: Optional[np.ndarray] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Per-image CSV, WBS section 6.5: image_path, true_label, pred_label,
    prob_0..prob_{K-1}, ilar, eil_post, eil_pre.

    Classification (probs/preds) and ilar are computed over the FULL test
    dataset. eil_post/eil_pre are computed only for images whose position
    is in `cam_subset` (NaN elsewhere, per section 7.1's NaN-not-zero
    convention -- a 0 would silently read as "no faithfulness" for images
    that were simply never scored). Pass cam_subset=None to skip Grad-CAM
    entirely (e.g. a quick classification-only pass).

    Model must have already been moved to `device` and loaded with the
    checkpoint being evaluated.
    """
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    has_cam = cam_subset is not None and len(cam_subset) > 0
    tap_post = tap_pre = None
    if has_cam:
        tap_post, tap_pre = get_taps(model)
    cam_subset_set = set(int(i) for i in cam_subset) if has_cam else set()

    image_paths = [
        str(test_dataset.base_dataset.samples[test_dataset.indices[i]][0]) for i in range(len(test_dataset))
    ]

    model.eval()
    rows: List[Dict[str, Any]] = []
    offset = 0
    for images, labels, masks in loader:
        bsz = images.shape[0]
        batch_positions = list(range(offset, offset + bsz))

        images_d = images.to(device)
        masks_d = masks.to(device).float()
        if masks_d.ndim == 3:
            masks_d = masks_d.unsqueeze(1)

        with torch.no_grad():
            logits, att, _att_logits = model(images_d)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

        if att is not None:
            batch_ilar = ilar(att.detach().float(), masks_d).cpu().numpy()
        else:
            batch_ilar = [None] * bsz

        batch_eil_post: List[Optional[float]] = [None] * bsz
        batch_eil_pre: List[Optional[float]] = [None] * bsz
        if has_cam:
            local_subset_idx = [j for j, pos in enumerate(batch_positions) if pos in cam_subset_set]
            if local_subset_idx:
                sub_images = images[local_subset_idx]
                sub_masks = masks[local_subset_idx].float()
                if sub_masks.ndim == 3:
                    sub_masks = sub_masks.unsqueeze(1)
                cam_post, _ = cam_for(model, sub_images, tap_post, device)
                cam_pre, _ = cam_for(model, sub_images, tap_pre, device)
                eil_post_vals = energy_inside_lung(cam_post, sub_masks.to(device)).detach().cpu().numpy()
                eil_pre_vals = energy_inside_lung(cam_pre, sub_masks.to(device)).detach().cpu().numpy()
                for k, j in enumerate(local_subset_idx):
                    batch_eil_post[j] = float(eil_post_vals[k])
                    batch_eil_pre[j] = float(eil_pre_vals[k])

        for j in range(bsz):
            row: Dict[str, Any] = {
                "image_path": image_paths[offset + j],
                "true_label": class_names[int(labels[j].item())],
                "pred_label": class_names[int(preds[j])],
            }
            for k, cname in enumerate(class_names):
                row[f"prob_{k}"] = float(probs[j, k])
            row["ilar"] = float(batch_ilar[j]) if batch_ilar[j] is not None else float("nan")
            row["eil_post"] = batch_eil_post[j] if batch_eil_post[j] is not None else float("nan")
            row["eil_pre"] = batch_eil_pre[j] if batch_eil_pre[j] is not None else float("nan")
            rows.append(row)

        offset += bsz

    return pd.DataFrame(rows)


def build_comparison_table(
    arms: Dict[str, Dict[str, Any]],
    reference_arm: Optional[str] = None,
    output_csv=None,
) -> pd.DataFrame:
    """The WBS section S11 comparison table.

    `arms`: {arm_name: {"gate_mode": str, "lambda_att": float,
                        "evaluate_results": dict (evaluate()'s return value),
                        "per_image_df": pd.DataFrame (build_per_image_predictions())}}

    Deltas (delta_macro_f1, delta_eil_post, delta_eil_pre) are relative to
    `reference_arm` if given, else whichever arm has gate_mode=="none" and
    lambda_att==0.0 (arm A0's shape), else the first arm in `arms`.
    """
    rows = []
    for name, a in arms.items():
        res = a["evaluate_results"]
        report = res["classification_report"]
        pidf = a["per_image_df"]
        eil_post_col = pidf["eil_post"].dropna()
        eil_pre_col = pidf["eil_pre"].dropna()
        rows.append({
            "arm": name,
            "gate": a["gate_mode"],
            "lambda": a["lambda_att"],
            "test_acc": report["accuracy"],
            "test_macro_f1": report["macro avg"]["f1-score"],
            "test_auc_macro": res.get("roc_auc_macro"),
            "ILAR": res.get("attention_ilar") if res.get("attention_ilar") is not None else float("nan"),
            "att_dice": res.get("attention_dice") if res.get("attention_dice") is not None else float("nan"),
            "EIL_post": float(eil_post_col.mean()) if len(eil_post_col) else float("nan"),
            "EIL_pre": float(eil_pre_col.mean()) if len(eil_pre_col) else float("nan"),
        })
    df = pd.DataFrame(rows)

    if reference_arm is not None:
        ref = df[df["arm"] == reference_arm]
    else:
        ref = df[(df["gate"] == "none") & (df["lambda"] == 0.0)]
    ref_row = ref.iloc[0] if not ref.empty else df.iloc[0]

    df["delta_macro_f1"] = df["test_macro_f1"] - ref_row["test_macro_f1"]
    df["delta_eil_post"] = df["EIL_post"] - ref_row["EIL_post"]
    df["delta_eil_pre"] = df["EIL_pre"] - ref_row["EIL_pre"]

    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return df


def check_acceptance_criteria(comparison_df: pd.DataFrame, headline_arm: str = "A2_full") -> Dict[str, Dict[str, Any]]:
    """WBS section 1.3, criteria A1-A4 -- computed here in S11 (A5-A9 are
    checked elsewhere: A5 by S8-S10's completeness, A6 by S13's efficiency
    numbers, A7/A9 by S15's handoff, A8 by S12's figure count).

    These are goals, not gates on effort (WBS section 1.3's note): a failing
    verdict is a legitimate finding to report, not something to keep tuning
    against. Every verdict includes the actual number, not just pass/fail.
    """
    match = comparison_df[comparison_df["arm"] == headline_arm]
    if match.empty:
        raise KeyError(f"headline_arm '{headline_arm}' not found in comparison_df's arm column")
    row = match.iloc[0]

    return {
        "A1_attention_matches_lung_mask": {
            "target": "Attention-Dice >= 0.80 on test", "value": float(row["att_dice"]),
            "pass": bool(row["att_dice"] >= 0.80),
        },
        "A2_evidence_inside_lung_up": {
            "target": "delta Grad-CAM EIL (post-gate) >= +0.05", "value": float(row["delta_eil_post"]),
            "pass": bool(row["delta_eil_post"] >= 0.05),
        },
        "A3_mechanism_is_real": {
            "target": "delta Grad-CAM EIL (pre-gate) > 0", "value": float(row["delta_eil_pre"]),
            "pass": bool(row["delta_eil_pre"] > 0),
        },
        "A4_no_accuracy_tax": {
            "target": "delta test macro-F1 >= -1.0pp vs. A0", "value": float(row["delta_macro_f1"]),
            "pass": bool(row["delta_macro_f1"] >= -0.01),
        },
    }


def summarize_multiseed(
    per_seed_comparison_dfs: Dict[int, pd.DataFrame],
    arms: Optional[List[str]] = None,
    metrics: List[str] = ["test_macro_f1", "EIL_post"],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean +/- std across seeds for the given metrics, per arm (WBS
    section S14): "A2 beat A0 by 0.06 EIL" -> "A2 beat A0 by 0.06 +/- 0.01".

    `per_seed_comparison_dfs`: {seed: comparison_df} -- one
    build_comparison_table() output per seed (e.g. {42: df_seed42,
    123: df_seed123, 2026: df_seed2026}), each with the same arm names in
    its "arm" column.

    Requires >= 2 seeds. A single seed's "mean +/- std" is meaningless and
    the WBS explicitly warns against implying repeats that weren't run
    (section S14: "do not imply repeats you didn't run") -- raises
    ValueError rather than silently returning std=0 or std=NaN for one
    seed, which could be mistaken for a real (if small) spread.

    Returns {arm_name: {metric: {"mean": float, "std": float, "n_seeds": int,
    "values": [float, ...]}}}. `values` is kept so a caller can report the
    raw per-seed numbers alongside the summary, not just the aggregate.
    """
    if len(per_seed_comparison_dfs) < 2:
        raise ValueError(
            f"summarize_multiseed needs >= 2 seeds to report a meaningful mean +/- std; "
            f"got {len(per_seed_comparison_dfs)}. If only one seed was run, state that "
            f"explicitly (WBS section S14) rather than calling this function."
        )

    seeds = sorted(per_seed_comparison_dfs.keys())
    if arms is None:
        first_df = per_seed_comparison_dfs[seeds[0]]
        arms = list(first_df["arm"].unique())

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm_name in arms:
        summary[arm_name] = {}
        for metric in metrics:
            values = []
            for seed in seeds:
                df = per_seed_comparison_dfs[seed]
                match = df[df["arm"] == arm_name]
                if match.empty:
                    raise KeyError(f"arm '{arm_name}' not found in seed {seed}'s comparison_df")
                values.append(float(match.iloc[0][metric]))
            arr = np.array(values)
            summary[arm_name][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)),  # sample std -- these are a sample of seeds, not the population
                "n_seeds": len(values),
                "values": values,
                "seeds": seeds,
            }
    return summary
