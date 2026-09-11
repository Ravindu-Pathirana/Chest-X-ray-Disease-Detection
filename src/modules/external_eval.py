"""External-test protocol: RSNA label mapping + out-of-distribution (OOD)
robustness metrics (T32).

This is Axis 5 (Robustness) of the trustworthiness framework: internal test
performance vs. external RSNA performance, on the SAME already-trained
model, per docs/experiment_policy.md and the proposal (DNN.pdf section 5.1):
"the difference between internal and external accuracy and ROC-AUC."

The open design question this module exists to resolve (flagged explicitly
in "Project Documents/Each task description.md", P15): the primary model
predicts 4 classes (COVID, Lung_Opacity, Normal, Viral Pneumonia), but RSNA
only has 2 (Normal, Pneumonia). There is no single obviously-correct way to
make these comparable, so this module implements BOTH candidate mappings
discussed with the team rather than silently picking one:

- "binary_collapse": Normal vs. Abnormal (COVID + Lung_Opacity + Viral
  Pneumonia all collapsed into "Abnormal"). Uses every image on both sides.
  Clinically loose -- "Abnormal" lumps together unrelated conditions.
- "class_matched": Normal vs. (Viral) Pneumonia only. Drops COVID/
  Lung_Opacity images from the *internal* side for this specific
  comparison, and renormalizes the model's softmax over just the
  Normal/Viral-Pneumonia channels for both internal and external images.
  More clinically defensible (RSNA's "Pneumonia" is closer in meaning to
  "Viral Pneumonia" than to the broader "Lung Opacity"), but uses less data.

Which one is "the" number for the paper is a team decision, not this
module's -- run_external_test_protocol() takes `mapping` as an explicit
argument and reports which one was used in its output, so results are never
ambiguous about which comparison produced them.

Both entry points consume the same per-image-predictions DataFrame shape
that comparison.py's build_per_image_predictions() already produces
(image_path, true_label, pred_label, prob_0..prob_{K-1}), so this plugs
directly into the existing pipeline without a new prediction format.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# Primary dataset's 4-class label -> 2-class label, "binary_collapse" mapping.
BINARY_COLLAPSE_MAP = {
    "COVID": "Abnormal",
    "Lung_Opacity": "Abnormal",
    "Normal": "Normal",
    "Viral Pneumonia": "Abnormal",
}

# RSNA's native 2-class label -> the same 2-class label, "binary_collapse" mapping.
RSNA_BINARY_COLLAPSE_MAP = {
    "Normal": "Normal",
    "Pneumonia": "Abnormal",
}

# Primary dataset classes kept for the "class_matched" mapping -- COVID and
# Lung_Opacity are dropped because RSNA has no comparable class for them.
CLASS_MATCHED_KEEP = ["Normal", "Viral Pneumonia"]

# RSNA's native label -> the primary dataset's matching class name, so both
# sides share identical label strings ("Viral Pneumonia", not "Pneumonia").
RSNA_CLASS_MATCHED_MAP = {
    "Normal": "Normal",
    "Pneumonia": "Viral Pneumonia",
}


def _prob_columns(class_names: List[str]) -> List[str]:
    return [f"prob_{i}" for i in range(len(class_names))]


def map_binary_collapse(per_image_df: pd.DataFrame, class_names: List[str], source: str) -> pd.DataFrame:
    """Reduces a 4-class (or RSNA's native 2-class) per-image predictions
    DataFrame to the "binary_collapse" mapping: Normal vs. Abnormal.

    `source` is `"primary"` (4-class softmax, needs summing) or `"rsna"`
    (already 2-class, needs only relabeling) -- the two datasets start from
    different numbers of columns, so they're handled by two branches, not
    one that guesses from the DataFrame's shape.

    Returns a DataFrame with `image_path`, `true_binary` (Normal/Abnormal),
    `prob_abnormal` (probability of the Abnormal class, in [0, 1]).
    """
    if source == "primary":
        prob_cols = _prob_columns(class_names)
        abnormal_cols = [
            f"prob_{i}" for i, name in enumerate(class_names) if BINARY_COLLAPSE_MAP[name] == "Abnormal"
        ]
        prob_abnormal = per_image_df[abnormal_cols].sum(axis=1)
        true_binary = per_image_df["true_label"].map(BINARY_COLLAPSE_MAP)
    elif source == "rsna":
        # RSNA predictions are still produced by the 4-class model (T32 runs
        # the already-trained model on RSNA images), so prob columns are
        # still indexed by the primary dataset's class_names -- only the
        # ground-truth label is natively RSNA's own Normal/Pneumonia.
        prob_cols = _prob_columns(class_names)
        abnormal_cols = [
            f"prob_{i}" for i, name in enumerate(class_names) if BINARY_COLLAPSE_MAP[name] == "Abnormal"
        ]
        prob_abnormal = per_image_df[abnormal_cols].sum(axis=1)
        true_binary = per_image_df["true_label"].map(RSNA_BINARY_COLLAPSE_MAP)
    else:
        raise ValueError(f"source must be 'primary' or 'rsna', got {source!r}")

    if true_binary.isna().any():
        unmapped = per_image_df.loc[true_binary.isna(), "true_label"].unique().tolist()
        raise ValueError(f"Unmapped label(s) found for source={source!r}: {unmapped}")

    return pd.DataFrame(
        {
            "image_path": per_image_df["image_path"],
            "true_binary": true_binary,
            "prob_abnormal": prob_abnormal,
        }
    )


def map_class_matched(per_image_df: pd.DataFrame, class_names: List[str], source: str) -> pd.DataFrame:
    """Reduces predictions to the "class_matched" mapping: Normal vs. Viral
    Pneumonia only, dropping every other class/image.

    For `source="primary"`: drops rows whose true_label isn't in
    CLASS_MATCHED_KEEP, then renormalizes the softmax over just the
    Normal/Viral-Pneumonia columns (so the two probabilities sum to 1 again
    -- comparing a raw, un-renormalized "Viral Pneumonia" probability
    against RSNA's binary Pneumonia probability would be comparing
    different scales, since the primary model's Viral Pneumonia column
    also competes against COVID and Lung_Opacity).

    For `source="rsna"`: no rows are dropped (RSNA only has these 2 classes
    already), but the same column renormalization is applied for
    consistency -- the model was still run as a 4-class classifier on RSNA
    images, so its raw Normal/Viral-Pneumonia columns don't sum to 1 on
    their own either.

    Returns a DataFrame with `image_path`, `true_matched` (Normal/Viral
    Pneumonia), `prob_pneumonia_equivalent` (renormalized probability of
    the Viral Pneumonia class, in [0, 1]).
    """
    keep_cols = [f"prob_{i}" for i, name in enumerate(class_names) if name in CLASS_MATCHED_KEEP]
    pneumonia_col = f"prob_{class_names.index('Viral Pneumonia')}"

    if source == "primary":
        true_matched = per_image_df["true_label"]
        keep_mask = true_matched.isin(CLASS_MATCHED_KEEP)
        df = per_image_df.loc[keep_mask].copy()
        true_matched = true_matched.loc[keep_mask]
    elif source == "rsna":
        df = per_image_df.copy()
        true_matched = df["true_label"].map(RSNA_CLASS_MATCHED_MAP)
        if true_matched.isna().any():
            unmapped = df.loc[true_matched.isna(), "true_label"].unique().tolist()
            raise ValueError(f"Unmapped RSNA label(s): {unmapped}")
    else:
        raise ValueError(f"source must be 'primary' or 'rsna', got {source!r}")

    kept_prob_sum = df[keep_cols].sum(axis=1)
    if (kept_prob_sum == 0).any():
        raise ValueError(
            "Found row(s) with zero total probability across the kept classes -- "
            "cannot renormalize; check the prob_* columns are real softmax outputs."
        )
    prob_pneumonia_equivalent = df[pneumonia_col] / kept_prob_sum

    return pd.DataFrame(
        {
            "image_path": df["image_path"],
            "true_matched": true_matched,
            "prob_pneumonia_equivalent": prob_pneumonia_equivalent,
        }
    )


def compute_ood_metrics(
    internal_df: pd.DataFrame,
    external_df: pd.DataFrame,
    true_column: str,
    prob_column: str,
    positive_label: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes internal vs. external AUC/accuracy and their drop
    (internal - external), per DNN.pdf's stated robustness definition.

    Works on either mapping's output DataFrame -- pass `true_column`/
    `prob_column` matching whichever mapping produced `internal_df`/
    `external_df` (e.g. "true_binary"/"prob_abnormal" for binary_collapse,
    "true_matched"/"prob_pneumonia_equivalent" for class_matched).

    A positive AUC/accuracy drop means the model is doing worse
    out-of-distribution than in-distribution -- the shortcut-learning
    signature this whole project is built to detect and reduce.
    """
    internal_true = (internal_df[true_column] == positive_label).astype(int)
    external_true = (external_df[true_column] == positive_label).astype(int)

    internal_pred = (internal_df[prob_column] >= threshold).astype(int)
    external_pred = (external_df[prob_column] >= threshold).astype(int)

    internal_auc = roc_auc_score(internal_true, internal_df[prob_column])
    external_auc = roc_auc_score(external_true, external_df[prob_column])
    internal_acc = accuracy_score(internal_true, internal_pred)
    external_acc = accuracy_score(external_true, external_pred)

    return {
        "internal_auc": float(internal_auc),
        "external_auc": float(external_auc),
        "auc_drop": float(internal_auc - external_auc),
        "internal_accuracy": float(internal_acc),
        "external_accuracy": float(external_acc),
        "accuracy_drop": float(internal_acc - external_acc),
        "n_internal": int(len(internal_df)),
        "n_external": int(len(external_df)),
    }


def run_external_test_protocol(
    internal_per_image_df: pd.DataFrame,
    external_per_image_df: pd.DataFrame,
    class_names: List[str],
    mapping: str = "binary_collapse",
) -> Dict[str, object]:
    """Main entry point: runs the chosen mapping end-to-end and returns OOD
    metrics, tagged with which mapping produced them so results are never
    ambiguous about which of the two comparisons (binary_collapse vs.
    class_matched) they represent.

    `internal_per_image_df` / `external_per_image_df` must have the shape
    comparison.py's build_per_image_predictions() produces: image_path,
    true_label, prob_0..prob_{K-1} (indexed per `class_names`).
    `external_per_image_df`'s true_label values must be RSNA's native
    "Normal"/"Pneumonia" strings, not already remapped.
    """
    if mapping == "binary_collapse":
        internal_mapped = map_binary_collapse(internal_per_image_df, class_names, source="primary")
        external_mapped = map_binary_collapse(external_per_image_df, class_names, source="rsna")
        metrics = compute_ood_metrics(
            internal_mapped, external_mapped,
            true_column="true_binary", prob_column="prob_abnormal", positive_label="Abnormal",
        )
    elif mapping == "class_matched":
        internal_mapped = map_class_matched(internal_per_image_df, class_names, source="primary")
        external_mapped = map_class_matched(external_per_image_df, class_names, source="rsna")
        metrics = compute_ood_metrics(
            internal_mapped, external_mapped,
            true_column="true_matched", prob_column="prob_pneumonia_equivalent", positive_label="Viral Pneumonia",
        )
    else:
        raise ValueError(f"mapping must be 'binary_collapse' or 'class_matched', got {mapping!r}")

    metrics["mapping"] = mapping
    return metrics
