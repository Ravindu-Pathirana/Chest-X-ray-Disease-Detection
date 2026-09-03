"""T18 section S12: heat-map overlay figures.

lambda_sweep.png (WBS S12 step 3) is already produced in S9 -- nothing
here duplicates it. This module covers steps 1-2: the 12 per-image
heat-map overlay rows and the 4x4 attention grid.

Deliberately split into two layers:
- select_heatmap_images(): picks WHICH images to render, from already-
  computed per-image CSVs (S11's output) -- no model, no GPU.
- plot_heatmap_row() / plot_attention_grid(): pure rendering from
  already-computed numpy arrays (image/mask/attention/CAM) -- no model,
  no GPU, no dataset loading. Fast and independently testable with
  synthetic arrays, unlike the actual data (real X-rays, real trained
  attention/CAM maps), which needs S8-S11's real Kaggle checkpoints.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def select_heatmap_images(
    a0_per_image_df: pd.DataFrame,
    a2_per_image_df: pd.DataFrame,
    class_names: List[str],
    n_per_class: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Selects up to n_per_class images per class (WBS section S12 step 1):
    (1) correctly classified by A2, (2) A0 wrong but A2 right -- if any
    exist for that class, else a second correctly-classified example
    instead (the WBS explicitly allows this case to be absent), and
    (3) a shortcut-artifact candidate -- the correctly-classified image
    with the LOWEST A2 ilar (i.e. attention mass furthest from the lungs
    while still getting the label right). This is a PROXY signal, not a
    verified detection of a visible artifact (text marker/device/border)
    -- visually confirm before using it in the paper, and swap the
    image_path by hand if it doesn't actually show one.

    Returns columns: class_name, reason, image_path. Up to
    3 * len(class_names) rows; fewer if a class has too few test images
    or no A0-wrong/A2-right case and no spare correct example either.
    """
    empty_result = pd.DataFrame(columns=["class_name", "reason", "image_path"])
    if a0_per_image_df.empty or a2_per_image_df.empty:
        return empty_result
    if "image_path" not in a0_per_image_df.columns or "image_path" not in a2_per_image_df.columns:
        return empty_result

    merged = a0_per_image_df.merge(a2_per_image_df, on="image_path", suffixes=("_a0", "_a2"), how="inner")
    if merged.empty:
        return empty_result

    rows = []
    for cname in class_names:
        cls_rows = merged[merged["true_label_a0"] == cname]
        if cls_rows.empty:
            continue
        picked: set = set()
        correct = cls_rows[cls_rows["pred_label_a2"] == cls_rows["true_label_a2"]]

        # Priority order matters: each step below claims its candidate
        # before the next, least-specific step runs. The artifact-proxy
        # criterion (lowest ilar) picks out one particular image; if the
        # generic "any correct example" pick ran first and happened to
        # land on that same image by chance, the artifact step would be
        # left with a worse (or no) candidate for no principled reason.
        # a0_wrong_a2_right is similarly specific (a real, not engineered,
        # case) and should not be pre-empted by the generic pick either.

        # 1) shortcut-artifact proxy: lowest ilar among A2's correct
        # predictions for this class. "ilar_a2" is the normal case (both
        # per-image DFs carry an ilar column, per build_per_image_predictions's
        # real schema, so the merge suffixes it); falls back to a bare
        # "ilar" column for callers that only attach it to the A2 side.
        ilar_col = "ilar_a2" if "ilar_a2" in cls_rows.columns else ("ilar" if "ilar" in cls_rows.columns else None)
        artifact_pick = None
        if ilar_col is not None and not correct.empty and correct[ilar_col].notna().any():
            ranked = correct[correct[ilar_col].notna()].sort_values(ilar_col, ascending=True)
            artifact_pick = ranked.iloc[0]
            picked.add(artifact_pick["image_path"])

        # 2) A0 wrong, A2 right (the WBS explicitly allows this to be
        # absent for a class -- falls back to a second correct example).
        a0_wrong_a2_right = cls_rows[
            (cls_rows["pred_label_a0"] != cls_rows["true_label_a0"])
            & (cls_rows["pred_label_a2"] == cls_rows["true_label_a2"])
            & (~cls_rows["image_path"].isin(picked))
        ]
        a0_right_pick = None
        a0_right_reason = "a0_wrong_a2_right"
        if not a0_wrong_a2_right.empty:
            a0_right_pick = a0_wrong_a2_right.sample(1, random_state=seed).iloc[0]
        else:
            fallback = correct[~correct["image_path"].isin(picked)]
            if not fallback.empty:
                a0_right_pick = fallback.sample(1, random_state=seed).iloc[0]
                a0_right_reason = "correctly_classified_fallback"
        if a0_right_pick is not None:
            picked.add(a0_right_pick["image_path"])

        # 3) any correctly-classified example -- least specific, runs last,
        # takes whatever's left.
        remaining_correct = correct[~correct["image_path"].isin(picked)]
        generic_pick = remaining_correct.sample(1, random_state=seed).iloc[0] if not remaining_correct.empty else None

        if generic_pick is not None:
            rows.append({"class_name": cname, "reason": "correctly_classified", "image_path": generic_pick["image_path"]})
        if a0_right_pick is not None:
            rows.append({"class_name": cname, "reason": a0_right_reason, "image_path": a0_right_pick["image_path"]})
        if artifact_pick is not None:
            rows.append({
                "class_name": cname, "reason": "shortcut_artifact_proxy_VERIFY_VISUALLY",
                "image_path": artifact_pick["image_path"],
            })

    return pd.DataFrame(rows, columns=["class_name", "reason", "image_path"])


def plot_heatmap_row(
    image: np.ndarray,
    mask: np.ndarray,
    a0_cam: np.ndarray,
    a2_attention: np.ndarray,
    a2_cam: np.ndarray,
    title: str,
    output_path,
    dpi: int = 150,
    alpha: float = 0.4,
) -> None:
    """One heat-map overlay row (WBS section S12 step 1):
    X-ray | ground-truth mask | A0 Grad-CAM | A2 attention | A2 Grad-CAM.

    `image` is [H,W] grayscale in [0,1]; `mask`/`a0_cam`/`a2_attention`/
    `a2_cam` are [H,W] in [0,1] (upsample attention/CAM maps to image
    resolution before calling this -- it does no resizing itself). A
    consistent 'jet' colormap at alpha~0.4, vmin=0/vmax=1 on every heat
    panel so the five images in a row (and across different calls) share
    one colour scale, per the WBS's explicit requirement.
    """
    import matplotlib.pyplot as plt

    panels = [
        ("X-ray", None),
        ("Ground-truth mask", mask),
        ("A0 Grad-CAM", a0_cam),
        ("A2 attention", a2_attention),
        ("A2 Grad-CAM", a2_cam),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (panel_title, overlay) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        if overlay is not None:
            ax.imshow(overlay, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        ax.set_title(panel_title, fontsize=11)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_attention_grid(
    images: List[np.ndarray],
    attentions: List[np.ndarray],
    labels: List[str],
    class_names: List[str],
    output_path,
    n_per_class: int = 4,
    dpi: int = 150,
    alpha: float = 0.5,
) -> None:
    """4x4 grid (default: 4 classes x 4 samples) of A2 attention maps
    overlaid on their X-rays -- the "does it look like lungs?" figure
    (WBS section S12 step 2), described there as potentially the single
    most persuasive image in the section.

    `images`/`attentions`/`labels` are parallel lists (same length, same
    order); grouped into rows by `labels` matching entries of
    `class_names`. A class with fewer than n_per_class available samples
    gets blank panels for the remainder rather than an error.
    """
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    by_class = {c: [] for c in class_names}
    for img, att, lbl in zip(images, attentions, labels):
        if lbl in by_class:
            by_class[lbl].append((img, att))

    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(4 * n_per_class, 4 * n_classes), squeeze=False)
    for row, cname in enumerate(class_names):
        samples = by_class[cname][:n_per_class]
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(samples):
                img, att = samples[col]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.imshow(att, cmap="jet", alpha=alpha, vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cname, fontsize=11)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
    fig.suptitle("A2 attention maps by class")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
