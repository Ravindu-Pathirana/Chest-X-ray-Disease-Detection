"""Matched, per-image lung/background perturbation measurements.

This is evaluation-only: no fitting, model selection or result interpretation.
The swap pairing and area-matched pixel choices are independent of model arms.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

from .counterfactual import _as_logits, _dataset_image_paths, perturb_background, perturb_lung


def image_key(path: str) -> str:
    """Dataset-relative class/images/name key, independent of mount point."""
    parts = str(path).replace("\\", "/").split("/")
    if len(parts) < 3 or parts[-2] != "images":
        raise ValueError(f"expected class/images/filename path: {path}")
    return "/".join(parts[-3:])


def build_swap_pairs(manifest: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Choose one different-true-class test donor per image, once for all arms."""
    required = {"image_path", "label", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"split manifest lacks {sorted(required - set(manifest.columns))}")
    test = manifest.loc[manifest["split"] == "test", ["image_path", "label"]].copy()
    test["image_path"] = test["image_path"].map(image_key)
    if test.empty or test["image_path"].duplicated().any() or test["label"].isna().any():
        raise ValueError("test split must be nonempty with unique paths and labels")
    test = test.sort_values("image_path").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    labels = test["label"].to_numpy()
    paths = test["image_path"].to_numpy()
    rows = []
    for path, label in zip(paths, labels):
        eligible = np.flatnonzero(labels != label)
        if not len(eligible):
            raise ValueError("different-class swap donors are unavailable")
        donor = int(rng.choice(eligible))
        rows.append({"image_path": path, "true_label": label,
                     "donor_image_path": str(paths[donor]), "donor_label": str(labels[donor]),
                     "pair_seed": seed})
    return pd.DataFrame(rows)


def area_matched_background_mask(mask: torch.Tensor, image_keys: Sequence[str],
                                 *, seed: int = 42) -> torch.Tensor:
    """Sample as many background pixels as lung pixels, identically per arm."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1 or len(mask) != len(image_keys):
        raise ValueError("mask must be [N,1,H,W] with one image key per row")
    output = torch.zeros_like(mask, dtype=torch.float32)
    for i, key in enumerate(image_keys):
        lung = (mask[i, 0] > 0.5).flatten().cpu().numpy()
        n_lung = int(lung.sum())
        background = np.flatnonzero(~lung)
        if n_lung == 0 or n_lung > len(background):
            raise ValueError(f"cannot area-match lung mask for {key}: {n_lung} lung, "
                             f"{len(background)} background pixels")
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        chosen = rng.choice(background, size=n_lung, replace=False)
        output[i, 0].view(-1)[torch.as_tensor(chosen, device=output.device)] = 1.0
    return output


def total_variation_distance(original: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
    """Per-image TV distance between class-probability vectors."""
    if original.ndim != 2 or original.shape != perturbed.shape:
        raise ValueError("probabilities must have matching [N,C] shapes")
    if not torch.isfinite(original).all() or not torch.isfinite(perturbed).all():
        raise ValueError("probabilities must be finite")
    if bool((original < 0).any() or (perturbed < 0).any()):
        raise ValueError("probabilities cannot be negative")
    if not torch.allclose(original.sum(1), torch.ones_like(original[:, 0]), atol=1e-5):
        raise ValueError("original rows must sum to one")
    if not torch.allclose(perturbed.sum(1), torch.ones_like(perturbed[:, 0]), atol=1e-5):
        raise ValueError("perturbed rows must sum to one")
    return 0.5 * (original - perturbed).abs().sum(dim=1)


@torch.no_grad()
def evaluate_xai_dependence(model, loader, class_names: Sequence[str],
                            swap_pairs: pd.DataFrame, device: torch.device,
                            *, arm: str, seed: int = 42, pairing_seed: int = 42,
                            blur_kernel: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run blur and swap for lung, background, and area-matched background.

    The deterministic loader must expose the manifest-backed test dataset.
    Swap donor tensors are loaded from that same test dataset (with its
    deterministic evaluation transform). Returns long-form per-image rows and
    a derived summary; callers save the per-image data first.
    """
    dataset = getattr(loader, "dataset", None)
    if not isinstance(getattr(loader, "sampler", None), SequentialSampler):
        raise ValueError("dependence loader must use sequential sampling for fixed image pairing")
    dataset_paths = _dataset_image_paths(dataset)
    if dataset_paths is None or len(dataset_paths) != len(dataset):
        raise ValueError("requires a manifest-backed dataset with image paths")
    keys = [image_key(p) for p in dataset_paths]
    if len(set(keys)) != len(keys):
        raise ValueError("test dataset has duplicate image keys")
    pair_cols = {"image_path", "true_label", "donor_image_path", "donor_label"}
    if not pair_cols.issubset(swap_pairs.columns):
        raise ValueError(f"swap pairs lack {sorted(pair_cols - set(swap_pairs.columns))}")
    if "pair_seed" in swap_pairs.columns and not swap_pairs["pair_seed"].eq(pairing_seed).all():
        raise ValueError("swap-pair seed differs from the declared pairing seed")
    pairs = swap_pairs.set_index("image_path", verify_integrity=True)
    if set(pairs.index) != set(keys):
        raise ValueError("swap-pair image set differs from test dataset")
    key_to_index = {key: i for i, key in enumerate(keys)}
    for key in keys:
        pair = pairs.loc[key]
        if pair["donor_image_path"] not in key_to_index or pair["true_label"] == pair["donor_label"]:
            raise ValueError(f"invalid different-class donor for {key}")
    if not class_names:
        raise ValueError("class_names cannot be empty")
    model.eval()
    rows = []
    offset = 0
    for images, labels, masks in loader:
        batch_keys = keys[offset:offset + len(images)]
        if len(batch_keys) != len(images):
            raise ValueError("loader yielded more images than its dataset")
        for j, key in enumerate(batch_keys):
            if str(pairs.loc[key, "true_label"]) != str(class_names[int(labels[j])]):
                raise ValueError(f"label mismatch for {key}; loader may be shuffled")
        donor_indices = [key_to_index[pairs.loc[key, "donor_image_path"]] for key in batch_keys]
        donors = torch.stack([dataset[i][0] for i in donor_indices]).to(device)
        images, masks = images.to(device), masks.to(device)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if masks.shape[-2:] != images.shape[-2:]:
            masks = F.interpolate(masks.float(), size=images.shape[-2:], mode="nearest")
        area = area_matched_background_mask(masks, batch_keys, seed=pairing_seed)
        original = torch.softmax(_as_logits(model(images)).float(), dim=1)
        original_pred = original.argmax(1)
        for mode in ("blur", "swap"):
            variants = {
                "lung": perturb_lung(images, masks, mode=mode, blur_kernel=blur_kernel,
                                     donor_images=donors),
                "background": perturb_background(images, masks, mode=mode,
                                                 blur_kernel=blur_kernel, donor_images=donors),
                "background_area_matched": perturb_background(
                    images, masks, mode=mode, blur_kernel=blur_kernel,
                    donor_images=donors, area_mask=area.to(device)),
            }
            for region, changed in variants.items():
                perturbed = torch.softmax(_as_logits(model(changed)).float(), dim=1)
                changed_pred = perturbed.argmax(1)
                tv = total_variation_distance(original, perturbed)
                drop = original.gather(1, original_pred[:, None]).squeeze(1) - \
                    perturbed.gather(1, original_pred[:, None]).squeeze(1)
                for j, key in enumerate(batch_keys):
                    donor_label = str(pairs.loc[key, "donor_label"])
                    rows.append({
                        "image_path": key, "true_label": str(class_names[int(labels[j])]),
                        "arm": arm, "seed": seed, "pairing_seed": pairing_seed,
                        "mode": mode, "region": region,
                        "original_prediction": str(class_names[int(original_pred[j])]),
                        "perturbed_prediction": str(class_names[int(changed_pred[j])]),
                        "original_probabilities": json.dumps(original[j].cpu().tolist()),
                        "perturbed_probabilities": json.dumps(perturbed[j].cpu().tolist()),
                        "dP": float(tv[j]), "original_class_probability_drop": float(drop[j]),
                        "prediction_flipped": bool(original_pred[j] != changed_pred[j]),
                        "donor_image_path": str(pairs.loc[key, "donor_image_path"]),
                        "donor_class": donor_label,
                        "flip_to_donor_class": bool(
                            changed_pred[j] != original_pred[j]
                            and class_names[int(changed_pred[j])] == donor_label),
                        "lung_pixels": int((masks[j] > 0.5).sum()),
                        "area_matched_background_pixels": int(area[j].sum()),
                    })
        offset += len(images)
    if offset != len(dataset):
        raise ValueError("loader did not cover the full test dataset")
    per_image = pd.DataFrame(rows)
    if per_image.empty:
        raise ValueError("test loader is empty")
    wide = per_image.pivot(index=["image_path", "mode"], columns="region", values="dP")
    wide["LRG"] = wide["lung"] - wide["background"]
    per_image = per_image.merge(wide[["LRG"]].reset_index(), on=["image_path", "mode"],
                                validate="many_to_one")
    summary_rows = []
    for mode, group in per_image.groupby("mode", sort=False):
        indexed = group.set_index(["image_path", "region"])
        summary_rows.append({
            "arm": arm, "seed": seed, "mode": mode, "n_images": group["image_path"].nunique(),
            "mean_dP_lung": indexed.xs("lung", level="region")["dP"].mean(),
            "mean_dP_background": indexed.xs("background", level="region")["dP"].mean(),
            "mean_dP_bg_area_matched": indexed.xs("background_area_matched", level="region")["dP"].mean(),
            "LRG": group["LRG"].mean(),
            "lung_flip_rate": indexed.xs("lung", level="region")["prediction_flipped"].mean(),
            "background_flip_rate": indexed.xs("background", level="region")["prediction_flipped"].mean(),
        })
    return per_image, pd.DataFrame(summary_rows)
