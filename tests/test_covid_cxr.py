"""Tests for src/datasets/covid_cxr.py -- the shared mask-paired CXR data pipeline.

Uses synthetic fixtures only (no dependency on the real ~21k-image dataset),
so these run identically in CI and locally.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.datasets import (
    CXRWithMaskDataset,
    JointTransform,
    compute_class_weights,
    load_split_indices_from_manifest,
    only_images_folder,
)


# ---------------------------------------------------------------------------
# only_images_folder
# ---------------------------------------------------------------------------

def test_only_images_folder_accepts_images_subfolder():
    assert only_images_folder("/data/COVID/images/COVID-1.png")


def test_only_images_folder_rejects_masks_subfolder():
    assert not only_images_folder("/data/COVID/masks/COVID-1.png")


def test_only_images_folder_rejects_wrong_extension():
    assert not only_images_folder("/data/COVID/images/COVID.metadata.xlsx")


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------

def test_compute_class_weights_hand_computable():
    # counts: [10, 10, 20, 60] -> N=100, K=4 -> w_c = 100 / (4 * n_c)
    targets = np.array([0] * 10 + [1] * 10 + [2] * 20 + [3] * 60)
    w = compute_class_weights(targets, num_classes=4)
    expected = torch.tensor([100 / (4 * 10), 100 / (4 * 10), 100 / (4 * 20), 100 / (4 * 60)])
    torch.testing.assert_close(w, expected)


def test_compute_class_weights_missing_class_no_div_by_zero():
    targets = np.array([0, 0, 0, 1])  # class 2, 3 absent
    w = compute_class_weights(targets, num_classes=4)
    assert torch.isfinite(w).all()


# ---------------------------------------------------------------------------
# JointTransform
# ---------------------------------------------------------------------------

def _synthetic_image_and_mask(size: int = 64):
    """Left half bright, right half dark; mask marks the bright half."""
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[:, : size // 2] = 220
    arr[:, size // 2 :] = 30
    image = Image.fromarray(arr).convert("L")

    mask_arr = np.zeros((size, size), dtype=np.uint8)
    mask_arr[:, : size // 2] = 255
    mask = Image.fromarray(mask_arr).convert("L")
    return image, mask


def test_joint_transform_train_output_contract():
    image, mask = _synthetic_image_and_mask()
    random.seed(0)
    tf = JointTransform(img_size=224, train=True)
    img_t, mask_t = tf(image, mask)

    assert img_t.shape == (3, 224, 224)
    assert mask_t.shape == (1, 224, 224)
    assert img_t.dtype == torch.float32
    assert set(torch.unique(mask_t).tolist()) <= {0.0, 1.0}


def test_joint_transform_eval_has_no_randomness():
    image, mask = _synthetic_image_and_mask()
    tf = JointTransform(img_size=224, train=False)

    # No random.seed() reset between calls -- if eval mode used any
    # randomness, these would differ.
    img_a, mask_a = tf(image, mask)
    img_b, mask_b = tf(image, mask)

    torch.testing.assert_close(img_a, img_b)
    torch.testing.assert_close(mask_a, mask_b)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_joint_transform_keeps_image_and_mask_co_registered(seed):
    """The same random draw (rotation/crop/flip) must apply to both image and
    mask. If a future refactor accidentally computed separate random params
    for each, the bright/dark halves of this synthetic pair would decorrelate
    from the mask -- this test would start failing.
    """
    image, mask = _synthetic_image_and_mask()
    random.seed(seed)
    tf = JointTransform(img_size=224, train=True)
    img_t, mask_t = tf(image, mask)

    inside = img_t[:, mask_t[0] > 0.5]
    outside = img_t[:, mask_t[0] <= 0.5]
    assert inside.numel() > 0 and outside.numel() > 0
    assert inside.mean().item() > outside.mean().item()


def test_joint_transform_mask_uses_nearest_not_bilinear():
    # A mask resized with NEAREST from a small binary source stays binary
    # (module-level TF.to_tensor + threshold already enforces this, but this
    # guards against someone swapping the mask resize to BILINEAR upstream).
    image, mask = _synthetic_image_and_mask(size=17)  # deliberately not a power of 2
    random.seed(0)
    tf = JointTransform(img_size=224, train=False)
    _img_t, mask_t = tf(image, mask)
    assert set(torch.unique(mask_t).tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# CXRWithMaskDataset
# ---------------------------------------------------------------------------

def _write_tiny_dataset(root: Path, classes=("COVID", "Normal")):
    for cls in classes:
        (root / cls / "images").mkdir(parents=True, exist_ok=True)
        (root / cls / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(2):
            img, mask = _synthetic_image_and_mask()
            img.save(root / cls / "images" / f"{cls}-{i}.png")
            mask.save(root / cls / "masks" / f"{cls}-{i}.png")


def test_cxr_with_mask_dataset_returns_triple(tmp_path):
    _write_tiny_dataset(tmp_path)
    base = ImageFolder(root=str(tmp_path), is_valid_file=only_images_folder)
    ds = CXRWithMaskDataset(base, np.arange(len(base)), JointTransform(img_size=224, train=False))

    image, label, mask = ds[0]
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert mask.shape == (1, 224, 224)


def test_cxr_with_mask_dataset_raises_on_missing_mask(tmp_path):
    _write_tiny_dataset(tmp_path)
    # Delete one mask to simulate a stale/incomplete dataset.
    missing = tmp_path / "COVID" / "masks" / "COVID-0.png"
    missing.unlink()

    base = ImageFolder(root=str(tmp_path), is_valid_file=only_images_folder)
    ds = CXRWithMaskDataset(base, np.arange(len(base)), JointTransform(img_size=224, train=False))

    idx = next(i for i, (p, _t) in enumerate(base.samples) if Path(p).name == "COVID-0.png")
    with pytest.raises(FileNotFoundError):
        ds[idx]  # indices=np.arange(len(base)), so dataset index k -> base_idx k


# ---------------------------------------------------------------------------
# load_split_indices_from_manifest
# ---------------------------------------------------------------------------

def test_load_split_indices_from_manifest(tmp_path):
    _write_tiny_dataset(tmp_path)  # 2 classes x 2 images = 4 samples
    base = ImageFolder(root=str(tmp_path), is_valid_file=only_images_folder)

    rows = []
    for path, target in base.samples:
        rel = str(Path(path).relative_to(tmp_path))
        split = "train" if "COVID-0" in rel or "Normal-0" in rel else "val"
        rows.append({"image_path": rel, "patient_id": "", "split": split, "label": base.classes[target]})
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    train_idx, val_idx, test_idx = load_split_indices_from_manifest(base, manifest_path, tmp_path)

    assert len(train_idx) == 2
    assert len(val_idx) == 2
    assert len(test_idx) == 0
    # Every sample assigned to exactly one split, none dropped.
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(len(base)))


def test_load_split_indices_from_manifest_raises_on_missing_sample(tmp_path):
    _write_tiny_dataset(tmp_path)
    base = ImageFolder(root=str(tmp_path), is_valid_file=only_images_folder)

    # Manifest only covers one sample -- the rest are "missing".
    path, target = base.samples[0]
    rel = str(Path(path).relative_to(tmp_path))
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"image_path": rel, "patient_id": "", "split": "train", "label": base.classes[target]}]).to_csv(
        manifest_path, index=False
    )

    with pytest.raises(KeyError):
        load_split_indices_from_manifest(base, manifest_path, tmp_path)
