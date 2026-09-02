"""Shared mask-paired CXR data pipeline for the COVID-19 Radiography Dataset.

`JointTransform` / `CXRWithMaskDataset` / `stratified_split` / `compute_class_weights`
are copied verbatim from the AuxSeg (Candidate B) notebook
(kusal-notebooks/baseline-cnn-model-dnn-research-auxseg-updated.ipynb, cells 10-15)
so T18's data path matches the only other mask-carrying pipeline in the repo exactly
-- see Claude Working Files/T18_Lung_Region_Attention_WBS.md section 4.6.

`load_split_indices_from_manifest` is new: it makes `build_dataloaders` load the
committed split manifest (artifacts/splits/split_manifest_v1.csv) instead of
recomputing the split at training time, which is what docs/experiment_policy.md's
"Dataset Split" section actually requires (the manifest existing on disk isn't
enough on its own -- something has to read it).

Promoted to src/datasets/ rather than left as notebook-local code (as it currently is
in the AuxSeg notebook) because CLAUDE.md's own architecture section says reusable
dataset/split/preprocessing code belongs in src/ so multiple model owners' notebooks
can import it -- and T23 (applying the winning module to ResNet50) needs this exact
same pipeline again.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.utils import create_generator, seed_worker

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def only_images_folder(path: Union[str, Path]) -> bool:
    """ImageFolder file filter: only files inside a class's images/ subfolder.

    Matches AuxSeg's exact extension set (no .gif) -- T12/T14/T15/T16 include
    .gif too, but the dataset only ships .png files, so this is behaviorally
    identical either way; kept aligned with our chosen lineage (WBS section 2.5).
    """
    p = Path(path)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return p.parent.name.lower() == "images" and p.suffix.lower() in valid_exts


class JointTransform:
    """Applies identical spatial augmentation to an (image, mask) pair.

    Copied verbatim from the AuxSeg notebook. Spatial ops (resize/pad/crop/
    flip/rotate) are applied to both image and mask with the same random
    draw; brightness/contrast jitter is image-only. Mask uses NEAREST
    interpolation throughout (never bilinear, which would invent grey
    boundary values before the final >0.5 threshold) and is padded with 0
    ("not lung"), not reflected.
    """

    def __init__(self, img_size: int = 224, train: bool = True):
        self.img_size = img_size
        self.train = train
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        if self.train:
            image = TF.pad(image, padding=8, padding_mode="reflect")
            mask = TF.pad(mask, padding=8, fill=0)

            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.img_size, self.img_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

            image = self.color_jitter(image)

        image = TF.to_grayscale(image, num_output_channels=3)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = TF.to_grayscale(mask, num_output_channels=1)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask


class CXRWithMaskDataset(Dataset):
    """Returns (image_tensor, label_int, mask_tensor).

    Mask is located by swapping "images" for "masks" one level up from the
    file, matching the dataset's actual layout:
    COVID/images/COVID-1.png -> COVID/masks/COVID-1.png
    """

    def __init__(self, base_dataset: ImageFolder, indices: np.ndarray, transform: JointTransform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        image_path, label = self.base_dataset.samples[base_idx]
        image_path = Path(image_path)
        mask_path = image_path.parent.parent / "masks" / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for: {image_path}\nExpected mask: {mask_path}")

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        image, mask = self.transform(image, mask)

        return image, label, mask


def stratified_split(dataset: ImageFolder, seed: int = 42):
    """70/15/15 stratified split. Verbatim across every notebook in the repo.

    Prefer `load_split_indices_from_manifest` for actual training runs --
    this is kept for generating/regenerating the manifest itself, and as a
    fallback when no manifest is available yet.
    """
    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=targets, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=targets[temp_idx], random_state=seed)

    return train_idx, val_idx, test_idx


def load_split_indices_from_manifest(
    base_dataset: ImageFolder,
    manifest_path: Union[str, Path],
    data_root: Union[str, Path],
):
    """Reads the committed split manifest and returns (train_idx, val_idx, test_idx)
    aligned to base_dataset.samples order -- instead of recomputing the split.

    This is the actual mechanism docs/experiment_policy.md's "Dataset Split"
    section requires: the split is saved once and loaded thereafter, never
    regenerated per training run. `data_root` must be the same directory
    passed to `ImageFolder(root=...)` so the relative paths line up; the
    manifest stores paths relative to the dataset root precisely so it's
    portable across machines (local vs Kaggle) without editing it.

    Raises:
        KeyError: if any sample in `base_dataset` is missing from the manifest
            (a stale manifest, or a data_root mismatch) -- fails loudly rather
            than silently training on a partial/wrong split.
    """
    manifest = pd.read_csv(manifest_path)
    split_of = dict(zip(manifest["image_path"], manifest["split"]))

    data_root = Path(data_root)
    train_idx, val_idx, test_idx = [], [], []
    buckets = {"train": train_idx, "val": val_idx, "test": test_idx}

    for i, (path, _target) in enumerate(base_dataset.samples):
        rel = str(Path(path).relative_to(data_root))
        split = split_of.get(rel)
        if split is None:
            raise KeyError(f"{rel} not found in split manifest {manifest_path} (data_root={data_root})")
        buckets[split].append(i)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def compute_class_weights(train_targets: np.ndarray, num_classes: int) -> torch.Tensor:
    """w_c = N / (K * n_c). Verbatim across every notebook in the repo."""
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir: Union[str, Path],
    img_size: int,
    batch_size: int,
    seed: int,
    num_workers: int = 4,
    split_manifest_path: Optional[Union[str, Path]] = None,
):
    """Builds (train, val, test) DataLoaders of (image, label, mask) triples.

    If `split_manifest_path` is given, the split is loaded from the manifest
    (the policy-compliant path for real training runs). Otherwise falls back
    to `stratified_split(seed)` for callers that don't have a manifest yet.

    Adds reproducible worker seeding to the (shuffling) train_loader only --
    WBS section 4.8. val/test loaders don't shuffle and JointTransform(train=
    False) has no randomness, so a generator/worker_init_fn there would be a
    no-op.
    """
    base_dataset = ImageFolder(root=str(data_dir), is_valid_file=only_images_folder)

    if split_manifest_path is not None:
        train_idx, val_idx, test_idx = load_split_indices_from_manifest(base_dataset, split_manifest_path, data_dir)
    else:
        train_idx, val_idx, test_idx = stratified_split(base_dataset, seed=seed)

    train_tf = JointTransform(img_size=img_size, train=True)
    eval_tf = JointTransform(img_size=img_size, train=False)

    train_ds = CXRWithMaskDataset(base_dataset, train_idx, train_tf)
    val_ds = CXRWithMaskDataset(base_dataset, val_idx, eval_tf)
    test_ds = CXRWithMaskDataset(base_dataset, test_idx, eval_tf)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=create_generator(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    class_names = base_dataset.classes
    train_targets = np.array(base_dataset.targets)[train_idx]
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

    return train_loader, val_loader, test_loader, class_names, train_targets, datasets
