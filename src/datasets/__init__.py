"""Shared, mask-aware CXR data pipeline (COVID-19 Radiography Dataset).

Used by T18 (Lung-Region Attention) and reusable by T23 (applying the
winning shortcut-suppression module to ResNet50). Import from here, not
from `src.datasets.covid_cxr` directly, matching this repo's convention for
`src.utils` (see CLAUDE.md).
"""

from .covid_cxr import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CXRWithMaskDataset,
    JointTransform,
    build_dataloaders,
    compute_class_weights,
    load_split_indices_from_manifest,
    only_images_folder,
    stratified_split,
)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "JointTransform",
    "CXRWithMaskDataset",
    "only_images_folder",
    "stratified_split",
    "load_split_indices_from_manifest",
    "compute_class_weights",
    "build_dataloaders",
]
