"""Shortcut-suppression modules shared across model owners.

Currently: T18's Lung-Region Attention Module (Candidate A), built on
DenseNet121 and designed to be backbone-agnostic for reuse in T23
(applying the winning module to ResNet50). Import from here, not from
`src.modules.lung_attention` directly, matching this repo's convention
for `src.utils` and `src.datasets` (see CLAUDE.md).
"""

from .attention_metrics import (
    attention_dice,
    attention_entropy,
    attention_iou,
    background_attention,
    energy_inside_lung,
    ilar,
)
from .comparison import (
    build_comparison_table,
    build_per_image_predictions,
    check_acceptance_criteria,
    stratified_cam_subset,
)
from .efficiency_check import check_module_efficiency
from .figures import plot_attention_grid, plot_heatmap_row, select_heatmap_images
from .gradcam import cam_for, get_taps
from .lung_attention import (
    CBAMSpatialAttention,
    DenseNetLungAttention,
    LogitsOnly,
    LungRegionAttention,
    attention_guidance_loss,
    build_model,
    compute_total_loss,
    freeze_backbone,
    print_trainable_parameters,
    unfreeze_final_blocks,
)
from .training import best_history_row, build_optimizer, build_scheduler, evaluate, run_epoch, run_full_arm, train_phase

__all__ = [
    "LungRegionAttention",
    "CBAMSpatialAttention",
    "DenseNetLungAttention",
    "LogitsOnly",
    "build_model",
    "attention_guidance_loss",
    "compute_total_loss",
    "freeze_backbone",
    "unfreeze_final_blocks",
    "print_trainable_parameters",
    "ilar",
    "attention_iou",
    "attention_dice",
    "attention_entropy",
    "background_attention",
    "energy_inside_lung",
    "cam_for",
    "get_taps",
    "run_epoch",
    "train_phase",
    "evaluate",
    "build_optimizer",
    "build_scheduler",
    "best_history_row",
    "run_full_arm",
    "stratified_cam_subset",
    "build_per_image_predictions",
    "build_comparison_table",
    "check_acceptance_criteria",
    "select_heatmap_images",
    "plot_heatmap_row",
    "plot_attention_grid",
    "check_module_efficiency",
]
