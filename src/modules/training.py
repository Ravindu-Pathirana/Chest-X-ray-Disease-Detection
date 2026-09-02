"""Training loop for T18 (and reusable by T23 on ResNet50): run_epoch,
train_phase, evaluate.

Adapted from the AuxSeg notebook's run_epoch/train_phase/evaluate
(kusal-notebooks/baseline-cnn-model-dnn-research-auxseg-updated.ipynb,
cells 23-25), which already handle the (image, label, mask) batch
contract, AMP + GradScaler, early stopping, and history JSON. Per WBS
section 8 (S6), only the loss block and the logged metric names change:
- classification-only loss -> compute_total_loss (CE + lambda*attention
  guidance) from lung_attention.py
- segmentation Dice/IoU -> attention ilar/dice/iou from attention_metrics.py
- adds per-epoch validation precision/recall/F1/AUROC (mandatory per
  docs/experiment_policy.md's "Required Information"), which the AuxSeg
  loop this is based on does not compute.

One function handles all T18 arms via the model's own use_attention/
gate_mode flags and this module's lambda_att -- no per-arm branching or
copy-pasted training code (WBS S6 DoD).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from src.utils import (  # noqa: F401  (initialize_wandb/generate_run_name re-exported for caller convenience)
    finish_run,
    generate_run_name,
    initialize_wandb,
    log_metrics,
    log_summary_metrics,
    set_seed,
)

from .attention_metrics import attention_dice, attention_iou, ilar
from .lung_attention import build_model, compute_total_loss, freeze_backbone, unfreeze_final_blocks


def build_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Matches every other team notebook's build_optimizer (T12/T14/T15/T16/T19) --
    only trainable (requires_grad=True) parameters are passed, so this is safe
    to call in either phase 1 (frozen backbone) or phase 2 (partially unfrozen)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, name: str, epochs: int):
    """Matches every other team notebook's build_scheduler. Returns None for
    "none" -- train_phase already handles a None scheduler as a no-op."""
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def best_history_row(history_file) -> Dict[str, Any]:
    """T16's exact helper (kusal-notebooks/efficientnet-b0-baseline-model.ipynb,
    cell 25) -- the epoch with the lowest val_loss, i.e. the one train_phase
    actually kept as best_state. Reused verbatim per WBS section S9."""
    with open(history_file, "r") as f:
        history = json.load(f)
    return min(history, key=lambda x: x["val_loss"])


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    lambda_att: float = 0.5,
    target_mode: str = "soft",
    desc: str = "",
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, Any]:
    """One pass over `loader`. Returns loss/cls_loss/att_loss/accuracy/ilar
    always; precision/recall/f1/auroc additionally when train=False (WBS
    S6 -- validation-only, matching docs/experiment_policy.md's per-epoch
    Required Information; computing these on the full training set every
    epoch would be needless overhead the policy doesn't ask for).

    `ilar` is NaN when the model has no attention map (arm A0) rather than
    0 -- a 0 would silently read as "attention entirely on background" and
    corrupt any downstream average across arms (WBS section 7.1's
    NaN-not-zero convention, applied here too).
    """
    model.train() if train else model.eval()

    total_loss = total_cls_loss = total_att_loss = 0.0
    correct = total = 0
    ilar_sum = 0.0
    ilar_count = 0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    non_blocking = device.type == "cuda"
    context = torch.enable_grad() if train else torch.no_grad()
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    with context:
        for images, labels, masks in progress:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking).long()
            masks = masks.to(device, non_blocking=non_blocking).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                class_logits, att, att_logits = model(images)
                loss, cls_loss, att_loss = compute_total_loss(
                    class_logits, att_logits, labels, masks, criterion,
                    lambda_att=lambda_att, target_mode=target_mode,
                )

            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            total_att_loss += att_loss.item() * batch_size

            probs = torch.softmax(class_logits, dim=1).detach()
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            if not train:
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            if att is not None:
                batch_ilar = ilar(att.detach().float(), masks)
                ilar_sum += batch_ilar.sum().item()
                ilar_count += batch_ilar.numel()

            progress.set_postfix(
                loss=f"{total_loss / total:.4f}",
                cls=f"{total_cls_loss / total:.4f}",
                att=f"{total_att_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

    metrics: Dict[str, Any] = {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "att_loss": total_att_loss / total,
        "accuracy": correct / total,
        "ilar": (ilar_sum / ilar_count) if ilar_count > 0 else float("nan"),
    }

    if not train:
        labels_arr = np.array(all_labels)
        preds_arr = np.array(all_preds)
        probs_arr = np.array(all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, preds_arr, average="macro", zero_division=0
        )
        try:
            auroc = roc_auc_score(labels_arr, probs_arr, multi_class="ovr", average="macro")
        except ValueError:
            auroc = None
        metrics.update({"precision": precision, "recall": recall, "f1": f1, "auroc": auroc})

    return metrics


def train_phase(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    phase_name: str,
    output_dir,
    lambda_att: float = 0.5,
    target_mode: str = "soft",
    wandb_enabled: bool = True,
) -> torch.nn.Module:
    """Trains for up to `epochs`, early-stopping on val_loss (the total
    objective -- WBS section S6 notes this isn't comparable *across* arms
    since A2's total includes lambda*att_loss; val_cls_loss is the
    cross-arm-comparable one, logged alongside).

    best_val_f1/best_val_auroc in the W&B summary are the values AT the
    checkpoint actually kept (the val_loss-best epoch), not the best value
    seen at any epoch -- reporting a metric from an epoch whose weights
    weren't saved would be misleading.

    wandb_enabled=False skips every W&B call entirely (no `wandb.run`
    needed, `wandb` doesn't even need to be installed) -- use this for
    smoke tests / unit tests of the loop itself; real runs should leave it
    True and call initialize_wandb() first (mode="disabled" still counts
    as "enabled" here -- it sets a real, no-op wandb.run).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_f1: Optional[float] = None
    best_val_auroc: Optional[float] = None
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        train_desc = f"{phase_name} epoch {epoch}/{epochs} train"
        val_desc = f"{phase_name} epoch {epoch}/{epochs} val"

        tr = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            lambda_att=lambda_att, target_mode=target_mode, desc=train_desc, scaler=scaler,
        )
        va = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False,
            lambda_att=lambda_att, target_mode=target_mode, desc=val_desc,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va["loss"])
            else:
                scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"], "train_cls_loss": tr["cls_loss"], "train_att_loss": tr["att_loss"],
            "train_acc": tr["accuracy"], "train_ilar": tr["ilar"],
            "val_loss": va["loss"], "val_cls_loss": va["cls_loss"], "val_att_loss": va["att_loss"],
            "val_acc": va["accuracy"], "val_ilar": va["ilar"],
            "val_precision": va["precision"], "val_recall": va["recall"],
            "val_f1": va["f1"], "val_auroc": va["auroc"],
        }
        history.append(row)

        print(
            f"[{phase_name}] epoch {epoch}/{epochs} "
            f"train_loss={tr['loss']:.4f} (cls={tr['cls_loss']:.4f}, att={tr['att_loss']:.4f}) "
            f"train_acc={tr['accuracy']:.4f} | "
            f"val_loss={va['loss']:.4f} (cls={va['cls_loss']:.4f}, att={va['att_loss']:.4f}) "
            f"val_acc={va['accuracy']:.4f} val_f1={va['f1']:.4f} val_ilar={va['ilar']:.4f}"
        )

        if wandb_enabled:
            log_metrics({
                "epoch": epoch,
                "train_loss": tr["loss"], "val_loss": va["loss"],
                "train_cls_loss": tr["cls_loss"], "val_cls_loss": va["cls_loss"],
                "train_att_loss": tr["att_loss"], "val_att_loss": va["att_loss"],
                "val_accuracy": va["accuracy"], "val_precision": va["precision"],
                "val_recall": va["recall"], "val_f1": va["f1"], "val_auroc": va["auroc"],
                "val_ilar": va["ilar"] if not np.isnan(va["ilar"]) else None,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            best_val_f1 = va["f1"]
            best_val_auroc = va["auroc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{phase_name}] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)

    with open(output_dir / f"{phase_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    if wandb_enabled:
        log_summary_metrics({
            f"{phase_name}_best_epoch": best_epoch,
            f"{phase_name}_best_val_loss": best_val_loss,
            f"{phase_name}_best_val_f1": best_val_f1,
            f"{phase_name}_best_val_auroc": best_val_auroc,
        })

    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, class_names: List[str], device: torch.device, output_dir, evaluation_type: str) -> Dict[str, Any]:
    """Full-pass evaluation. Writes {evaluation_type}_test_results.json,
    _confusion_matrix.csv, _summary_metrics.csv.

    test_results.json keeps AuxSeg's key set unchanged (classification_report,
    confusion_matrix, roc_auc_macro, roc_auc_per_class, class_names) and adds
    keys rather than renaming any (WBS section 7.1) -- Member 5's tooling
    reads the existing ones.

    summary_metrics.csv writes the 7-column classification superset plus
    5 module columns (WBS 7.1) so it concatenates with T16's and AuxSeg's
    CSVs via pd.concat(..., join="outer"). cam_eil_post/cam_eil_pre are
    written as NaN here -- Grad-CAM evaluation is a separate, heavier pass
    (S11), not part of this function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_ilar: List[float] = []
    all_dice: List[float] = []
    all_iou: List[float] = []

    non_blocking = device.type == "cuda"

    for images, labels, masks in loader:
        images = images.to(device, non_blocking=non_blocking)
        masks = masks.to(device, non_blocking=non_blocking).float()
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        class_logits, att, _att_logits = model(images)
        probs = torch.softmax(class_logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

        if att is not None:
            att = att.float()
            all_ilar.extend(ilar(att, masks).cpu().numpy())
            all_dice.extend(attention_dice(att, masks).cpu().numpy())
            all_iou.extend(attention_iou(att, masks).cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        auc_per_class = roc_auc_score(all_labels, all_probs, multi_class="ovr", average=None)
    except ValueError:
        auc_macro, auc_per_class = None, None

    mean_ilar = float(np.mean(all_ilar)) if all_ilar else None
    mean_dice = float(np.mean(all_dice)) if all_dice else None
    mean_iou = float(np.mean(all_iou)) if all_iou else None

    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "roc_auc_macro": auc_macro,
        "roc_auc_per_class": auc_per_class.tolist() if auc_per_class is not None else None,
        "class_names": class_names,
        "attention_ilar": mean_ilar,
        "attention_dice": mean_dice,
        "attention_iou": mean_iou,
    }

    with open(output_dir / f"{evaluation_type}_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Test set performance ===")
    print(f"Test accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    print("Confusion matrix:\n", cm)
    if auc_macro is not None:
        print(f"Macro ROC-AUC: {auc_macro:.4f}")
    if mean_ilar is not None:
        print(f"Attention ILAR: {mean_ilar:.4f} | Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / f"{evaluation_type}_confusion_matrix.csv")

    summary = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_precision": report["macro avg"]["precision"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "weighted_recall": report["weighted avg"]["recall"],
        "roc_auc_macro": auc_macro,
        "attention_ilar": mean_ilar if mean_ilar is not None else float("nan"),
        "attention_dice": mean_dice if mean_dice is not None else float("nan"),
        "attention_iou": mean_iou if mean_iou is not None else float("nan"),
        "cam_eil_post": float("nan"),
        "cam_eil_pre": float("nan"),
    }
    pd.DataFrame([summary]).to_csv(output_dir / f"{evaluation_type}_summary_metrics.csv", index=False)

    return results


def run_full_arm(
    arm_name: str,
    arm_cfg: Dict[str, Any],
    train_loader,
    val_loader,
    test_loader,
    class_names: List[str],
    criterion,
    device: torch.device,
    output_dir,
    backbone_name: str = "densenet121",
    wandb_enabled: bool = True,
) -> Dict[str, Any]:
    """Runs one T18 arm's complete two-phase schedule: build model -> freeze
    -> phase1 train_phase -> evaluate -> unfreeze -> phase2 train_phase ->
    evaluate -> save checkpoint. Returns phase2's evaluate() results dict.

    `arm_cfg` must be a fully-resolved config (e.g. via merge_overrides) --
    every model-building and hyperparameter value is read from arm_cfg
    itself (arm_cfg["module"], arm_cfg["training"], arm_cfg["scheduler"]),
    not passed as separate arguments. This is what makes ONE function
    handle every arm (A0-A5) with no per-arm branching or copy-pasted
    training code -- exactly the WBS section S6 DoD, applied here to S8/S10
    too, since without this a 4th/5th arm would otherwise mean copy-pasting
    S8's ~40-line cell body 4-5 times with small, easy-to-typo edits.

    Re-seeds fresh at the start (WBS section S10 pitfall: continuing a
    session without re-seeding means arm N's init depends on how many
    random numbers arm N-1 consumed -- a subtle, real confound between arms).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = arm_cfg["experiment"]["seed"]
    m = arm_cfg["module"]
    t = arm_cfg["training"]

    set_seed(seed)
    if wandb_enabled:
        initialize_wandb(arm_cfg, run_name=generate_run_name(backbone_name, f"T18-{arm_name}", seed))

    try:
        model = build_model(
            num_classes=arm_cfg["model"]["num_classes"],
            use_attention=m["use_attention"],
            attention=m.get("attention", "lung"),
            gate_mode=m.get("gate_mode", "residual"),
            reduction=m.get("reduction", 8),
            backbone_name=backbone_name,
            pretrained=arm_cfg["model"]["pretrained"],
        ).to(device)
        lambda_att = m["lambda_att"]

        freeze_backbone(model)
        opt1 = build_optimizer(model, t["optimizer"], t["phase1_lr"], t["weight_decay"])
        sched1 = build_scheduler(opt1, arm_cfg["scheduler"]["name"], t["phase1_epochs"])
        model = train_phase(
            model, train_loader, val_loader, criterion, opt1, sched1, device,
            epochs=t["phase1_epochs"], patience=t["patience"], phase_name="phase1_frozen",
            output_dir=output_dir, lambda_att=lambda_att, wandb_enabled=wandb_enabled,
        )
        evaluate(model, test_loader, class_names, device, output_dir, "phase1_frozen")

        unfreeze_final_blocks(model, t["unfreeze_blocks"])
        opt2 = build_optimizer(model, t["optimizer"], t["phase2_lr"], t["weight_decay"])
        sched2 = build_scheduler(opt2, arm_cfg["scheduler"]["name"], t["phase2_epochs"])
        model = train_phase(
            model, train_loader, val_loader, criterion, opt2, sched2, device,
            epochs=t["phase2_epochs"], patience=t["patience"], phase_name="phase2_finetune",
            output_dir=output_dir, lambda_att=lambda_att, wandb_enabled=wandb_enabled,
        )
        results = evaluate(model, test_loader, class_names, device, output_dir, "phase2_finetune")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "config": arm_cfg,
                "arm": arm_name,
                "seed": seed,
            },
            output_dir / f"{backbone_name}_{arm_name}.pt",
        )
    finally:
        if wandb_enabled:
            finish_run()

    return results
