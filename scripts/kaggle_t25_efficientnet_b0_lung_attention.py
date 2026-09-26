"""T25 EfficientNet-B0 lung-attention A0-A6 experiment runner.

Selection stages train short candidates and inspect validation metrics only.
The final stage evaluates the requested arms on test and adds calibration,
Grad-CAM EIL, efficiency, and lung-preserving background counterfactuals.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd
import torch
import torch.nn as nn


LAMBDA_GRID = (0.0, 0.1, 0.3, 0.5, 1.0)
ARM_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stage", choices=("smoke", "lambda-att-sweep", "lambda-bg-sweep", "final"), required=True
    )
    parser.add_argument("--arms", nargs="+", choices=ARM_ORDER, default=list(ARM_ORDER))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--winner", choices=ARM_ORDER)
    parser.add_argument("--lambda-att", type=float)
    parser.add_argument("--lambda-bg", type=float)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cam-subset-size", type=int, default=1000)
    parser.add_argument("--skip-cam", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args(argv)


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        target = result
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return result


def arm_overrides(lambda_att: float, lambda_bg: float) -> Dict[str, Dict[str, Any]]:
    """Return the original team A0-A6 contract with resolved lambdas."""
    common = {"module.lambda_bg": 0.0}
    return {
        "A0": {**common, "module.use_attention": False, "module.attention": "lung", "module.gate_mode": "none", "module.lambda_att": 0.0},
        "A1": {**common, "module.use_attention": True, "module.attention": "lung", "module.gate_mode": "residual", "module.lambda_att": 0.0},
        "A2": {**common, "module.use_attention": True, "module.attention": "lung", "module.gate_mode": "residual", "module.lambda_att": lambda_att},
        "A3": {**common, "module.use_attention": True, "module.attention": "lung", "module.gate_mode": "multiply", "module.lambda_att": lambda_att},
        "A4": {**common, "module.use_attention": True, "module.attention": "lung", "module.gate_mode": "none", "module.lambda_att": lambda_att},
        "A5": {**common, "module.use_attention": True, "module.attention": "cbam", "module.gate_mode": "multiply", "module.lambda_att": 0.0},
        "A6": {"module.use_attention": True, "module.attention": "lung", "module.gate_mode": "residual", "module.lambda_att": lambda_att, "module.lambda_bg": lambda_bg},
    }


def select_largest_within_f1_tolerance(
    rows: Iterable[Mapping[str, Any]], value_key: str, tolerance: float
) -> Dict[str, Any]:
    """Official rule: compare with lambda=0 using validation only."""
    candidates = [dict(row) for row in rows]
    reference = next((row for row in candidates if float(row[value_key]) == 0.0), None)
    if reference is None:
        raise ValueError(f"selection rows must contain {value_key}=0.0")
    floor = float(reference["val_macro_f1"]) - tolerance
    eligible = [row for row in candidates if float(row["val_macro_f1"]) >= floor]
    if not eligible:
        raise RuntimeError("no candidate preserves validation macro-F1 within tolerance")
    eligible.sort(key=lambda row: (float(row[value_key]), float(row.get("val_ilar", float("-inf")))), reverse=True)
    return eligible[0]


def _load_selected(path: Path, key: str) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Run the corresponding validation sweep first; missing {path}")
    return float(json.loads(path.read_text(encoding="utf-8"))[key])


def _build_loaders(args: argparse.Namespace, cfg: Mapping[str, Any]):
    """Reset RNG, then rebuild the shuffled loader for every candidate/arm."""
    from src.datasets import build_dataloaders
    from src.utils import set_seed

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    manifest = args.repo_root / cfg["dataset"]["split_manifest"]
    if not manifest.is_file():
        raise FileNotFoundError(f"fixed split manifest not found: {manifest}")
    return build_dataloaders(
        data_dir=args.data_dir,
        img_size=cfg["dataset"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        seed=seed,
        num_workers=args.num_workers,
        split_manifest_path=manifest,
    )


def _criterion(train_targets, num_classes: int, device: torch.device):
    from src.datasets import compute_class_weights
    return nn.CrossEntropyLoss(weight=compute_class_weights(train_targets, num_classes).to(device))


def _build_model(cfg: Mapping[str, Any], pretrained: bool | None = None):
    from src.modules import build_model
    module = cfg["module"]
    return build_model(
        num_classes=cfg["model"]["num_classes"],
        use_attention=module["use_attention"],
        attention=module.get("attention", "lung"),
        gate_mode=module.get("gate_mode", "residual"),
        reduction=module.get("reduction", 16),
        backbone_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"] if pretrained is None else pretrained,
        drop_rate=cfg["model"].get("drop_rate", 0.0),
    )


def audit_model_contract(cfg: Mapping[str, Any]) -> Dict[str, float]:
    """Verify runtime channels and the reduction=16 parameter overhead."""
    import timm

    plain = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=cfg["model"]["num_classes"])
    runtime_features = int(plain.num_features)
    configured_features = int(cfg["module"]["in_channels"])
    if runtime_features != configured_features:
        raise RuntimeError(f"num_features mismatch: timm={runtime_features}, config={configured_features}")
    guided = _build_model(cfg, pretrained=False)
    plain_params = sum(p.numel() for p in plain.parameters())
    guided_params = sum(p.numel() for p in guided.parameters())
    overhead = (guided_params - plain_params) / plain_params
    if overhead > 0.03:
        raise RuntimeError(f"attention parameter overhead {overhead:.2%} exceeds 3%")
    return {"num_features": runtime_features, "plain_params": plain_params, "guided_params": guided_params, "parameter_overhead_fraction": overhead}


def run_selection_candidate(args, cfg, output_dir: Path, value_key: str, value: float, device):
    from src.modules import build_optimizer, build_scheduler, freeze_backbone, train_phase, unfreeze_final_blocks
    from src.modules.training import best_history_row

    candidate = copy.deepcopy(cfg)
    candidate["experiment"]["seed"] = args.seed
    candidate["training"]["phase1_epochs"] = candidate["selection"]["phase1_epochs"]
    candidate["training"]["phase2_epochs"] = candidate["selection"]["phase2_epochs"]
    candidate["training"]["patience"] = candidate["selection"]["patience"]
    candidate["module"][value_key] = value
    train_loader, val_loader, _test_loader, class_names, train_targets, _datasets = _build_loaders(args, candidate)
    criterion = _criterion(train_targets, len(class_names), device)
    model = _build_model(candidate).to(device)
    training = candidate["training"]
    module = candidate["module"]
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_backbone(model)
    opt1 = build_optimizer(model, training["optimizer"], training["phase1_lr"], training["weight_decay"])
    sched1 = build_scheduler(opt1, candidate["scheduler"]["name"], training["phase1_epochs"])
    model = train_phase(model, train_loader, val_loader, criterion, opt1, sched1, device,
                        training["phase1_epochs"], training["patience"], "phase1_frozen", output_dir,
                        lambda_att=module["lambda_att"], target_mode=module["target_mode"],
                        lambda_bg=module["lambda_bg"], wandb_enabled=False)
    unfreeze_final_blocks(model, training["unfreeze_blocks"])
    opt2 = build_optimizer(model, training["optimizer"], training["phase2_lr"], training["weight_decay"])
    sched2 = build_scheduler(opt2, candidate["scheduler"]["name"], training["phase2_epochs"])
    train_phase(model, train_loader, val_loader, criterion, opt2, sched2, device,
                training["phase2_epochs"], training["patience"], "phase2_finetune", output_dir,
                lambda_att=module["lambda_att"], target_mode=module["target_mode"],
                lambda_bg=module["lambda_bg"], wandb_enabled=False)
    best = best_history_row(output_dir / "phase2_finetune_history.json")
    return {
        value_key: value,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "val_macro_f1": best["val_f1"],
        "val_ilar": best["val_ilar"],
        "seed": args.seed,
    }


def run_sweep(args, cfg, output_root: Path, kind: str, device) -> None:
    value_key = "lambda_att" if kind == "att" else "lambda_bg"
    if kind == "bg":
        selected_att = args.lambda_att
        if selected_att is None:
            selected_att = _load_selected(output_root / "sweeps/lambda_att/selected_config.json", "lambda_att")
        cfg["module"]["lambda_att"] = selected_att
    sweep_dir = output_root / "sweeps" / ("lambda_att" if kind == "att" else "lambda_bg")
    rows = []
    for value in cfg["selection"][f"{value_key}_grid"]:
        rows.append(run_selection_candidate(args, cfg, sweep_dir / f"{value_key}_{value}", value_key, float(value), device))
    table = pd.DataFrame(rows)
    table.to_csv(sweep_dir / "tuning_summary.csv", index=False)
    tolerance = float(cfg["selection"]["macro_f1_tolerance"])
    selected = select_largest_within_f1_tolerance(rows, value_key, tolerance)
    payload = {
        **selected,
        "rule": f"largest {value_key} whose validation macro-F1 is within {tolerance:.3f} of {value_key}=0.0; tie-break higher validation ILAR",
        "reference": f"{value_key}=0.0",
        "test_metrics_used_for_selection": False,
    }
    (sweep_dir / "selected_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _load_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint)


def _write_validation_metrics(arm_dir: Path) -> None:
    from src.modules.training import best_history_row
    row = best_history_row(arm_dir / "phase2_finetune_history.json")
    (arm_dir / "validation_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


def _run_efficiency(model, checkpoint: Path, arm: str):
    from notebooks.efficiency import benchmark_model
    from src.modules import LogitsOnly
    return benchmark_model(LogitsOnly(model), "EfficientNet-B0", arm, checkpoint_path=checkpoint,
                           cpu_runs=20, gpu_runs=50)


def run_final(args, cfg, output_root: Path, device, smoke: bool = False) -> None:
    from src.modules import (
        LogitsOnly, build_per_image_predictions, calibration_report,
        evaluate_counterfactual_robustness, run_full_arm, stratified_cam_subset,
    )

    att_path = output_root / "sweeps/lambda_att/selected_config.json"
    bg_path = output_root / "sweeps/lambda_bg/selected_config.json"
    selected_att = args.lambda_att if args.lambda_att is not None else _load_selected(att_path, "lambda_att")
    selected_bg = args.lambda_bg if args.lambda_bg is not None else _load_selected(bg_path, "lambda_bg")
    overrides = arm_overrides(selected_att, selected_bg)
    if args.seed != 42 and (len(args.arms) != 2 or "A0" not in args.arms or not args.winner or args.winner not in args.arms):
        raise ValueError("multi-seed runs must contain only A0 and --winner, per the T25 protocol")
    if smoke:
        cfg["training"].update({"phase1_epochs": 1, "phase2_epochs": 1, "patience": 1})

    run_root = output_root if args.seed == 42 else output_root / "runs_multiseed" / f"seed_{args.seed}"
    results: Dict[str, Dict[str, Any]] = {}
    configs: Dict[str, Dict[str, Any]] = {}
    per_images: Dict[str, pd.DataFrame] = {}
    efficiency_rows = []
    cf_summaries = []
    cf_per_images = []

    for arm in args.arms:
        arm_cfg = apply_overrides(cfg, overrides[arm])
        arm_cfg["experiment"]["seed"] = args.seed
        configs[arm] = arm_cfg
        train_loader, val_loader, test_loader, class_names, train_targets, datasets = _build_loaders(args, arm_cfg)
        criterion = _criterion(train_targets, len(class_names), device)
        arm_dir = run_root / "runs" / arm
        results[arm] = run_full_arm(arm, arm_cfg, train_loader, val_loader, test_loader, class_names,
                                    criterion, device, arm_dir, backbone_name="efficientnet_b0",
                                    wandb_enabled=args.wandb)
        _write_validation_metrics(arm_dir)
        checkpoint = arm_dir / f"efficientnet_b0_{arm}.pt"
        model = _build_model(arm_cfg, pretrained=False)
        model.load_state_dict(_load_state(checkpoint), strict=True)
        model = model.to(device)

        cam_subset = None
        if not args.skip_cam and not smoke:
            subset_file = run_root / "cam_subset_indices.json"
            if subset_file.exists():
                cam_subset = json.loads(subset_file.read_text(encoding="utf-8"))
            else:
                cam_subset = stratified_cam_subset(datasets["test"], n=args.cam_subset_size, seed=args.seed)
                subset_file.parent.mkdir(parents=True, exist_ok=True)
                subset_file.write_text(json.dumps([int(i) for i in cam_subset], indent=2), encoding="utf-8")
        per_image = build_per_image_predictions(model, datasets["test"], class_names, device,
                                                 cam_subset=cam_subset, batch_size=arm_cfg["training"]["batch_size"])
        per_image.to_csv(arm_dir / "per_image_predictions.csv", index=False)
        per_images[arm] = per_image

        if not args.skip_calibration and not smoke:
            calibration_report(LogitsOnly(model), val_loader, test_loader, class_names, device,
                               arm_dir / "calibration", model_name=f"EfficientNet-B0-{arm}")
        if not args.skip_counterfactual and not smoke:
            per_cf = arm_dir / "counterfactual_per_image.csv"
            cf_summaries.append(evaluate_counterfactual_robustness(
                model, test_loader, device, seed=args.seed, arm_name=arm, per_image_csv=per_cf
            ))
            cf_per_images.append(pd.read_csv(per_cf))
        if not args.skip_efficiency and not smoke:
            efficiency_rows.append(_run_efficiency(model, checkpoint, arm))
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    efficiency_by_arm = {row["module"]: row for row in efficiency_rows}
    comparison_rows = []
    reference_f1 = results["A0"]["classification_report"]["macro avg"]["f1-score"] if "A0" in results else float("nan")
    reference_eil = per_images["A0"]["eil_post"].mean() if "A0" in per_images else float("nan")
    for arm in args.arms:
        report = results[arm]["classification_report"]
        eff = efficiency_by_arm.get(arm, {})
        comparison_rows.append({
            "arm": arm,
            "gate_mode": configs[arm]["module"]["gate_mode"],
            "lambda_att": configs[arm]["module"]["lambda_att"],
            "lambda_bg": configs[arm]["module"]["lambda_bg"],
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_auc": results[arm].get("roc_auc_macro"),
            "attention_ilar": results[arm].get("attention_ilar"),
            "attention_dice": results[arm].get("attention_dice"),
            "attention_iou": results[arm].get("attention_iou"),
            "eil_pre": per_images[arm]["eil_pre"].mean(),
            "eil_post": per_images[arm]["eil_post"].mean(),
            "delta_f1_vs_A0": report["macro avg"]["f1-score"] - reference_f1,
            "delta_eil_vs_A0": per_images[arm]["eil_post"].mean() - reference_eil,
            "params": eff.get("params_total"),
            "trainable_params": eff.get("params_trainable"),
            "gflops": eff.get("gflops"),
            "state_dict_size_mb": eff.get("state_dict_size_mb"),
            "checkpoint_size_mb": eff.get("checkpoint_size_mb"),
            "cpu_latency_ms": eff.get("cpu_latency_ms"),
            "cpu_throughput_img_s": eff.get("cpu_throughput_img_s"),
            "gpu_latency_ms": eff.get("gpu_latency_ms"),
            "gpu_throughput_img_s": eff.get("gpu_throughput_img_s"),
        })
    pd.DataFrame(comparison_rows).to_csv(run_root / "T25_comparison_table.csv", index=False)
    if efficiency_rows:
        pd.DataFrame(efficiency_rows).to_csv(run_root / "T25_efficiency.csv", index=False)
    if cf_summaries:
        pd.concat(cf_summaries, ignore_index=True).to_csv(run_root / "counterfactual_summary.csv", index=False)
        pd.concat(cf_per_images, ignore_index=True).to_csv(run_root / "counterfactual_per_image.csv", index=False)
    (run_root / "run_manifest.json").write_text(json.dumps({
        "seed": args.seed, "arms": args.arms, "winner": args.winner,
        "lambda_att": selected_att, "lambda_bg": selected_bg,
        "split_manifest": cfg["dataset"]["split_manifest"], "test_used_for_selection": False,
    }, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(args.repo_root))
    from src.utils import load_config

    config_path = args.config or args.repo_root / "configs/efficientnet_b0_lung_attention.yaml"
    cfg = load_config(config_path)
    cfg["experiment"]["seed"] = args.seed
    output_root = args.output_dir or args.repo_root / "artifacts/T25_efficientnet_b0_lung_attention"
    output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_model_contract(cfg)
    (output_root / "model_contract_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "lambda-att-sweep":
        cfg["module"].update({"use_attention": True, "attention": "lung", "gate_mode": "residual", "lambda_bg": 0.0})
        run_sweep(args, cfg, output_root, "att", device)
    elif args.stage == "lambda-bg-sweep":
        cfg["module"].update({"use_attention": True, "attention": "lung", "gate_mode": "residual"})
        run_sweep(args, cfg, output_root, "bg", device)
    else:
        run_final(args, cfg, output_root, device, smoke=args.stage == "smoke")
        archive = shutil.make_archive(str(output_root.parent / f"T25_results_seed{args.seed}"), "zip", root_dir=output_root)
        print(f"Results archive: {archive}")


if __name__ == "__main__":
    main()
