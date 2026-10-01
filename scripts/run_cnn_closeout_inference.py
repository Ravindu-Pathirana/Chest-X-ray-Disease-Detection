"""Run checkpoint-only CNN closeout evaluations without retraining.

Example (repeat with each backbone/arm/seed checkpoint):
python scripts/run_cnn_closeout_inference.py --backbone densenet121 \
    --arm A2_full --seed 42 --checkpoint /weights/densenet_a2_seed42.pt \
    --data-dir /data/COVID-19_Radiography_Dataset --tasks all

The data directory must contain class/images and class/masks subdirectories.
Never use test labels to select a checkpoint or fit calibration temperature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPECS = {
    "densenet121": {
        "name": "DenseNet121", "config": "configs/densenet121_lung_attention.yaml",
        "arms": ("A0_vanilla", "A2_full"), "artifact": "T18_lung_attention",
    },
    "resnet50": {
        "name": "ResNet50", "config": "configs/resnet50_lung_attention.yaml",
        "arms": ("A0_vanilla", "A3_multiply"), "artifact": "T23_resnet50_lung_attention",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0", "config": "configs/efficientnet_b0_lung_attention.yaml",
        "arms": ("A0", "A3"), "artifact": "T25_efficientnet_b0_lung_attention",
    },
}
TASKS = ("calibration", "counterfactual", "occlusion", "cam", "efficiency", "dependence")
CLASS_NAMES = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=SPECS)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=("all", *TASKS), default=["all"])
    parser.add_argument("--smoke-images", type=int, default=100)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def selected_tasks(requested):
    return list(TASKS) if "all" in requested else list(dict.fromkeys(requested))


def arm_settings(backbone: str, arm: str) -> dict:
    if arm not in SPECS[backbone]["arms"]:
        raise ValueError(f"{backbone} expects one of {SPECS[backbone]['arms']}; got {arm}")
    baseline = arm == SPECS[backbone]["arms"][0]
    return {
        "use_attention": not baseline,
        "gate_mode": "multiply" if arm in ("A3_multiply", "A3") else "residual",
    }


def canonical_checkpoint_arm(arm: str, seed: int) -> str:
    """Training stored repeat-run DenseNet arms as ``A2_full_seed123``."""
    suffix = f"_seed{seed}"
    return arm[:-len(suffix)] if arm.endswith(suffix) else arm


def load_cam_positions(backbone: str, seed: int, test_size: int) -> list[int]:
    base = ROOT / "artifacts" / SPECS[backbone]["artifact"]
    # Seed-42 files are identical across all three backbones; reuse those
    # positions for every seed so EIL comparisons score the same images.
    path = base / "cam_subset_indices.json"
    if not path.is_file():
        raise FileNotFoundError(f"fixed CAM subset missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    positions = payload["positions"] if isinstance(payload, dict) else payload
    if len(positions) != 1000 or len(set(positions)) != 1000 or not all(
        isinstance(i, int) and 0 <= i < test_size for i in positions
    ):
        raise ValueError(f"invalid fixed 1,000-image CAM subset: {path}")
    return positions


def load_model(checkpoint: Path, cfg: dict, backbone: str, arm: str, seed: int):
    import torch
    from src.modules import build_model

    settings = arm_settings(backbone, arm)
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        backbone_name=cfg["model"]["name"], pretrained=False,
        drop_rate=cfg["model"].get("drop_rate", 0.0),
        reduction=cfg["module"]["reduction"],
        attention=cfg["module"].get("attention", "lung"), **settings,
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_data, dict):
        raise TypeError("checkpoint must be a dictionary")
    if "arm" in checkpoint_data and canonical_checkpoint_arm(
        str(checkpoint_data["arm"]), seed
    ) != arm:
        raise ValueError(f"checkpoint arm {checkpoint_data['arm']} does not match {arm}")
    if "seed" in checkpoint_data and int(checkpoint_data["seed"]) != seed:
        raise ValueError(f"checkpoint seed {checkpoint_data['seed']} does not match {seed}")
    if "class_names" in checkpoint_data and tuple(checkpoint_data["class_names"]) != CLASS_NAMES:
        raise ValueError("checkpoint class order does not match primary dataset")
    if "config" in checkpoint_data and checkpoint_data["config"]["model"]["name"] != backbone:
        raise ValueError("checkpoint backbone does not match requested backbone")
    state = checkpoint_data.get("model_state_dict", checkpoint_data)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a state dict or contain model_state_dict")
    model.load_state_dict(state, strict=True)
    return model.eval()


def smoke_test(model, loader, device, n_images: int, n_classes: int) -> int:
    import torch

    if n_images < 1:
        raise ValueError("smoke image count must be positive")
    seen = 0
    with torch.inference_mode():
        for images, _labels, _masks in loader:
            logits = model(images.to(device))[0]
            if logits.ndim != 2 or logits.shape[1] != n_classes or not torch.isfinite(logits).all():
                raise ValueError("checkpoint produced invalid classification logits")
            seen += len(images)
            if seen >= n_images:
                break
    if seen < n_images:
        raise ValueError(f"only {seen} test images available; requested {n_images}")
    return seen


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_test_accuracy(model, loader, device, class_names, expected_csv: Path,
                         *, tolerance: float = 1e-4, expected_n: int = 3175) -> dict:
    """Stop new XAI inference if fixed-test accuracy differs from saved output."""
    import pandas as pd
    import torch

    from src.modules.counterfactual import _as_logits, _dataset_image_paths
    from src.modules.xai_dependence import image_key

    expected = pd.read_csv(expected_csv)
    required = {"image_path", "true_label", "pred_label"}
    if not required.issubset(expected.columns) or len(expected) != expected_n:
        raise ValueError(f"committed predictions must have {expected_n} rows and label/path columns")
    expected["key"] = expected["image_path"].map(image_key)
    if expected["key"].duplicated().any():
        raise ValueError("committed predictions contain duplicate images")
    paths = _dataset_image_paths(getattr(loader, "dataset", None))
    if paths is None or len(paths) != len(expected):
        raise ValueError("test loader paths do not match committed predictions")
    rows = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for images, labels, _masks in loader:
            pred = _as_logits(model(images.to(device))).argmax(1).cpu().tolist()
            for j, p in enumerate(pred):
                rows.append((image_key(paths[offset + j]), class_names[int(labels[j])], class_names[p]))
            offset += len(images)
    observed = pd.DataFrame(rows, columns=["key", "true_label", "pred_label"])
    if observed["key"].duplicated().any() or set(observed["key"]) != set(expected["key"]):
        raise ValueError("evaluated image set differs from committed predictions")
    compared = observed.merge(expected[["key", "true_label", "pred_label"]],
                              on="key", suffixes=("_observed", "_committed"),
                              validate="one_to_one")
    if not (compared["true_label_observed"] == compared["true_label_committed"]).all():
        raise ValueError("test labels differ from committed predictions")
    accuracy = float((compared["true_label_observed"] == compared["pred_label_observed"]).mean())
    expected_accuracy = float((compared["true_label_committed"] == compared["pred_label_committed"]).mean())
    if abs(accuracy - expected_accuracy) > tolerance:
        raise ValueError(f"checkpoint accuracy {accuracy:.6f} does not match committed "
                         f"{expected_accuracy:.6f} within {tolerance}")
    return {"accuracy": accuracy, "committed_accuracy": expected_accuracy,
            "tolerance": tolerance, "images": len(compared),
            "committed_predictions": str(expected_csv)}


def run(args) -> Path:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"primary image directory not found: {args.data_dir}")
    arm_settings(args.backbone, args.arm)

    import pandas as pd
    import torch
    import yaml
    from src.datasets import build_dataloaders
    from src.modules import (
        LogitsOnly, build_per_image_predictions, calibration_report,
        evaluate_counterfactual_robustness, evaluate_region_occlusion,
    )
    from src.utils import set_seed

    spec = SPECS[args.backbone]
    cfg = yaml.safe_load((ROOT / spec["config"]).read_text(encoding="utf-8"))
    split_path = ROOT / "artifacts/splits/split_manifest_v1.csv"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    set_seed(args.seed)
    _train, val_loader, test_loader, class_names, _targets, datasets = build_dataloaders(
        data_dir=args.data_dir, img_size=cfg["dataset"]["image_size"],
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
        split_manifest_path=split_path,
    )
    if tuple(class_names) != CLASS_NAMES:
        raise ValueError(f"unexpected class order: {class_names}")
    if len(datasets["val"]) != 3175 or len(datasets["test"]) != 3175:
        raise ValueError("fixed split must have 3,175 validation and 3,175 test images")
    model = load_model(args.checkpoint, cfg, args.backbone, args.arm, args.seed).to(device)
    n_smoke = smoke_test(model, test_loader, device, args.smoke_images, len(class_names))
    tasks = [] if args.smoke_only else selected_tasks(args.tasks)
    accuracy_verification = None
    if "dependence" in tasks:
        from scripts.build_cnn_closeout import SPECS as CLOSEOUT_SPECS, prediction_dir

        closeout_spec = next(s for s in CLOSEOUT_SPECS if s.backbone == spec["name"])
        expected_csv = prediction_dir(closeout_spec, args.seed,
                                      args.arm == closeout_spec.selected_arm) / "per_image_predictions.csv"
        accuracy_verification = verify_test_accuracy(
            model, test_loader, device, class_names, expected_csv)
    output = args.output_dir or (ROOT / "artifacts/cnn_closeout/inference" /
                                 args.backbone / args.arm / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "backbone": args.backbone, "arm": args.arm, "seed": args.seed,
        "config": spec["config"], "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "split_manifest": str(split_path), "split_manifest_sha256": file_sha256(split_path),
        "data_dir": str(args.data_dir.resolve()), "val_images": len(datasets["val"]),
        "test_images": len(datasets["test"]), "smoke_images": n_smoke,
        "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
        "tasks": tasks, "accuracy_verification": accuracy_verification,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.smoke_only:
        return output

    if "calibration" in tasks:
        calibration_report(LogitsOnly(model), val_loader, test_loader, class_names,
                           device, output / "calibration", model_name=f"{spec['name']}-{args.arm}")
    if "counterfactual" in tasks:
        evaluate_counterfactual_robustness(
            model, test_loader, device, seed=args.seed, arm_name=args.arm,
            per_image_csv=output / "counterfactual_per_image.csv",
        ).to_csv(output / "counterfactual_summary.csv", index=False)
    if "occlusion" in tasks:
        evaluate_region_occlusion(
            model, test_loader, device, arm_name=args.arm,
            per_image_csv=output / "occlusion_per_image.csv",
        ).to_csv(output / "occlusion_summary.csv", index=False)
    if "cam" in tasks:
        positions = load_cam_positions(args.backbone, args.seed, len(datasets["test"]))
        build_per_image_predictions(
            model, datasets["test"], class_names, device, cam_subset=positions,
            batch_size=args.batch_size,
        ).to_csv(output / "per_image_predictions.csv", index=False)
    if "efficiency" in tasks:
        from notebooks.efficiency import benchmark_model

        row = benchmark_model(LogitsOnly(model), spec["name"], args.arm,
                              checkpoint_path=args.checkpoint,
                              input_shape=(1, 3, cfg["dataset"]["image_size"],
                                           cfg["dataset"]["image_size"]),
                              cpu_runs=20, gpu_runs=50)
        pd.DataFrame([row]).to_csv(output / "efficiency.csv", index=False)
    if "dependence" in tasks:
        from src.modules.xai_dependence import evaluate_xai_dependence

        pairs_path = ROOT / "artifacts/explainable_ai/swap_pairs_seed42.csv"
        per_image, summary = evaluate_xai_dependence(
            model, test_loader, class_names, pd.read_csv(pairs_path), device,
            arm=args.arm, seed=args.seed, pairing_seed=42,
        )
        per_image.to_csv(output / "dependence_per_image.csv", index=False)
        summary.insert(0, "backbone", args.backbone)
        summary.to_csv(output / "dependence_summary.csv", index=False)
    return output


def main():
    args = parse_args()
    output = run(args)
    print(f"CNN closeout evaluation written to {output}")


if __name__ == "__main__":
    main()
