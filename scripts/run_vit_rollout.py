"""Attention-rollout EIL for the T26 ViT-Base checkpoints, without retraining.

One checkpoint (writes artifacts/vit_rollout/<arm>/seed_<seed>/):
python scripts/run_vit_rollout.py --arm A3_multiply --seed 42 \
    --checkpoint /weights/vit_base_patch16_224_A3_multiply.pt \
    --data-dir /data/COVID-19_Radiography_Dataset

Then collect every finished run into one table:
python scripts/run_vit_rollout.py --aggregate

Scored on the fixed 3,175-image test split. The ``cam_subset`` rows use the
same 1,000 images as the Grad-CAM EIL numbers (T26), so the two methods can be
compared image for image. Never use test results to pick a checkpoint.
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

ARMS = ("A0_vanilla", "A1_gate_only", "A2_full", "A3_multiply", "A4_guidance_only", "A5_cbam", "A6_full_bg")
CLASS_NAMES = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")
CONFIG = "configs/vit_base_lung_attention.yaml"
CAM_SUBSET = "artifacts/T26_vit_base_lung_attention/cam_subset_indices.json"
SPLIT_MANIFEST = "artifacts/splits/split_manifest_v1.csv"
OUT_ROOT = ROOT / "artifacts" / "vit_rollout"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregate", action="store_true", help="only collect finished runs into vit_rollout_master.csv")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--residual-weight", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, help="smoke test: stop after N images (output goes to a _smoke folder)")
    args = parser.parse_args(argv)
    if not args.aggregate:
        missing = [n for n in ("arm", "checkpoint", "data_dir") if getattr(args, n) is None]
        if missing:
            parser.error("missing required: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_arm(arm: str, seed: int) -> str:
    """Repeat-run checkpoints may store the arm as ``A3_multiply_seed123``."""
    suffix = f"_seed{seed}"
    return arm[:-len(suffix)] if arm.endswith(suffix) else arm


def load_cam_positions(test_size: int) -> list:
    path = ROOT / CAM_SUBSET
    payload = json.loads(path.read_text(encoding="utf-8"))
    positions = payload["positions"] if isinstance(payload, dict) else payload
    if len(positions) != 1000 or len(set(positions)) != 1000 or not all(
        isinstance(i, int) and 0 <= i < test_size for i in positions
    ):
        raise ValueError(f"invalid fixed 1,000-image CAM subset: {path}")
    return positions


def load_vit(checkpoint: Path, arm: str, seed: int):
    """Strict-loads a T26 checkpoint using the settings stored inside it."""
    import torch
    from src.modules import build_model

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "model_state_dict" not in data or "config" not in data:
        raise TypeError("expected a T26 checkpoint dict with 'config' and 'model_state_dict'")
    if "arm" in data and canonical_arm(str(data["arm"]), seed) != arm:
        raise ValueError(f"checkpoint arm {data['arm']!r} does not match --arm {arm}")
    if "seed" in data and int(data["seed"]) != seed:
        raise ValueError(f"checkpoint seed {data['seed']} does not match --seed {seed}")
    if "class_names" in data and tuple(data["class_names"]) != CLASS_NAMES:
        raise ValueError("checkpoint class order does not match the primary dataset")
    cfg, module = data["config"], data["config"]["module"]
    name = cfg["model"]["name"]
    if not name.startswith("vit"):
        raise ValueError(f"not a ViT checkpoint: {name}")
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        use_attention=module["use_attention"],
        attention=module.get("attention", "lung"),
        gate_mode=module.get("gate_mode", "residual"),
        reduction=module.get("reduction", 8),
        backbone_name=name, pretrained=False,
        drop_rate=cfg["model"].get("drop_rate", 0.0),
    )
    model.load_state_dict(data["model_state_dict"], strict=True)
    return model.eval()


def run(args) -> Path:
    import timm
    import torch
    import yaml
    from src.datasets import build_dataloaders
    from src.modules import evaluate_rollout_eil
    from src.utils import set_seed

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"primary image directory not found: {args.data_dir}")
    cfg = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    split_path = ROOT / SPLIT_MANIFEST
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)

    _train, _val, test_loader, class_names, _targets, datasets = build_dataloaders(
        data_dir=args.data_dir, img_size=cfg["dataset"]["image_size"], batch_size=args.batch_size,
        seed=args.seed, num_workers=args.num_workers, split_manifest_path=split_path,
    )
    if tuple(class_names) != CLASS_NAMES:
        raise ValueError(f"unexpected class order: {class_names}")
    if len(datasets["test"]) != 3175:
        raise ValueError("fixed split must have 3,175 test images")
    positions = load_cam_positions(len(datasets["test"]))

    output = args.output_dir or (OUT_ROOT / ("_smoke" if args.max_images else "") / args.arm / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    model = load_vit(args.checkpoint, args.arm, args.seed).to(device)
    summary = evaluate_rollout_eil(
        model, test_loader, device, class_names, arm_name=args.arm, subset_positions=positions,
        residual_weight=args.residual_weight, max_images=args.max_images,
        per_image_csv=output / "rollout_per_image.csv",
    )
    summary.insert(1, "seed", args.seed)
    summary.to_csv(output / "rollout_summary.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "source_commit": commit, "arm": args.arm, "seed": args.seed, "config": CONFIG,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "split_manifest": SPLIT_MANIFEST, "split_manifest_sha256": file_sha256(split_path),
        "cam_subset": CAM_SUBSET, "data_dir": str(args.data_dir.resolve()),
        "test_images": len(datasets["test"]), "images_scored": int(summary["n_images"].max()),
        "residual_weight": args.residual_weight, "head_fusion": "mean",
        "sources": ["patch_mean", "cls"], "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "timm": timm.__version__, "smoke": args.max_images is not None,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def aggregate(root: Path = OUT_ROOT) -> Path:
    import pandas as pd

    frames = []
    for path in sorted(root.glob("*/seed_*/rollout_summary.csv")):
        if path.parts[-3].startswith("_"):
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no finished runs under {root}")
    table = pd.concat(frames, ignore_index=True).sort_values(["source", "scope", "arm", "seed"])
    out = root / "vit_rollout_master.csv"
    table.to_csv(out, index=False)
    return out


def main():
    args = parse_args()
    if args.aggregate:
        print(f"rollout master table written to {aggregate()}")
        return
    output = run(args)
    print(f"ViT rollout evaluation written to {output}")


if __name__ == "__main__":
    main()
