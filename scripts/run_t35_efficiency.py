"""Benchmark the eight A0/selected architectures in one hardware session.

No dataset or trained weights are needed for parameter/FLOP/latency
measurements. This script labels such results architecture-only; checkpoint
sizes and final GPU timing require the supplied weights and target GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.efficiency import benchmark_model
from src.modules import LogitsOnly, ViTLungAttention, build_model


GRID = (
    ("DenseNet121", "densenet121", "A0_vanilla", False, "residual", 8),
    ("DenseNet121", "densenet121", "A2_full", True, "residual", 8),
    ("ResNet50", "resnet50", "A0_vanilla", False, "residual", 8),
    ("ResNet50", "resnet50", "A3_multiply", True, "multiply", 8),
    ("EfficientNet-B0", "efficientnet_b0", "A0", False, "residual", 16),
    ("EfficientNet-B0", "efficientnet_b0", "A3", True, "multiply", 16),
    ("ViT-Base/16", "vit_base_patch16_224", "A0", False, "residual", 8),
    ("ViT-Base/16", "vit_base_patch16_224", "A2_full", True, "residual", 8),
)

CONFIGS = {
    "densenet121": "configs/densenet121_lung_attention.yaml",
    "resnet50": "configs/resnet50_lung_attention.yaml",
    "efficientnet_b0": "configs/efficientnet_b0_lung_attention.yaml",
}


def build_architecture(backbone: str, use_attention: bool,
                       gate_mode: str, reduction: int):
    if backbone == "vit_base_patch16_224":
        return ViTLungAttention(num_classes=4, pretrained=False,
                                use_attention=use_attention, reduction=reduction,
                                gate_mode=gate_mode)
    config = yaml.safe_load((ROOT / CONFIGS[backbone]).read_text(encoding="utf-8"))
    return build_model(
        num_classes=4, backbone_name=backbone, pretrained=False,
        drop_rate=float(config["model"].get("drop_rate", 0.0)),
        use_attention=use_attention, reduction=reduction,
        gate_mode=gate_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-runs", type=int, default=20)
    parser.add_argument("--gpu-runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/T35_efficiency")
    parser.add_argument("--include-vit", action="store_true",
                        help="include architecture-only ViT pair; weights not yet validated")
    args = parser.parse_args()
    if min(args.cpu_runs, args.gpu_runs, args.warmup) < 1:
        raise ValueError("run and warmup counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, backbone, arm, use_attention, gate_mode, reduction in GRID:
        if backbone.startswith("vit_") and not args.include_vit:
            continue
        model = LogitsOnly(build_architecture(backbone, use_attention, gate_mode, reduction))
        result = benchmark_model(
            model, name, arm, input_shape=(1, 3, 224, 224),
            cpu_warmup=args.warmup, cpu_runs=args.cpu_runs,
            gpu_warmup=args.warmup, gpu_runs=args.gpu_runs,
        )
        if result["gflops"] is None or not math.isfinite(result["gflops"]):
            raise RuntimeError(f"FLOPs unavailable for {backbone}/{arm}; do not report a partial T35 table")
        rows.append({"backbone": backbone, "arm": arm,
                     "status": "architecture_only_untrained",
                     "flops_status": "fvcore_partial_unsupported_ops_possible",
                     **result})
        del model
        gc.collect()
    output = args.output_dir / "architecture_only_efficiency.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    details = {
        "scope": "architecture-only; random initialisation, no checkpoint or dataset",
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "input_shape": [1, 3, 224, 224], "cpu_runs": args.cpu_runs,
        "gpu_runs": args.gpu_runs, "warmup": args.warmup,
        "model_count": len(rows),
        "warning": "Do not combine latency values from different hardware sessions or claim trained-model inference accuracy. fvcore reported unsupported operators, especially fused ViT attention, so GFLOPs are partial estimates rather than complete operation counts.",
    }
    (args.output_dir / "architecture_only_hardware.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} architecture-only rows to {output}")


if __name__ == "__main__":
    main()
