"""Record the inputs available for the three-CNN inference closeout.

Run with optional --checkpoint-root and --data-root paths. This does not
load weights or infer that similarly named files are compatible checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "cnn_closeout" / "run_readiness_manifest.json"
MODELS = (
    ("DenseNet121", "A0_vanilla", "configs/densenet121_lung_attention.yaml"),
    ("DenseNet121", "A2_full", "configs/densenet121_lung_attention.yaml"),
    ("ResNet50", "A0_vanilla", "configs/resnet50_lung_attention.yaml"),
    ("ResNet50", "A3_multiply", "configs/resnet50_lung_attention.yaml"),
    ("EfficientNet-B0", "A0", "configs/efficientnet_b0_lung_attention.yaml"),
    ("EfficientNet-B0", "A3", "configs/efficientnet_b0_lung_attention.yaml"),
)
SEEDS = (42, 123, 2026)


def split_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            counts[split] = counts.get(split, 0) + 1
    return counts


def candidate_checkpoints(roots: list[Path]) -> list[str]:
    extensions = {".pt", ".pth", ".ckpt", ".safetensors"}
    found = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() in extensions:
            found.add(str(root.resolve()))
        elif root.is_dir():
            for file in root.rglob("*"):
                if file.is_file() and file.suffix.lower() in extensions:
                    found.add(str(file.resolve()))
    return sorted(found)


def build_manifest(checkpoint_roots: list[Path], data_root: Path | None) -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    split = ROOT / "artifacts/splits/split_manifest_v1.csv"
    external = ROOT / "artifacts/segmentation/external_manifest_v1.csv"
    candidates = candidate_checkpoints(checkpoint_roots)
    runs = [
        {
            "backbone": backbone,
            "arm": arm,
            "seed": seed,
            "config": config,
            "split_manifest": str(split.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint": None,
            "checkpoint_status": "unverified; assign exact path and load strict state dict",
            "pending": ["checkpoint smoke test", "counterfactual", "region occlusion",
                        "validation-fitted calibration", "external evaluation"],
        }
        for backbone, arm, config in MODELS for seed in SEEDS
    ]
    return {
        "source_commit": commit,
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "dataset": {
            "split_counts": split_counts(split),
            "external_manifest_counts": split_counts(external),
            "data_root": str(data_root.resolve()) if data_root else None,
            "data_root_exists": data_root.is_dir() if data_root else False,
        },
        "checkpoint_search_roots": [str(p.resolve()) for p in checkpoint_roots],
        "candidate_checkpoints": candidates,
        "runs": runs,
        "note": "Candidate files are not assigned to runs until architecture, arm, seed, and weights are verified.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, action="append", default=[])
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    payload = build_manifest(args.checkpoint_root or [ROOT / "artifacts"], args.data_root)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}: {len(payload['runs'])} run slots, "
          f"{len(payload['candidate_checkpoints'])} checkpoint candidates")


if __name__ == "__main__":
    main()
