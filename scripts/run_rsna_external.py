"""Evaluate a trained four-class CNN on the prepared RSNA binary manifest.

Declare --positive-classes and --protocol-id before running full inference.
The score is the sum of the chosen primary-class probabilities. This is a
binary discrimination analysis, not a four-class RSNA accuracy estimate.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cnn_closeout_inference import CLASS_NAMES, SPECS, file_sha256, load_model


class ExternalDataset:
    """Pickle-safe dataset so DataLoader workers also work on Windows."""

    def __init__(self, frame, image_size):
        from src.datasets import JointTransform

        self.paths = frame["local_path"].tolist()
        self.transform = JointTransform(img_size=image_size, train=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        image = Image.open(self.paths[index]).convert("L")
        blank = Image.new("L", image.size, 0)
        tensor, _mask = self.transform(image, blank)
        return tensor, index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=SPECS)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--rsna-dir", required=True, type=Path,
                        help="Directory containing Normal/ and Pneumonia/ PNG folders")
    parser.add_argument("--positive-classes", nargs="+", required=True, choices=CLASS_NAMES)
    parser.add_argument("--protocol-id", required=True,
                        help="Name of the predeclared label mapping and analysis protocol")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def validate_mapping(positive_classes, protocol_id):
    selected = set(positive_classes)
    if not protocol_id.strip():
        raise ValueError("a nonempty, predeclared protocol ID is required")
    if not selected or "Normal" in selected or selected - set(CLASS_NAMES):
        raise ValueError("positive classes must be a nonempty subset of non-Normal primary classes")
    return sorted(selected)


def resolve_manifest(manifest_path: Path, rsna_dir: Path):
    import pandas as pd

    frame = pd.read_csv(manifest_path)
    required = {"patient_id", "image_path", "label", "split"}
    if not required <= set(frame.columns):
        raise ValueError(f"external manifest is missing {sorted(required - set(frame.columns))}")
    if frame["patient_id"].duplicated().any() or not frame["split"].eq("external_test").all():
        raise ValueError("external manifest must contain unique patients on external_test only")
    if set(frame["label"]) != {"Normal", "Pneumonia"}:
        raise ValueError("external manifest labels must be Normal and Pneumonia")
    frame["local_path"] = [str(rsna_dir / label / Path(source).name)
                           for label, source in zip(frame["label"], frame["image_path"])]
    missing = [path for path in frame["local_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} RSNA images missing; first: {missing[0]}")
    return frame


def run(args):
    positive = validate_mapping(args.positive_classes, args.protocol_id)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.rsna_dir.is_dir():
        raise FileNotFoundError(f"RSNA image directory not found: {args.rsna_dir}")

    import pandas as pd
    import torch
    import yaml
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    cfg = yaml.safe_load((ROOT / SPECS[args.backbone]["config"]).read_text(encoding="utf-8"))
    manifest_path = ROOT / "artifacts/segmentation/external_manifest_v1.csv"
    frame = resolve_manifest(manifest_path, args.rsna_dir)
    model = load_model(args.checkpoint, cfg, args.backbone, args.arm, args.seed).to(device).eval()
    loader = DataLoader(ExternalDataset(frame, cfg["dataset"]["image_size"]),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    probability_rows = []
    with torch.inference_mode():
        for images, indices in loader:
            logits = model(images.to(device))[0]
            if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES) or not torch.isfinite(logits).all():
                raise ValueError("checkpoint produced invalid external logits")
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            probability_rows.extend(zip(indices.tolist(), probs))
            if args.smoke_only and len(probability_rows) >= 100:
                break
    if args.smoke_only:
        print(f"RSNA smoke test passed: {len(probability_rows)} images")
        return None
    if len(probability_rows) != len(frame):
        raise ValueError("external inference did not score every manifest row")
    if [i for i, _ in probability_rows] != list(range(len(frame))):
        raise ValueError("external image order changed during inference")

    output = args.output_dir or (ROOT / "artifacts/cnn_closeout/rsna_external" /
                                 args.backbone / args.arm / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    scores = []
    per_image = []
    selected_indices = [CLASS_NAMES.index(name) for name in positive]
    for row, (_index, probs) in zip(frame.itertuples(index=False), probability_rows):
        score = sum(probs[i] for i in selected_indices)
        scores.append(score)
        per_image.append({
            "patient_id": row.patient_id, "image_path": row.image_path,
            "rsna_label": row.label, "positive_score": score,
            **{f"prob_{i}": prob for i, prob in enumerate(probs)},
        })
    labels = frame["label"].eq("Pneumonia").astype(int)
    summary = {
        "protocol_id": args.protocol_id,
        "positive_classes": positive,
        "score_definition": "sum of selected primary-class softmax probabilities",
        "n_external": len(frame), "n_positive": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "backbone": args.backbone, "arm": args.arm, "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "external_manifest_sha256": file_sha256(manifest_path),
        "rsna_dir": str(args.rsna_dir.resolve()),
        "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
    }
    pd.DataFrame(per_image).to_csv(output / "rsna_per_image.csv", index=False)
    (output / "rsna_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    result = run(parse_args())
    if result is not None:
        print(f"RSNA external evaluation written to {result}")


if __name__ == "__main__":
    main()
