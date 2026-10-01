"""Inventory checkpoint ZIPs without extracting or committing model weights.

Example: python scripts/audit_checkpoint_archives.py --archive results-dense.zip
--archive efficientnet.zip --strict-load
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cnn_closeout_inference import CLASS_NAMES, SPECS, canonical_checkpoint_arm

OUTPUT = ROOT / "artifacts" / "explainable_ai"


def expected_core_runs() -> set[tuple[str, str, int]]:
    return {(name, arm, seed) for name, spec in SPECS.items()
            for arm in spec["arms"] for seed in (42, 123, 2026)}


def inspect_checkpoint(payload: bytes, *, strict_load: bool = False) -> dict:
    """Load only tensor-safe checkpoint contents and verify metadata/state."""
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("expected a checkpoint dictionary with model_state_dict")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no training config")
    model_cfg = config["model"]
    module_cfg = config["module"]
    backbone = str(model_cfg["name"])
    arm = str(checkpoint["arm"])
    seed = int(checkpoint["seed"])
    class_names = tuple(checkpoint["class_names"])
    if class_names != CLASS_NAMES:
        raise ValueError(f"unexpected class order: {class_names}")
    if strict_load:
        from src.modules import build_model

        model = build_model(
            num_classes=int(model_cfg["num_classes"]),
            backbone_name=backbone,
            pretrained=False,
            drop_rate=float(model_cfg.get("drop_rate", 0.0)),
            use_attention=bool(module_cfg["use_attention"]),
            attention=module_cfg.get("attention", "lung"),
            gate_mode=module_cfg.get("gate_mode", "residual"),
            reduction=int(module_cfg.get("reduction", 8)),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "backbone": backbone, "arm": arm,
        "canonical_arm": canonical_checkpoint_arm(arm, seed), "seed": seed,
        "class_names": "|".join(class_names),
        "strict_model_load": bool(strict_load),
        "state_tensors": len(checkpoint["model_state_dict"]),
    }


def inspect_archives(paths: list[Path], *, strict_load: bool = False) -> tuple[pd.DataFrame, dict]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"corrupt ZIP member in {path.name}: {bad}")
            for member in archive.infolist():
                if Path(member.filename).suffix.lower() not in {".pt", ".pth", ".ckpt"}:
                    continue
                payload = archive.read(member)
                row = {
                    "archive": path.name,
                    "member": member.filename,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                if "smoke_test/" in member.filename.replace("\\", "/"):
                    row["status"] = "excluded_smoke"
                    row["error"] = "smoke-test weights are not a final research checkpoint"
                    rows.append(row)
                    continue
                try:
                    row.update(inspect_checkpoint(payload, strict_load=strict_load))
                    row["status"] = "verified"
                    row["error"] = ""
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    frame = pd.DataFrame(rows)
    found = {(r.backbone, r.canonical_arm, int(r.seed)) for r in frame.itertuples()
             if r.status == "verified"}
    expected = expected_core_runs()
    coverage = {
        "strict_model_load_performed": strict_load,
        "checkpoint_entries": len(frame),
        "verified_entries": int(frame["status"].eq("verified").sum()) if len(frame) else 0,
        "core_runs_verified": len(expected & found),
        "core_runs_expected": len(expected),
        "missing_core_runs": [
            {"backbone": backbone, "arm": arm, "seed": seed}
            for backbone, arm, seed in sorted(expected - found)
        ],
        "note": "Checkpoint availability does not imply image data or masks are available for inference.",
    }
    return frame, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--strict-load", action="store_true",
                        help="construct each model and load its state dict with strict=True")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    frame, coverage = inspect_archives(args.archive, strict_load=args.strict_load)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "checkpoint_inventory.csv", index=False)
    (args.output_dir / "checkpoint_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {coverage['verified_entries']} checkpoint entries; "
          f"{coverage['core_runs_verified']}/{coverage['core_runs_expected']} core runs covered")


if __name__ == "__main__":
    main()
