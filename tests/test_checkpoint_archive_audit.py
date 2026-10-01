"""Archive inventory reads checkpoint metadata without unpacking model files."""
from __future__ import annotations

import io
from zipfile import ZipFile

import torch

from scripts.audit_checkpoint_archives import inspect_archives, inspect_checkpoint
from scripts.run_cnn_closeout_inference import CLASS_NAMES, canonical_checkpoint_arm


def test_checkpoint_inventory_identifies_metadata_and_missing_runs(tmp_path):
    assert canonical_checkpoint_arm("A2_full_seed123", 123) == "A2_full"
    payload = io.BytesIO()
    torch.save({
        "model_state_dict": {"weight": torch.ones(1)},
        "class_names": list(CLASS_NAMES),
        "arm": "A0_vanilla", "seed": 42,
        "config": {"model": {"name": "densenet121", "num_classes": 4},
                   "module": {"use_attention": False}},
    }, payload)
    assert inspect_checkpoint(payload.getvalue())["backbone"] == "densenet121"
    archive = tmp_path / "model.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("runs/A0.pt", payload.getvalue())
    table, coverage = inspect_archives([archive])
    assert table.iloc[0]["status"] == "verified"
    assert coverage["core_runs_verified"] == 1
    assert coverage["core_runs_expected"] == 18
