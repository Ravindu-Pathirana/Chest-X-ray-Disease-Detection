"""Contract checks for checkpoint-only closeout runners."""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from scripts.run_cnn_closeout_inference import (
    CLASS_NAMES, arm_settings, load_cam_positions, load_model, smoke_test,
)
from scripts.run_rsna_external import resolve_manifest, validate_mapping
from scripts.summarize_cnn_inference import paired_result


def test_arm_contract_and_fixed_cam_subset():
    assert arm_settings("densenet121", "A0_vanilla")["use_attention"] is False
    assert arm_settings("densenet121", "A2_full")["gate_mode"] == "residual"
    assert arm_settings("resnet50", "A3_multiply")["gate_mode"] == "multiply"
    assert arm_settings("efficientnet_b0", "A3")["gate_mode"] == "multiply"
    with pytest.raises(ValueError):
        arm_settings("densenet121", "A3")
    positions = [load_cam_positions(name, 123, 3175) for name in
                 ("densenet121", "resnet50", "efficientnet_b0")]
    assert positions[0] == positions[1] == positions[2]


def test_checkpoint_metadata_mismatch_is_rejected(tmp_path, monkeypatch):
    import src.modules

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    monkeypatch.setattr(src.modules, "build_model", lambda **kwargs: Tiny())
    path = tmp_path / "wrong_seed.pt"
    torch.save({"model_state_dict": Tiny().state_dict(), "arm": "A0_vanilla",
                "seed": 123, "class_names": list(CLASS_NAMES),
                "config": {"model": {"name": "densenet121"}}}, path)
    cfg = {"model": {"num_classes": 4, "name": "densenet121"},
           "module": {"reduction": 8}}
    with pytest.raises(ValueError, match="checkpoint seed"):
        load_model(path, cfg, "densenet121", "A0_vanilla", 42)
    loaded = load_model(path, cfg, "densenet121", "A0_vanilla", 123)
    torch.testing.assert_close(loaded.weight, torch.ones(1))


def test_smoke_rejects_nonfinite_logits():
    class Bad(torch.nn.Module):
        def forward(self, images):
            return torch.full((len(images), 4), float("nan")), None, None

    loader = [(torch.zeros(2, 3, 8, 8), torch.zeros(2), torch.zeros(2, 1, 8, 8))]
    with pytest.raises(ValueError, match="invalid classification logits"):
        smoke_test(Bad(), loader, torch.device("cpu"), 2, 4)


def test_rsna_requires_explicit_binary_mapping_and_complete_images(tmp_path):
    assert validate_mapping(["Lung_Opacity"], "opacity_v1") == ["Lung_Opacity"]
    with pytest.raises(ValueError):
        validate_mapping(["Normal"], "bad")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"patient_id": "a", "image_path": "/kaggle/Normal/a.png",
         "label": "Normal", "split": "external_test"},
        {"patient_id": "b", "image_path": "/kaggle/Pneumonia/b.png",
         "label": "Pneumonia", "split": "external_test"},
    ]).to_csv(manifest, index=False)
    (tmp_path / "Normal").mkdir()
    (tmp_path / "Pneumonia").mkdir()
    (tmp_path / "Normal/a.png").touch()
    with pytest.raises(FileNotFoundError, match="1 RSNA images missing"):
        resolve_manifest(manifest, tmp_path)
    (tmp_path / "Pneumonia/b.png").touch()
    assert len(resolve_manifest(manifest, tmp_path)) == 2


def test_paired_intervention_requires_matching_images_and_labels():
    reference = pd.DataFrame({"image_path": ["a", "b"], "true_label": [0, 1],
                              "stability": [0.4, 0.6]})
    selected = pd.DataFrame({"image_path": ["a", "b"], "true_label": [0, 1],
                             "stability": [0.5, 0.7]})
    result = paired_result(reference, selected, ["image_path"], "stability", expected_n=2)
    assert result["selected_minus_baseline_mean"] == pytest.approx(0.1)
    selected.loc[1, "true_label"] = 0
    with pytest.raises(ValueError, match="labels differ"):
        paired_result(reference, selected, ["image_path"], "stability", expected_n=2)
