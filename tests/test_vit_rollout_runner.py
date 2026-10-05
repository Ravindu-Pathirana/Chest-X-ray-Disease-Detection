"""Tests for scripts/run_vit_rollout.py: checkpoint checks, output files, aggregation.

No dataset is needed: `build_dataloaders` is replaced by a small fake loader,
and the checkpoint is a tiny randomly initialised ViT saved in the T26 format.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

import scripts.run_vit_rollout as runner
from src.modules import build_model

TINY = "vit_tiny_patch16_224"
CLASSES = runner.CLASS_NAMES


def _save_checkpoint(path, arm="A3_multiply", seed=42, use_attention=True, **overrides):
    torch.manual_seed(0)
    model = build_model(use_attention=use_attention, pretrained=False, backbone_name=TINY,
                        gate_mode="multiply")
    payload = {
        "arm": arm, "seed": seed, "class_names": list(CLASSES),
        "config": {
            "model": {"name": TINY, "num_classes": 4, "drop_rate": 0.0},
            "module": {"use_attention": use_attention, "attention": "lung",
                       "gate_mode": "multiply", "reduction": 8},
        },
        "model_state_dict": model.state_dict(),
    }
    payload.update(overrides)
    torch.save(payload, path)
    return path


class _FakeTestSet:
    def __len__(self):
        return 3175


def _fake_build_dataloaders(**kwargs):
    g = torch.Generator().manual_seed(2)
    mask = torch.zeros(4, 224, 224)
    mask[:, 48:176, 48:176] = 1.0
    batches = [(torch.randn(4, 3, 224, 224, generator=g), torch.tensor([0, 1, 2, 3]), mask)
               for _ in range(2)]
    return None, None, batches, list(CLASSES), None, {"test": _FakeTestSet()}


def _args(tmp_path, ckpt, **kw):
    argv = ["--arm", kw.pop("arm", "A3_multiply"), "--seed", str(kw.pop("seed", 42)),
            "--checkpoint", str(ckpt), "--data-dir", str(tmp_path), "--device", "cpu",
            "--output-dir", str(tmp_path / "out"), "--num-workers", "0"]
    for key, value in kw.items():
        argv += [f"--{key.replace('_', '-')}"] + ([] if value is True else [str(value)])
    return runner.parse_args(argv)


def test_arm_is_validated_and_required_args_are_enforced(tmp_path):
    with pytest.raises(SystemExit):
        runner.parse_args(["--arm", "A9_nope", "--checkpoint", "x", "--data-dir", "y"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--arm", "A0_vanilla"])
    assert runner.parse_args(["--aggregate"]).aggregate


def test_load_vit_strict_loads_and_returns_eval_model(tmp_path):
    model = runner.load_vit(_save_checkpoint(tmp_path / "ok.pt"), "A3_multiply", 42)
    assert not model.training and model.attn is not None


def test_repeat_seed_checkpoints_store_arm_with_a_seed_suffix(tmp_path):
    path = _save_checkpoint(tmp_path / "s.pt", arm="A3_multiply_seed123", seed=123)
    assert runner.load_vit(path, "A3_multiply", 123) is not None


@pytest.mark.parametrize("bad, match", [
    ({"arm": "A0_vanilla"}, "arm"),
    ({"seed": 7}, "seed"),
    ({"class_names": ["a", "b", "c", "d"]}, "class order"),
])
def test_load_vit_rejects_mislabelled_checkpoints(tmp_path, bad, match):
    path = _save_checkpoint(tmp_path / "bad.pt", **bad)
    with pytest.raises(ValueError, match=match):
        runner.load_vit(path, "A3_multiply", 42)


def test_load_vit_rejects_a_cnn_checkpoint(tmp_path):
    path = _save_checkpoint(tmp_path / "cnn.pt")
    data = torch.load(path, weights_only=False)
    data["config"]["model"]["name"] = "resnet50"
    torch.save(data, path)
    with pytest.raises(ValueError, match="not a ViT"):
        runner.load_vit(path, "A3_multiply", 42)


def test_cam_subset_file_is_the_fixed_1000_images():
    positions = runner.load_cam_positions(3175)
    assert len(positions) == 1000 and len(set(positions)) == 1000


def test_run_writes_summary_per_image_and_manifest(tmp_path, monkeypatch):
    import src.datasets
    monkeypatch.setattr(src.datasets, "build_dataloaders", _fake_build_dataloaders)
    ckpt = _save_checkpoint(tmp_path / "c.pt")
    out = runner.run(_args(tmp_path, ckpt, max_images=8))
    summary = pd.read_csv(out / "rollout_summary.csv")
    assert {"arm", "seed", "source", "scope", "n_images", "mean_eil", "mean_eil_excess"} <= set(summary.columns)
    assert set(summary["source"]) == {"patch_mean", "cls"} and (summary["seed"] == 42).all()
    assert len(pd.read_csv(out / "rollout_per_image.csv")) == 8
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["arm"] == "A3_multiply" and manifest["test_images"] == 3175
    assert manifest["smoke"] is True and manifest["head_fusion"] == "mean"
    assert len(manifest["checkpoint_sha256"]) == 64 and len(manifest["split_manifest_sha256"]) == 64


def test_smoke_runs_default_to_a_folder_that_aggregation_ignores(tmp_path, monkeypatch):
    import src.datasets
    monkeypatch.setattr(src.datasets, "build_dataloaders", _fake_build_dataloaders)
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path / "root")
    ckpt = _save_checkpoint(tmp_path / "c.pt")
    args = runner.parse_args(["--arm", "A3_multiply", "--checkpoint", str(ckpt), "--data-dir", str(tmp_path),
                              "--device", "cpu", "--num-workers", "0", "--max-images", "8"])
    out = runner.run(args)
    assert "_smoke" in out.parts
    with pytest.raises(FileNotFoundError):
        runner.aggregate(tmp_path / "root")


def test_aggregate_collects_finished_runs_only(tmp_path):
    def write(arm, seed, mean, folder=None):
        d = tmp_path / (folder or arm) / f"seed_{seed}"
        d.mkdir(parents=True)
        pd.DataFrame([{"arm": arm, "seed": seed, "source": "patch_mean", "scope": "all", "mean_eil": mean}]
                     ).to_csv(d / "rollout_summary.csv", index=False)
    write("A3_multiply", 42, 0.5)
    write("A0_vanilla", 123, 0.2)
    write("A0_vanilla", 42, 0.1)
    write("A2_full", 42, 9.9, folder="_smoke")
    out = runner.aggregate(tmp_path)
    table = pd.read_csv(out)
    assert len(table) == 3 and "A2_full" not in set(table["arm"])
    assert list(zip(table["arm"], table["seed"])) == [("A0_vanilla", 42), ("A0_vanilla", 123), ("A3_multiply", 42)]
