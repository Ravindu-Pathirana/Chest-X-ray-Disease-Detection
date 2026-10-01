"""Deterministic toy tests for the new L3 dependence path; no real images."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.modules.counterfactual import perturb_background, perturb_lung
from src.modules.xai_dependence import (
    area_matched_background_mask, build_swap_pairs, evaluate_xai_dependence,
    total_variation_distance,
)
from scripts.run_cnn_closeout_inference import verify_test_accuracy


class TinyDataset(Dataset):
    def __init__(self):
        self.base_dataset = SimpleNamespace(samples=[
            (f"/dataset/{label}/images/{label}-{i}.png", j)
            for i, (label, j) in enumerate((("A", 0), ("B", 1), ("A", 0), ("B", 1)))
        ])
        self.indices = list(range(4))

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        image = torch.zeros(1, 8, 8)
        image[:, :, :2] = float(idx + 1)
        image[:, :, 2:] = float(4 - idx)
        mask = torch.zeros(1, 8, 8)
        mask[:, :, :2] = 1.0
        return image, self.base_dataset.samples[idx][1], mask


class TinyModel(torch.nn.Module):
    def forward(self, images):
        score = images[:, :, :, :2].mean((1, 2, 3)) - images[:, :, :, 2:].mean((1, 2, 3))
        return torch.stack((-score, score), dim=1)


def _pairs():
    ds = TinyDataset()
    manifest = pd.DataFrame({
        "image_path": [p for p, _ in ds.base_dataset.samples],
        "label": ["A", "B", "A", "B"], "split": ["test"] * 4,
    })
    return build_swap_pairs(manifest)


def test_swap_pairing_is_deterministic_and_different_class():
    a, b = _pairs(), _pairs()
    pd.testing.assert_frame_equal(a, b)
    assert (a["true_label"] != a["donor_label"]).all()
    assert a["image_path"].is_unique


def test_area_matching_is_repeatable_and_lung_excluding():
    _, _, mask = TinyDataset()[0]
    batch = mask.unsqueeze(0).repeat(2, 1, 1, 1)
    keys = ["A/images/a.png", "B/images/b.png"]
    a = area_matched_background_mask(batch, keys)
    b = area_matched_background_mask(batch, keys)
    torch.testing.assert_close(a, b)
    assert not bool(((a > 0) & (batch > 0)).any())
    assert a.sum().item() == batch.sum().item()
    with pytest.raises(ValueError, match="cannot area-match"):
        area_matched_background_mask(torch.ones_like(batch), keys)


def test_blur_and_swap_preserve_unselected_pixels():
    ds = TinyDataset()
    image, _, mask = ds[0]
    donor, _, _ = ds[1]
    image, mask, donor = (x.unsqueeze(0) for x in (image, mask, donor))
    area = area_matched_background_mask(mask, ["A/images/a.png"])
    for mode in ("blur", "swap"):
        bg = perturb_background(image, mask, mode=mode, blur_kernel=3, donor_images=donor)
        lung = perturb_lung(image, mask, mode=mode, blur_kernel=3, donor_images=donor)
        ctrl = perturb_background(image, mask, mode=mode, blur_kernel=3,
                                  donor_images=donor, area_mask=area)
        torch.testing.assert_close(bg[mask.bool()], image[mask.bool()])
        torch.testing.assert_close(lung[~mask.bool()], image[~mask.bool()])
        torch.testing.assert_close(ctrl[(1 - area).bool()], image[(1 - area).bool()])
    with pytest.raises(ValueError, match="donor_images"):
        perturb_background(image, mask, mode="swap")
    with pytest.raises(ValueError, match="donor_images"):
        perturb_lung(image, mask, mode="swap")


def test_total_variation_is_full_distribution_not_class_drop():
    first = torch.tensor([[0.7, 0.2, 0.1]])
    second = torch.tensor([[0.4, 0.6, 0.0]])
    assert total_variation_distance(first, second).item() == pytest.approx(0.4)
    with pytest.raises(ValueError, match="sum to one"):
        total_variation_distance(first, second * 2)


def test_evaluator_keeps_per_image_probabilities_and_paired_lrg():
    ds = TinyDataset()
    rows, summary = evaluate_xai_dependence(
        TinyModel(), DataLoader(ds, batch_size=2, shuffle=False), ("A", "B"),
        _pairs(), torch.device("cpu"), arm="A0", blur_kernel=3,
    )
    assert len(rows) == 4 * 2 * 3
    assert set(rows["region"]) == {"lung", "background", "background_area_matched"}
    assert set(summary["mode"]) == {"blur", "swap"}
    assert (rows["lung_pixels"] == rows["area_matched_background_pixels"]).all()
    assert rows.groupby(["image_path", "mode"])["LRG"].nunique().eq(1).all()
    assert rows["donor_class"].ne(rows["true_label"]).all()
    for row in rows.itertuples():
        original = json.loads(row.original_probabilities)
        changed = json.loads(row.perturbed_probabilities)
        assert sum(original) == pytest.approx(1.0)
        assert sum(changed) == pytest.approx(1.0)
        assert row.dP == pytest.approx(0.5 * sum(abs(a - b) for a, b in zip(original, changed)))


def test_evaluator_rejects_shuffled_loader_and_changed_pairing():
    ds = TinyDataset()
    with pytest.raises(ValueError, match="sequential sampling"):
        evaluate_xai_dependence(
            TinyModel(), DataLoader(ds, batch_size=2, shuffle=True), ("A", "B"),
            _pairs(), torch.device("cpu"), arm="A0", blur_kernel=3,
        )
    changed = _pairs()
    changed.loc[0, "pair_seed"] = 123
    with pytest.raises(ValueError, match="pair seed"):
        evaluate_xai_dependence(
            TinyModel(), DataLoader(ds, batch_size=2), ("A", "B"),
            changed, torch.device("cpu"), arm="A0", blur_kernel=3,
        )


def test_accuracy_verification_stops_mismatched_checkpoint(tmp_path):
    ds = TinyDataset()
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = TinyModel()
    names = ("A", "B")
    expected = []
    for i in range(len(ds)):
        image, label, _mask = ds[i]
        pred = int(model(image.unsqueeze(0)).argmax(1))
        expected.append({"image_path": ds.base_dataset.samples[i][0],
                         "true_label": names[label], "pred_label": names[pred]})
    path = tmp_path / "per_image_predictions.csv"
    pd.DataFrame(expected).to_csv(path, index=False)
    result = verify_test_accuracy(model, loader, torch.device("cpu"), names, path,
                                  expected_n=4)
    assert result["accuracy"] == result["committed_accuracy"]
    expected[0]["pred_label"] = names[1 - names.index(expected[0]["pred_label"])]
    pd.DataFrame(expected).to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        verify_test_accuracy(model, loader, torch.device("cpu"), names, path,
                             expected_n=4)
