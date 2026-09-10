"""T27: calibration harness -- ECE, Brier score, reliability diagrams, and
temperature scaling.

This is the trustworthiness axis every other model owner (T28) evaluates
their architecture on, the same way T34's efficiency.py is the shared
harness for the efficiency axis. Nothing here is T18-specific.

Protocol (matches docs/experiment_policy.md's general rule that the test
split is touched only for final reported numbers, never for
selection/fitting): temperature is fit on the VALIDATION split only, then
applied to the TEST split for the "after scaling" numbers. Fitting
temperature on the test split would be exactly the model-selection-on-test
the policy forbids -- do not swap the two loaders below.

``model`` passed into every function here must return a plain logits
tensor. T18-style models return `(logits, attention, attention_logits)` --
wrap those with `src.modules.LogitsOnly` first, the same convention
efficiency.py's benchmark_model and gradcam.py's cam_for already use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def collect_logits(model: nn.Module, loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs `model` over every batch in `loader`, returns (logits [N,C], labels [N]) on CPU.

    Tolerates both `(images, labels)` and `(images, labels, masks)` batches
    (this project's loaders use either, depending on whether they're
    mask-aware) -- only the first two elements are used.
    """
    model.eval()
    non_blocking = device.type == "cuda"
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=non_blocking)
        logits = model(images)
        if not torch.is_tensor(logits):
            raise TypeError(
                f"collect_logits expected model(images) to return a Tensor, got {type(logits)} -- "
                "wrap models that return a tuple (e.g. T18's attention models) with LogitsOnly first."
            )
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Top-label multiclass ECE (Guo et al. 2017, the proposal's own ref [8]).

    Bins predictions by the winning class's confidence into `n_bins` equal-
    width bins over [0,1]; ECE = sum_bins (|bin| / N) * |acc(bin) - conf(bin)|.
    0.0 is perfect calibration; there is no fixed upper bound, but values
    are typically well under 1.0 for a trained classifier.
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, dtype=probs.dtype)
    n = confidences.shape[0]
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = in_bin.sum()
        if count > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += (count.float() / n) * torch.abs(conf_in_bin - acc_in_bin)
    return ece.item()


def brier_score(probs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Multi-class Brier score: mean over samples of sum_c (p_c - y_c)^2,
    where y is the one-hot label. Range [0, 2]; 0.0 is perfect, lower is
    better (unlike accuracy/F1, this is a loss, not a score to maximize)."""
    one_hot = F.one_hot(labels, num_classes=num_classes).to(probs.dtype)
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()


def reliability_diagram_data(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> List[Dict[str, Any]]:
    """Per-bin (bin_lo, bin_hi, confidence, accuracy, count) for plotting or
    inspection -- the same binning `expected_calibration_error` uses
    internally, exposed so the two are provably consistent. Empty bins keep
    their bounds with confidence/accuracy set to None (not 0.0, which would
    misleadingly read as "fully miscalibrated" rather than "no data")."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    rows: List[Dict[str, Any]] = []
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum().item())
        row: Dict[str, Any] = {"bin_lo": lo.item(), "bin_hi": hi.item(), "count": count}
        if count > 0:
            row["confidence"] = confidences[in_bin].mean().item()
            row["accuracy"] = accuracies[in_bin].mean().item()
        else:
            row["confidence"] = None
            row["accuracy"] = None
        rows.append(row)
    return rows


class TemperatureScaler(nn.Module):
    """A single learnable scalar T > 0. calibrated_logits = logits / T.

    T > 1 softens an overconfident model (spreads probabilities toward
    uniform); T < 1 sharpens an underconfident one. Parameterized as
    log(T) so the optimizer can't drive T negative or through zero.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1.0 at init

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01) -> float:
    """Fits T by minimizing NLL (cross-entropy) on `(val_logits, val_labels)`
    via LBFGS -- must be called on the VALIDATION split only (see module
    docstring). Returns the fitted scalar T as a plain float."""
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(val_logits), val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature.item()


def plot_reliability_diagram(bins: List[Dict[str, Any]], title: str, output_path) -> None:
    """Saves a standard reliability diagram: accuracy per confidence bin as
    bars, against the y=x perfect-calibration line. Matplotlib is imported
    lazily so this module stays importable (and its non-plotting functions
    testable) without matplotlib installed, matching figures.py's convention."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bins = len(bins)
    width = 1.0 / n_bins
    centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
    accuracies = [b["accuracy"] if b["accuracy"] is not None else 0.0 for b in bins]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(centers, accuracies, width=width * 0.9, edgecolor="black", alpha=0.75, label="Accuracy")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def calibration_report(
    model: nn.Module,
    val_loader,
    test_loader,
    class_names: List[str],
    device: torch.device,
    output_dir,
    model_name: str = "model",
    n_bins: int = 15,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Full T27 harness for one model: fits temperature on `val_loader`,
    reports ECE/Brier on `test_loader` both before and after scaling, saves
    reliability diagrams for both, and writes `calibration_summary.json`.

    `model` must return plain logits -- see module docstring re: LogitsOnly.
    Returns the same dict that gets written to the JSON summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    raw_probs = F.softmax(test_logits, dim=1)
    raw_ece = expected_calibration_error(raw_probs, test_labels, n_bins)
    raw_brier = brier_score(raw_probs, test_labels, num_classes)

    temperature = fit_temperature(val_logits, val_labels)
    scaled_probs = F.softmax(test_logits / temperature, dim=1)
    scaled_ece = expected_calibration_error(scaled_probs, test_labels, n_bins)
    scaled_brier = brier_score(scaled_probs, test_labels, num_classes)

    if make_plots:
        raw_bins = reliability_diagram_data(raw_probs, test_labels, n_bins)
        scaled_bins = reliability_diagram_data(scaled_probs, test_labels, n_bins)
        plot_reliability_diagram(
            raw_bins, f"{model_name} -- before scaling (T=1.0, ECE={raw_ece:.4f})",
            output_dir / "reliability_before.png",
        )
        plot_reliability_diagram(
            scaled_bins, f"{model_name} -- after scaling (T={temperature:.3f}, ECE={scaled_ece:.4f})",
            output_dir / "reliability_after.png",
        )

    summary: Dict[str, Any] = {
        "model_name": model_name,
        "n_bins": n_bins,
        "temperature": temperature,
        "test_n": int(test_labels.shape[0]),
        "val_n_used_for_temperature_fit": int(val_labels.shape[0]),
        "before_scaling": {"ece": raw_ece, "brier": raw_brier},
        "after_scaling": {"ece": scaled_ece, "brier": scaled_brier},
    }
    with open(output_dir / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
