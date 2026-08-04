"""Best-model checkpoint management, decoupled from any specific architecture.

Answers "which epoch produced the selected model?" by tracking a single
monitored metric (e.g. `val_f1`, `val_auroc`) across epochs and saving a
checkpoint only when it improves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class BestCheckpointSaver:
    """Tracks a monitored metric across epochs and saves the best checkpoint.

    Example:
        saver = BestCheckpointSaver(
            run_name="resnet50_baseline_seed42",
            monitor="val_f1",
            mode="max",
            checkpoint_dir="artifacts/checkpoints",
        )
        for epoch in range(epochs):
            ...
            improved = saver.step(
                epoch=epoch,
                metric_value=val_f1,
                model=model,
                optimizer=optimizer,
                config=config,
            )

        log_summary_metrics(saver.summary())
    """

    def __init__(
        self,
        run_name: str,
        monitor: str = "val_f1",
        mode: str = "max",
        checkpoint_dir: Union[str, Path] = "artifacts/checkpoints",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self.run_name = run_name
        self.monitor = monitor
        self.mode = mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.checkpoint_path = self.checkpoint_dir / f"{run_name}_best.pth"

    def _is_improvement(self, metric_value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return metric_value > self.best_metric
        return metric_value < self.best_metric

    def step(
        self,
        epoch: int,
        metric_value: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Compare `metric_value` to the best seen so far; save a checkpoint if improved.

        Args:
            epoch: Current epoch number (0- or 1-indexed, just be consistent).
            metric_value: The monitored metric's value this epoch.
            model: The model whose `state_dict` should be saved.
            optimizer: The optimizer whose `state_dict` should be saved.
            config: The run's configuration dict, saved alongside the
                weights so the checkpoint is self-describing.
            scheduler: Optional LR scheduler; its `state_dict` is saved if given.
            extra: Optional extra key/value pairs to include in the checkpoint.

        Returns:
            True if this call produced a new best checkpoint, False otherwise.
        """
        if not self._is_improvement(metric_value):
            return False

        self.best_metric = metric_value
        self.best_epoch = epoch

        checkpoint: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": metric_value,
            "monitor": self.monitor,
            "mode": self.mode,
            "run_name": self.run_name,
            "config": config,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, self.checkpoint_path)
        return True

    def summary(self) -> Dict[str, Any]:
        """Return `{"best_epoch": ..., "best_<monitor>": ...}` for W&B run summary logging."""
        return {
            "best_epoch": self.best_epoch,
            f"best_{self.monitor}": self.best_metric,
        }


def load_checkpoint(path: Union[str, Path], map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint dict previously saved by `BestCheckpointSaver`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)
