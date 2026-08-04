"""Reproducibility utilities: RNG seeding and deterministic DataLoader support.

Baseline mode (`cudnn.deterministic=True`, `cudnn.benchmark=False`) gives
run-to-run reproducibility for most convolutional workloads on the *same*
hardware/software stack. It does NOT guarantee bit-identical results across
different GPU models, CUDA/cuDNN versions, or driver versions -- exact
reproducibility across heterogeneous environments (e.g. one teammate's
Kaggle GPU vs. another's Colab GPU) is not achievable with PyTorch today.

Strict mode (`strict=True`) additionally calls `torch.use_deterministic_algorithms(True)`,
which forces PyTorch to raise a RuntimeError rather than silently fall back
to a non-deterministic kernel for an operation. Some ops have no
deterministic implementation and will raise; strict mode can also be
noticeably slower. Use it only when tracking down a suspected
non-determinism bug, not as the default training mode. See
https://pytorch.org/docs/stable/notes/randomness.html for the full list of
caveats.

The project's default baseline seed is 42, stored in `configs/baseline.yaml`
(`experiment.seed`) -- read it from config rather than hardcoding it at call
sites.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, strict: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all visible CUDA devices).

    Args:
        seed: The seed value to use everywhere.
        deterministic: If True (default), configure cuDNN for the
            reproducible baseline mode: `cudnn.deterministic = True`,
            `cudnn.benchmark = False`. Set to False only when reproducibility
            doesn't matter and raw throughput does (e.g. a quick smoke test).
        strict: If True, also enable `torch.use_deterministic_algorithms`.
            See the module docstring for the tradeoffs before enabling this
            for a full training run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if strict:
        # Required by torch.use_deterministic_algorithms for some CUBLAS ops.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def create_generator(seed: int) -> torch.Generator:
    """Create a seeded `torch.Generator` for reproducible DataLoader shuffling.

    Example:
        generator = create_generator(seed)
        loader = DataLoader(dataset, batch_size=32, shuffle=True,
                             worker_init_fn=seed_worker, generator=generator)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for `DataLoader`: seeds each worker's `random`/`numpy` RNGs.

    PyTorch already seeds each worker's own `torch` RNG deterministically
    (base_seed + worker_id) when a seeded `generator` is passed to the
    DataLoader; this function only needs to propagate that same seed to the
    stdlib `random` and `numpy` RNGs, which PyTorch does not seed on its own
    and which augmentation libraries (e.g. Albumentations) commonly rely on.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
