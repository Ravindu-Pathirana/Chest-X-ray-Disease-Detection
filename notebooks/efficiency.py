"""
T34 - Model Efficiency Benchmark Harness

Measures:
- Total parameters
- Trainable parameters
- FLOPs / GFLOPs (batch size 1)
- Serialized model state_dict size (MB)
- Optional existing checkpoint file size (MB)
- CPU inference latency (ms/image)
- GPU inference latency (ms/image)
- CPU throughput (images/s)
- GPU throughput (images/s)

Designed for 224x224 chest X-ray classification models:
DenseNet121, ResNet50, EfficientNet-B0, ViT-Base,
with or without the selected shortcut-suppression module.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Basic model statistics
# ---------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
    }


def get_state_dict_size_mb(model: nn.Module) -> float:
    """
    Measure serialized model.state_dict() size on disk in MB.
    This gives a fair architecture-level size comparison independent
    of optimizer/scheduler states.
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name

    try:
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 ** 2)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return float(size_mb)


def get_checkpoint_size_mb(checkpoint_path: Optional[str | Path]) -> Optional[float]:
    """Return actual checkpoint file size in MB when a checkpoint is supplied."""
    if checkpoint_path is None:
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        return None

    return float(path.stat().st_size / (1024 ** 2))


# ---------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------

def compute_flops(
    model: nn.Module,
    input_shape=(1, 3, 224, 224),
) -> Dict[str, Optional[float]]:
    """
    Compute FLOPs using fvcore.

    Returns both raw FLOPs and GFLOPs.
    If fvcore is not installed or the model contains unsupported operations,
    the values are returned as None instead of stopping the benchmark.

    Install once with:
        pip install fvcore
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        device = next(model.parameters()).device
        dummy = torch.randn(*input_shape, device=device)

        model.eval()
        with torch.no_grad():
            flop_analysis = FlopCountAnalysis(model, dummy)
            total_flops = float(flop_analysis.total())

        return {
            "flops": total_flops,
            "gflops": total_flops / 1e9,
        }

    except Exception as exc:
        print(f"[FLOPs warning] Could not calculate FLOPs: {exc}")
        return {
            "flops": None,
            "gflops": None,
        }


# ---------------------------------------------------------------------
# Latency / throughput
# ---------------------------------------------------------------------

def _prepare_model_copy(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Move model to requested device.
    The caller should pass a separately constructed model if preserving
    the original device placement is important.
    """
    return model.to(device).eval()


def measure_cpu_latency(
    model: nn.Module,
    input_shape=(1, 3, 224, 224),
    warmup_runs: int = 10,
    measured_runs: int = 50,
) -> Dict[str, float]:
    """
    Measure CPU inference latency for batch size 1.

    Returns mean latency in milliseconds and throughput in images/sec.
    """
    device = torch.device("cpu")
    model = _prepare_model_copy(model, device)
    x = torch.randn(*input_shape, device=device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(x)

        timings = []
        for _ in range(measured_runs):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            timings.append(end - start)

    mean_seconds = sum(timings) / len(timings)

    return {
        "cpu_latency_ms": mean_seconds * 1000.0,
        "cpu_throughput_img_s": input_shape[0] / mean_seconds,
    }


def measure_gpu_latency(
    model: nn.Module,
    input_shape=(1, 3, 224, 224),
    warmup_runs: int = 20,
    measured_runs: int = 100,
) -> Dict[str, Optional[float]]:
    """
    Measure CUDA inference latency for batch size 1.

    torch.cuda.synchronize() is used around timing to avoid reporting
    asynchronous kernel-launch time rather than true inference time.
    """
    if not torch.cuda.is_available():
        return {
            "gpu_latency_ms": None,
            "gpu_throughput_img_s": None,
        }

    device = torch.device("cuda")
    model = _prepare_model_copy(model, device)
    x = torch.randn(*input_shape, device=device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(x)

        torch.cuda.synchronize()

        timings = []
        for _ in range(measured_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x)

            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

    mean_seconds = sum(timings) / len(timings)

    return {
        "gpu_latency_ms": mean_seconds * 1000.0,
        "gpu_throughput_img_s": input_shape[0] / mean_seconds,
    }


# ---------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------

def benchmark_model(
    model: nn.Module,
    model_name: str,
    module_status: str = "without_module",
    checkpoint_path: Optional[str | Path] = None,
    input_shape=(1, 3, 224, 224),
    cpu_warmup: int = 10,
    cpu_runs: int = 50,
    gpu_warmup: int = 20,
    gpu_runs: int = 100,
    output_csv: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Benchmark one model using a standardized 224x224 batch-1 protocol.

    Parameters
    ----------
    model:
        Fully constructed model. Load trained weights beforehand if desired.
        Efficiency numbers do not depend on classification accuracy.
    model_name:
        Example: "EfficientNet-B0"
    module_status:
        Example: "without_module" or "with_module"
    checkpoint_path:
        Optional path used only to report actual checkpoint file size.
    output_csv:
        Optional CSV path. If it exists, the new result is appended.

    Returns
    -------
    dict
        One benchmark result row.
    """
    print("\n" + "=" * 80)
    print(f"Efficiency benchmark: {model_name} [{module_status}]")
    print("=" * 80)

    # Start on CPU so architecture size/statistics are consistent.
    model = model.cpu().eval()

    stats = count_parameters(model)
    state_dict_mb = get_state_dict_size_mb(model)
    checkpoint_mb = get_checkpoint_size_mb(checkpoint_path)

    # FLOPs on CPU to avoid unnecessary GPU allocation.
    flop_stats = compute_flops(model, input_shape=input_shape)

    print("Measuring CPU latency...")
    cpu_stats = measure_cpu_latency(
        model,
        input_shape=input_shape,
        warmup_runs=cpu_warmup,
        measured_runs=cpu_runs,
    )

    print("Measuring GPU latency...")
    gpu_stats = measure_gpu_latency(
        model,
        input_shape=input_shape,
        warmup_runs=gpu_warmup,
        measured_runs=gpu_runs,
    )

    result = {
        "model": model_name,
        "module": module_status,
        "input_shape": "x".join(map(str, input_shape)),
        **stats,
        **flop_stats,
        "state_dict_size_mb": state_dict_mb,
        "checkpoint_size_mb": checkpoint_mb,
        **cpu_stats,
        **gpu_stats,
    }

    print("\nResult:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame([result])

        if output_csv.exists():
            old_df = pd.read_csv(output_csv)
            final_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        final_df.to_csv(output_csv, index=False)
        print(f"\nSaved/appended result to: {output_csv}")

    # Free CUDA memory before the next model.
    model = model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------
# Convenience helper for T35
# ---------------------------------------------------------------------

def results_to_table(
    results,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Turn a list of benchmark result dicts into a clean comparison table."""
    df = pd.DataFrame(results)

    preferred_columns = [
        "model",
        "module",
        "params_total",
        "params_trainable",
        "gflops",
        "state_dict_size_mb",
        "checkpoint_size_mb",
        "cpu_latency_ms",
        "gpu_latency_ms",
        "cpu_throughput_img_s",
        "gpu_throughput_img_s",
    ]

    cols = [c for c in preferred_columns if c in df.columns]
    df = df[cols]

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


if __name__ == "__main__":
    print(
        "T34 efficiency harness loaded successfully.\n"
        "Import benchmark_model() from this file and pass a constructed model.\n\n"
        "Example:\n"
        "  from efficiency import benchmark_model\n"
        "  result = benchmark_model(model, 'EfficientNet-B0', 'without_module')"
    )
