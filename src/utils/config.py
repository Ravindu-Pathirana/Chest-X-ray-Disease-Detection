"""Utilities for loading and overriding YAML experiment configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Union


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML experiment config file into a plain nested dict.

    Args:
        path: Path to a YAML file, e.g. "configs/baseline.yaml".

    Returns:
        The parsed configuration as a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file does not parse to a dict (e.g. empty file).
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file did not parse to a dictionary: {path}")

    return config


def merge_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `config` with dotted-key overrides applied.

    Useful for tweaking a couple of values in a notebook cell without
    creating a new YAML file.

    Example:
        merge_overrides(config, {"training.batch_size": 64, "model.name": "densenet121"})

    Args:
        config: The base configuration dict (not mutated).
        overrides: Mapping of dotted key paths to new values.

    Returns:
        A new dict with the overrides applied.
    """
    merged = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        node = merged
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return merged
