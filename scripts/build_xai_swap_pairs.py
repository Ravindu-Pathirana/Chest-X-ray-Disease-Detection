"""Freeze deterministic different-class test-image donor pairs without images."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.xai_dependence import build_swap_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "artifacts/splits/split_manifest_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/explainable_ai/swap_pairs_seed42.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairs = build_swap_pairs(pd.read_csv(args.manifest), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)
    print(f"Wrote {len(pairs)} fixed donor pairs to {args.output}")


if __name__ == "__main__":
    main()
