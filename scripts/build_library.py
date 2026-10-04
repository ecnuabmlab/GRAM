#!/usr/bin/env python
"""Build the pattern library and retrieval packs of target cells.

    python scripts/build_library.py --backbone sundial --datasets ETTh1 \
        --horizon 96 --gpu 0

Requires the prototype memory in ``data/knowledge``, the frozen forecasts in
``data/forecasts`` and the raw series in ``data/series``; see data/README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.config import (BACKBONES, GRID_DATASETS, K_BIAS, K_EXPAND,  # noqa: E402
                         K_SHAPE, N_CLUSTERS, SPAN, TOP_K, parse_names)
from gram.library.packs import build_packs                            # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="sundial")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--n_clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--span", type=int, default=SPAN)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--k_expand", type=int, default=K_EXPAND)
    parser.add_argument("--k_shape", type=int, default=K_SHAPE)
    parser.add_argument("--k_bias", type=int, default=K_BIAS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.pred_len = args.horizon
    args.dataset_list = parse_names(args.datasets, GRID_DATASETS)
    args.backbone_list = parse_names(args.backbone, BACKBONES)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"=== library build  pl{args.horizon}  "
          f"{len(args.dataset_list)} datasets x {len(args.backbone_list)} "
          f"backbones ===")
    build_packs(args)


if __name__ == "__main__":
    main()
