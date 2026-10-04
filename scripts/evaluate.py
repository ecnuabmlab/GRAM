#!/usr/bin/env python
"""Apply the frozen read-out to target cells and report relative gains.

    python scripts/evaluate.py --backbone sundial --horizon 96 --gpu 0

Every cell writes one row into ``results/<tag>.json``: the gain of the pooled
retrieval answer alone and of the full read-out, both relative to the frozen
backbone forecast on the same windows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.config import GRID_DATASETS, RESULTS, parse_names         # noqa
from gram.evaluate.grid import eval_frozen_per_cell                 # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="sundial")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--head_tag", default=None)
    parser.add_argument("--tag", default="frozen")
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.pred_len = args.horizon
    args.horizons = (args.horizon,)
    args.dataset_list = parse_names(args.datasets, GRID_DATASETS)
    args.backbone_list = parse_names(args.backbone, ("sundial",))
    args.frozen = args.init = True
    args.width, args.dropout = 96, 0.2
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"=== frozen read-out  {args.backbone}@{args.horizon}  {device}")
    eval_frozen_per_cell(args, device)


if __name__ == "__main__":
    main()
