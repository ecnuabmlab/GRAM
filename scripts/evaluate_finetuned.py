#!/usr/bin/env python
"""Apply the frozen read-out on top of a fine-tuned backbone.

    python scripts/evaluate_finetuned.py --horizon 96 --library data/library_ft

The library must be built from the fine-tuned backbone, so ``pred`` holds the
fine-tuned forecast; the read-out itself stays exactly as fitted on the corpus.
Each cell reports the gain over the fine-tuned forecast on the same windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.config import (BACKBONES, CHECKPOINTS, GRID_DATASETS,       # noqa: E402
                         RESULTS, parse_names)
from gram.evaluate.metrics import abs_pair                            # noqa: E402
from gram.model.fusion import GraphFusion                             # noqa: E402
from gram.model.readout import predict                                # noqa: E402
from gram.library.packs import load_one_pack                          # noqa: E402


def load_heads(horizon, backbone, width, dropout, device):
    files = sorted(CHECKPOINTS.glob(f"head_pl{horizon}_{backbone}_seed*.pt"))
    heads = []
    for path in files:
        model = GraphFusion(horizon, width=width, dropout=dropout).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        heads.append(model)
    return heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=str(Path("data/library_ft")))
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--backbones", default=None)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tag", default="finetuned")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    datasets = parse_names(args.datasets, GRID_DATASETS)
    backbones = parse_names(args.backbones, BACKBONES)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    library = Path(args.library)
    cells = {}
    for dataset in datasets:
        for backbone in backbones:
            pack = library / f"{backbone}@{dataset}@pl{args.horizon}.npz"
            if not pack.is_file():
                continue
            heads = load_heads(args.horizon, backbone, args.width,
                               args.dropout, device)
            if not heads:
                print(f"  {backbone}/{dataset}: no checkpoint")
                continue
            _, tables, test = load_one_pack(pack, device)
            rows = torch.arange(len(test["pred"]))
            true, pred = test["true"][rows], test["pred"][rows]
            corrected = pred + predict(heads, tables, test, rows, 8192, device)
            cell = {"ft_abs": abs_pair(true, pred),
                    "grag_abs": abs_pair(true, corrected)}
            cells[f"{backbone}/{dataset}"] = cell
            print(f"  {backbone}/{dataset:20s} MSE {cell['ft_abs'][0]:.4f} -> "
                  f"{cell['grag_abs'][0]:.4f}   MAE {cell['ft_abs'][1]:.4f} -> "
                  f"{cell['grag_abs'][1]:.4f}", flush=True)
    if not cells:
        raise SystemExit(f"no cells under {library}")
    out = RESULTS / f"{args.tag}_pl{args.horizon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cells": cells}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
