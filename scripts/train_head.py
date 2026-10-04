#!/usr/bin/env python
"""Fit a read-out head on the corpus libraries.

    python scripts/train_head.py --backbone sundial --horizon 96 --seeds 3

Only corpus packs are read: no target labels and no evaluation windows take
part in fitting.  Checkpoints land in ``data/checkpoints`` as
``head_pl{horizon}_{backbone}_{tag}_seed{seed}.pt``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.config import CHECKPOINTS, PRETRAIN                        # noqa: E402
from gram.library.packs import load_packs                            # noqa: E402
from gram.train.engine import train_fold                             # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="sundial")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--tag", default="grag")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--quota", type=int, default=256)
    parser.add_argument("--mae_weight", type=float, default=2.0)
    parser.add_argument("--drift_val", type=float, default=0.3)
    parser.add_argument("--watch_every", type=int, default=100)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--select", default="constrained",
                        choices=("constrained", "safest"))
    parser.add_argument("--refit", action="store_true", default=False)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.pred_len = args.horizon
    args.init = args.frozen = False
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    names, tables, data = load_packs(args.pred_len, device,
                                     directory=PRETRAIN)
    cal = data["cal"]
    print(f"=== read-out fit  {args.backbone}@{args.horizon}  {device}")
    print(f"    {len(names)} corpus libraries, {len(cal['pred'])} windows")
    for seed in range(args.seeds):
        out = CHECKPOINTS / (f"head_pl{args.horizon}_{args.backbone}"
                             f"_{args.tag}_seed{seed}.pt")
        if out.is_file() and not args.force:
            print(f"  seed{seed}: {out.name} exists, skipped")
            continue
        model = train_fold(tables, cal, list(range(len(names))), args,
                           "worst_cell", 2000 + seed, device)
        torch.save(model.state_dict(), out)
        print(f"  seed{seed} -> {out}")


if __name__ == "__main__":
    main()
