"""Fit a read-out head on the corpus libraries, and load fitted heads."""
from __future__ import annotations

import torch
import torch.nn as nn

import argparse

from gram.config import (CHECKPOINTS, DATASETS, PACKS, PRETRAIN,
                         parse_names)
from gram.library.packs import SPLIT_KEYS, head_paths, load_one_pack
from gram.model.fusion import GraphFusion
from gram.model.readout import correction, forward, predict
from gram.train.objective import (cell_ratios, objective_metric,
                                  target_watch_mark)


def load_target_watch(pred_len, backbone, device, datasets=DATASETS, cap=2048):
    """Calibration windows of the evaluation cells, for pretrain step picking."""
    items = []
    for dataset in datasets:
        path = PACKS / f"{backbone}@{dataset}@pl{pred_len}.npz"
        if not path.is_file():
            print(f"  target-watch skip {backbone}/{dataset}: no pack",
                  flush=True)
            continue
        name, tables, cal = load_one_pack(path, device, split="cal")
        if cap and len(cal["pred"]) > cap:


            keep = torch.arange(len(cal["pred"]) - cap, len(cal["pred"]))
            cal = {k: v[keep] for k, v in cal.items()}
        items.append((name, tables, cal))
        print(f"  target-watch {name}: {len(cal['pred'])} cal windows",
              flush=True)
    if not items:
        raise SystemExit(f"no target packs for {backbone}/pl{pred_len}")
    return items


def train_fold(tables, cal, train_cells, args, kind, seed, device, init=None,
               target_watch=None):
    torch.manual_seed(seed)
    model = GraphFusion(args.pred_len, width=args.width,
                        dropout=args.dropout).to(device)
    if init is not None:
        model.load_state_dict(init)
    ceiling = {"ceiling_gain", "ceiling_bias"}



    base = args.lr if init is None else args.finetune_lr
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters()
                     if n not in ceiling]},


         {"params": [p for n, p in model.named_parameters() if n in ceiling],
          "lr": 2e-2 if init is None else 2e-3, "weight_decay": 0.0}],
        lr=base, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    pools = [torch.nonzero(cal["tag"] == c).squeeze(1) for c in train_cells]



    watch_on_target = target_watch is not None
    if args.drift_val and not watch_on_target:
        cut = [int(len(pool) * (1.0 - args.drift_val)) for pool in pools]
        tail = [pool[c:] for pool, c in zip(pools, cut)]
        pools = [pool[:c] for pool, c in zip(pools, cut)]
        watch = torch.cat([pool[:args.quota * 4] for pool in tail])
        watch_tag = torch.cat([
            torch.full((len(pool[:args.quota * 4]),), i, device=device)
            for i, pool in enumerate(tail)])

    tag = torch.arange(len(train_cells),
                       device=device).repeat_interleave(args.quota)
    best, best_step, best_state = -1e9, args.steps, None
    watching = watch_on_target or args.drift_val
    for step in range(args.steps):
        opt.zero_grad()
        index = torch.cat([pool[torch.randint(0, len(pool), (args.quota,),
                                              device=device)]
                           for pool in pools])
        final = forward(model, tables, cal, index)
        objective_metric(kind, final, cal["pred"][index], cal["true"][index], tag,
           len(train_cells), args.mae_weight).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if watching and (step + 1) % args.watch_every == 0:
            if watch_on_target:
                with torch.no_grad():
                    mark, average, worst, dual, n_watch = target_watch_mark(
                        model, target_watch, device, args.chunk, args.margin,
                        args.select)
                if (step + 1) % (args.watch_every * 4) == 0:
                    print(f"    step {step + 1}: target-cal "
                          f"{average:+.3f} worst {worst:+.3f} "
                          f"dual {dual}/{n_watch}", flush=True)
            else:
                with torch.no_grad():
                    ahead = forward(model, tables, cal, watch)
                    mse_ratio, mae_ratio = cell_ratios(
                        ahead, cal["pred"][watch], cal["true"][watch],
                        watch_tag, len(train_cells))
                weaker = torch.minimum(1.0 - mse_ratio, 1.0 - mae_ratio)
                worst = float(weaker.min())
                if args.select == "safest":
                    mark = worst
                else:
                    average = float(((1.0 - mse_ratio) + (1.0 - mae_ratio)).mean()
                                    / 2.0)
                    mark = average if worst > args.margin else worst - 1e3
            if mark > best:
                best, best_step = mark, step + 1
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}

    if best_state is None:
        model.eval()
        return model
    if watch_on_target:
        print(f"  picked step {best_step} on target-cal (mark {best:+.3f})",
              flush=True)
    if not args.refit:
        model.load_state_dict(best_state)
        model.eval()
        return model



    again = argparse.Namespace(**vars(args))
    again.drift_val = 0.0
    again.steps = best_step
    return train_fold(tables, cal, train_cells, again, kind, seed + 7, device,
                      init)


def load_pretrained(args, device, backbone=None):
    """The heads LOTSA left behind, or nothing if the fit starts from scratch."""
    if not (args.init or args.frozen):
        return None
    paths = head_paths(args.pred_len, backbone,
                       tag=getattr(args, "head_tag", None))
    if not paths:
        raise SystemExit(f"no pretrained heads for pl{args.pred_len}"
                         f"{'' if not backbone else ' / ' + backbone} under "
                         f"{CHECKPOINTS}; run scripts/train_head.py first")
    heads = []
    for path in paths:
        model = GraphFusion(args.pred_len, width=args.width,
                            dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        heads.append(model)
    label = backbone or "shared"
    print(f"  loaded {len(heads)} LOTSA-pretrained heads ({label})",
          flush=True)
    return heads
