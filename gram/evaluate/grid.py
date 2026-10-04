"""Frozen evaluation over the target grid, and the merge into one table."""
from __future__ import annotations

import json

import numpy as np
import torch

from gram.config import PACKS, RESULTS
from gram.data.series import score
from gram.evaluate.metrics import abs_pair
from gram.library.packs import load_one_pack
from gram.model.readout import predict
from gram.train.engine import load_pretrained
from gram.utils import write_json


def eval_frozen_per_cell(args, device):
    """Frozen eval one pack at a time, flushed after every cell."""
    heads_by_bb = {}
    out = RESULTS / f"{args.tag}.json"
    blob = {"cells": {}, "summary": {}, "ceiling": {}, "config": vars(args)}
    if out.is_file() and not args.force:
        blob = json.loads(out.read_text())
        blob.setdefault("cells", {})
    wanted = [f"{bb}/{ds}" for ds in args.dataset_list
              for bb in args.backbone_list]
    for name in wanted:
        backbone, dataset = name.split("/", 1)
        pack = PACKS / f"{backbone}@{dataset}@pl{args.pred_len}.npz"
        if name in blob["cells"] and "graft" in blob["cells"][name] and not args.force:
            print(f"  {name:22s} skip (already in {out.name})", flush=True)
            continue
        if not pack.is_file():
            print(f"  {name:24s} missing pack {pack.name}", flush=True)
            continue
        if backbone not in heads_by_bb:
            heads_by_bb[backbone] = load_pretrained(args, device, backbone)
        heads = heads_by_bb[backbone]
        cell_name, tables, test = load_one_pack(pack, device)
        rows = torch.arange(len(test["pred"]))
        piece = predict(heads, tables, test, rows, args.chunk, device)
        true, bare = test["true"][rows], test["pred"][rows]
        corrected = bare + piece
        cells = blob.setdefault("cells", {})
        cells[cell_name] = {
            "bare": abs_pair(true, bare),
            "graft_abs": abs_pair(true, corrected),
            "anchor_0.7": score(true, bare,
                                0.7 * (test["anchor"] - 1.0) * test["exc"]),
            "graft": score(true, bare, piece),
        }
        row = cells[cell_name]
        print(f"  {cell_name:22s} MSE {row['bare'][0]:.4f} -> "
              f"{row['graft_abs'][0]:.4f}   MAE {row['bare'][1]:.4f} -> "
              f"{row['graft_abs'][1]:.4f}", flush=True)
        blob["summary"] = {
            "anchor_0.7": _summary_dict(cells, "anchor_0.7"),
            "graft": _summary_dict(cells, "graft"),
        }
        if heads:
            blob["ceiling"] = heads[0].report_ceiling()
        write_json(out, blob)
        del tables, test, piece
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not blob.get("cells"):
        raise SystemExit(f"no cells evaluated; build packs first under {PACKS}")
    write_json(out, blob)
    print(f"saved -> {out}")
    return blob


def _summary_dict(cells, key):
    mse = np.array([cells[k][key][0] for k in cells])
    mae = np.array([cells[k][key][1] for k in cells])
    dual = int(((mse > 0) & (mae > 0)).sum())
    return {"mse": float(mse.mean()), "mae": float(mae.mean()), "dual": dual,
            "worst_mse": float(mse.min()), "worst_mae": float(mae.min()),
            "n": int(len(cells))}
