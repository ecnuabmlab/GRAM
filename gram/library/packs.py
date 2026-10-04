"""Retrieval packs: prototype tables plus the calibration and test splits."""
from __future__ import annotations

import numpy as np
import torch

from gram.config import BACKBONES, CHECKPOINTS, GRID_DATASETS, PACKS
from gram.data.series import load_targets, standardized_series, trailing_mean
from gram.library.clustering import production_memory
from gram.library.graph import (anchor_amplitude, own_graph,
                                seeds_and_neighbours)
from gram.library.memory import pack_is_current, stamp_geom

NODE_KEYS = ("amp", "spread", "support", "coherence", "foreign")
SPLIT_KEYS = ("nodes", "sims", "anchor", "pred", "true", "exc", "scale")


def build_packs(args):
    PACKS.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    print("  production graph: K={0} span={1} top_k={2} expand={3} "
          "deg={4}; target-only, corpus prior in the anchor".format(
              args.n_clusters, args.span, args.top_k, args.k_expand,
              args.k_shape), flush=True)

    datasets = getattr(args, "dataset_list", list(GRID_DATASETS))
    backbones = getattr(args, "backbone_list", list(BACKBONES))
    for dataset in datasets:
        series = standardized_series(dataset)
        for backbone in backbones:
            out = PACKS / f"{backbone}@{dataset}@pl{args.pred_len}.npz"
            if out.is_file() and pack_is_current(out) and not args.force:
                continue
            memory = production_memory(
                dataset, backbone, args.pred_len, series,
                n_clusters=args.n_clusters, span=args.span)


            targets = load_targets(dataset, args.pred_len, backbone)
            if len(memory["emb"]) != len(targets["pred"]):
                print(f"  skipping {backbone}/{dataset}: "
                      f"{len(memory['emb'])} embeddings against "
                      f"{len(targets['pred'])} forecasts", flush=True)
                continue
            if out.is_file():
                out.unlink()
            graph = own_graph(memory, device, k_shape=args.k_shape,
                              k_bias=args.k_bias, k_expand=args.k_expand)
            blob = {k: graph[k].cpu().numpy() for k in NODE_KEYS}
            blob["rel_shape"] = graph["rel_shape"].cpu().numpy()
            blob["rel_bias"] = graph["rel_bias"].cpu().numpy()
            for split, emb_key, ch, ti, pk, tk in (
                    ("cal", "cal_emb", "cal_channel_ids", "cal_time_indices",
                     "cal_pred", "cal_true"),
                    ("test", "emb", "channel_ids", "time_indices",
                     "pred", "true")):
                level = trailing_mean(targets[ch], targets[ti], series,
                                      args.span)
                pred = np.asarray(targets[pk], np.float32)
                exc = pred - level[:, None]
                nodes, sims = seeds_and_neighbours(graph, memory[emb_key],
                                                   args.top_k, device)
                blob[f"{split}_nodes"] = nodes.cpu().numpy().astype(np.int16)
                blob[f"{split}_sims"] = sims.cpu().numpy()
                blob[f"{split}_anchor"] = anchor_amplitude(
                    memory, memory[emb_key], args.top_k, device).cpu().numpy()
                blob[f"{split}_pred"] = pred
                blob[f"{split}_true"] = np.asarray(targets[tk], np.float32)
                blob[f"{split}_exc"] = exc.astype(np.float32)
                blob[f"{split}_scale"] = np.abs(exc).mean(
                    1, keepdims=True).clip(1e-3).astype(np.float32)
            np.savez(out, **stamp_geom(blob))
            print(f"  packed {backbone}/{dataset}: {nodes.shape[1]} graph "
                  f"nodes, {len(memory['centers'])} protos", flush=True)


def load_packs(pred_len, device, directory=None):
    """All cells in one set of tables, addressed by a per-window cell tag."""
    directory = PACKS if directory is None else directory
    paths = sorted(directory.glob(f"*@pl{pred_len}.npz"))
    if not paths:
        raise SystemExit(f"no packs under {directory}; run scripts/build_library.py first")
    names, nodes, shape, bias = [], {k: [] for k in NODE_KEYS}, [], []
    reference = []
    splits = {s: {k: [] for k in SPLIT_KEYS} for s in ("cal", "test")}
    tags = {"cal": [], "test": []}
    per_cell = None
    for index, path in enumerate(paths):
        backbone, dataset, _ = path.stem.split("@")
        names.append(f"{backbone}/{dataset}")
        with np.load(path) as z:
            for key in NODE_KEYS:
                nodes[key].append(torch.as_tensor(z[key]))
            shape.append(torch.as_tensor(z["rel_shape"]))
            bias.append(torch.as_tensor(z["rel_bias"]))
            per_cell = len(z["support"])
            reference.append(float(np.median(z["support"])))
            for split in ("cal", "test"):
                for key in SPLIT_KEYS:
                    value = torch.as_tensor(z[f"{split}_{key}"])
                    if key == "nodes":
                        value = value.long() + index * per_cell
                    splits[split][key].append(value)
                tags[split].append(torch.full((len(z[f"{split}_pred"]),),
                                              index, dtype=torch.long))
    tables = {k: torch.cat(v).to(device) for k, v in nodes.items()}
    tables["rel_shape"] = torch.stack(shape).to(device)
    tables["rel_bias"] = torch.stack(bias).to(device)
    tables["per_cell"] = per_cell







    tables["support_ref"] = torch.tensor(reference, device=device)
    data = {}
    for split in ("cal", "test"):


        home = device if split == "cal" else torch.device("cpu")
        data[split] = {k: torch.cat(v).to(home)
                       for k, v in splits[split].items()}
        data[split]["tag"] = torch.cat(tags[split]).to(home)
    return names, tables, data


def head_paths(pred_len, backbone=None, tag=None):
    if backbone and tag:
        named = sorted(CHECKPOINTS.glob(
            f"head_pl{pred_len}_{backbone}_{tag}_seed*.pt"))
        if named:
            return named
    if backbone:
        named = sorted(CHECKPOINTS.glob(f"head_pl{pred_len}_{backbone}_seed*.pt"))
        if named:
            return named
    named = sorted(CHECKPOINTS.glob(f"head_pl{pred_len}_seed*.pt"))
    if named:
        return named


    if pred_len == 96:
        return sorted(CHECKPOINTS.glob("head_seed*.pt"))
    return []


def load_one_pack(path, device, split="test"):
    """One cell's tables and one split, so electricity never shares a batch."""
    backbone, dataset, _ = path.stem.split("@")
    name = f"{backbone}/{dataset}"
    with np.load(path) as z:
        tables = {k: torch.as_tensor(z[k], device=device) for k in NODE_KEYS}
        tables["rel_shape"] = torch.as_tensor(z["rel_shape"], device=device)[None]
        tables["rel_bias"] = torch.as_tensor(z["rel_bias"], device=device)[None]
        tables["per_cell"] = len(z["support"])
        tables["support_ref"] = torch.tensor(
            [float(np.median(z["support"]))], device=device)
        loaded = {k: torch.as_tensor(z[f"{split}_{k}"]) for k in SPLIT_KEYS}
        loaded["nodes"] = loaded["nodes"].long()
        loaded["tag"] = torch.zeros(len(loaded["pred"]), dtype=torch.long)
    return name, tables, loaded
