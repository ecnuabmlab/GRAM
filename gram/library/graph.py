"""Edges over the prototype library, and the retrieval that walks them."""
from __future__ import annotations

import numpy as np
import torch

from gram.config import K_BIAS, K_EXPAND, K_SHAPE, MIN_SUPPORT
from gram.library.memory import load_pack, stamp_geom, unit


def row_topk(affinity, k, allowed=None):
    """Keep each node's k strongest links, renormalised to sum to one."""
    a = affinity.clone()
    a.fill_diagonal_(-1e9)
    if allowed is not None:
        a = a.masked_fill(~allowed[None, :], -1e9)
    value, index = a.topk(k, dim=1)
    out = torch.zeros_like(a)
    out.scatter_(1, index, value.clamp_min(0.0))
    return out / out.sum(1, keepdim=True).clamp_min(1e-8), index
def refresh_pack_anchors(path, memory, queries, top_k, device):
    """Rewrite only the stored anchors, then stamp the current geometry."""
    blob = load_pack(path)
    for split, query in queries.items():
        blob[f"{split}_anchor"] = anchor_amplitude(
            memory, query, top_k, device).cpu().numpy()
    np.savez(path, **stamp_geom(blob))
def own_graph(own, device, k_shape=K_SHAPE, k_bias=K_BIAS, k_expand=K_EXPAND,
              center=1.0):
    """Target library only: no LOTSA nodes, no foreign flag, no borrowed edges."""
    centers = unit(torch.as_tensor(own["centers"], device=device))
    amplitude = torch.as_tensor(own["amplitude"], device=device)
    support = torch.as_tensor(own["support"], device=device)
    spread = torch.as_tensor(own["spread"], device=device)
    coherence = torch.as_tensor(own["coherence"], device=device)
    n_own = len(own["centers"])
    shape = centers @ centers.T
    centred = amplitude - center
    direction = centred / centred.norm(dim=1, keepdim=True).clamp_min(1e-6)
    bias = direction @ direction.T
    solid = support >= MIN_SUPPORT
    rel_shape, _ = row_topk(shape, k_shape)
    rel_bias, _ = row_topk(bias, k_bias, allowed=solid)
    _, near_shape = row_topk(shape, k_expand)
    _, near_bias = row_topk(bias, k_expand, allowed=solid)
    return {
        "centers": centers, "amp": amplitude, "spread": spread,
        "support": support, "coherence": coherence,
        "foreign": torch.zeros(n_own, device=device),
        "n_own": n_own, "solid": solid,
        "rel_shape": rel_shape, "rel_bias": rel_bias,
        "neighbours": torch.cat([near_shape, near_bias], dim=1),
    }


def edge_reliability(memory):
    """Does bias adjacency built on early leads survive on late leads?"""
    solid = memory["support"] >= MIN_SUPPORT
    centred = memory["amplitude"][solid] - 1.0
    half = centred.shape[1] // 2
    normalise = lambda x: x / np.linalg.norm(x, axis=1,
                                             keepdims=True).clip(1e-6)
    left, right = normalise(centred[:, :half]), normalise(centred[:, half:])
    return float(np.corrcoef((left @ left.T).ravel(),
                             (right @ right.T).ravel())[0, 1])


def seeds_and_neighbours(graph, query, top_k, device, chunk=16384):
    """Top-k shape seeds in the target library, expanded along both relations."""
    own = graph["centers"][:graph["n_own"]]
    picked, sims = [], []
    n_query = len(query)
    for start in range(0, n_query, chunk):
        block = unit(torch.as_tensor(query[start:start + chunk], device=device))
        _, seed = (block @ own.T).topk(top_k, dim=1)
        nodes = torch.cat([seed, graph["neighbours"][seed].flatten(1)], dim=1)
        picked.append(nodes)
        sims.append((block[:, None, :] * graph["centers"][nodes]).sum(-1))
    return torch.cat(picked), torch.cat(sims)


def anchor_amplitude(memory, query, top_k, device, chunk=16384):
    """The pooled retrieval answer, which joins the subgraph as one node.
"""
    centers = unit(torch.as_tensor(memory["centers"], device=device))
    amplitude = torch.as_tensor(memory["node_amplitude"], device=device)
    coherence = torch.as_tensor(memory["node_coherence"], device=device)
    out = []
    n_query = len(query)
    for start in range(0, n_query, chunk):
        q = unit(torch.as_tensor(query[start:start + chunk], device=device))
        sims, picked = (q @ centers.T).topk(top_k, dim=1)
        corpus = torch.full((len(q), 1), len(centers), device=device,
                            dtype=picked.dtype)
        picked = torch.cat([picked, corpus], dim=1)
        sims = torch.cat([sims, sims.mean(1, keepdim=True)], dim=1)
        vote = torch.softmax(sims / 0.10, dim=1) * coherence[picked].clamp_min(1e-3)
        vote = vote / vote.sum(1, keepdim=True).clamp_min(1e-8)
        out.append((vote[:, :, None] * amplitude[picked]).sum(1))
    return torch.cat(out)
