"""Assemble the read-out input of a batch of windows."""
from __future__ import annotations

import torch
import torch.nn as nn

from gram.config import MIN_SUPPORT



def prior_curve(tables, tag):
    """Evidence weighted mean of the library's amplitude curves."""
    per = tables["per_cell"]
    n_cells = len(tables["support"]) // per
    amp = tables["amp"].view(n_cells, per, -1)
    sup = tables["support"].view(n_cells, per)
    return ((sup[:, :, None] * amp).sum(1)
            / sup.sum(1).clamp_min(1e-6))[tag.long()]


def assemble(tables, split, index):
    """Node features and adjacency for one batch, built on demand."""
    nodes = split["nodes"][index]
    sims = split["sims"][index]
    anchor = split["anchor"][index]
    tag = split["tag"][index]
    reference = tables["support_ref"][tag][:, None]
    extras = torch.stack([
        torch.log1p(tables["support"][nodes]) - torch.log1p(reference),
        tables["coherence"][nodes],
        tables["foreign"][nodes],
        sims,
        (tables["support"][nodes] >= MIN_SUPPORT).float(),
    ], dim=-1)
    node = torch.cat([tables["amp"][nodes] - 1.0, tables["spread"][nodes],
                      extras], dim=-1)




    ones = torch.ones(len(anchor), 1, device=anchor.device)
    summary = torch.cat([anchor - 1.0, torch.zeros_like(anchor), ones, ones,
                         torch.zeros_like(ones), ones, ones], dim=1)
    node = torch.cat([node, summary[:, None]], dim=1)

    local = nodes % tables["per_cell"]
    rows = local[:, :, None].expand(-1, -1, local.shape[1])
    cols = local[:, None, :].expand(-1, local.shape[1], -1)
    cell = tag[:, None, None].expand_as(rows)
    pad = (0, 1, 0, 1)
    adj_shape = nn.functional.pad(tables["rel_shape"][cell, rows, cols], pad)
    adj_bias = nn.functional.pad(tables["rel_bias"][cell, rows, cols], pad)
    eye = torch.eye(node.shape[1], device=node.device)[None]

    exc, scale = split["exc"][index], split["scale"][index]
    normalised = exc / scale
    query = torch.cat([normalised, torch.log(scale),
                       normalised.std(1, keepdim=True),
                       normalised.abs().amax(1, keepdim=True)], dim=1)
    query = torch.cat([query, prior_curve(tables, tag)], dim=1)
    return node, adj_shape + eye, adj_bias + eye, query
