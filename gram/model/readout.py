"""Apply a read-out head to a batch of windows."""
from __future__ import annotations

import torch

from gram.library.assembly import assemble, prior_curve
from gram.utils import STATE


def correction(amplitude, exc):
    """Turn an amplitude curve into an additive correction."""
    return (amplitude - 1.0) * exc


def forward(model, tables, split, index):
    """The corrected forecast, around the level the series is sitting on."""
    node, adj_shape, adj_bias, query = assemble(tables, split, index)
    amplitude, _, _ = model(node, adj_shape, adj_bias, query)
    STATE["amp"] = amplitude
    STATE["prior"] = prior_curve(tables, split["tag"][index])
    exc = split["exc"][index]
    return split["pred"][index] - exc + amplitude * exc


def predict(models, tables, test, rows, chunk, device):
    """Mean correction of the given heads over a range of windows."""
    pieces = []
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            block = {k: v[rows[start:start + chunk]].to(device)
                     for k, v in test.items()}
            whole = torch.arange(len(block["pred"]), device=device)
            stack = [forward(model, tables, block, whole) for model in models]
            pieces.append((sum(stack) / len(stack) - block["pred"]).cpu())
    return torch.cat(pieces)
