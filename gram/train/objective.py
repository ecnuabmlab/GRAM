"""Training objective and the mark that picks a training step."""
from __future__ import annotations

import torch

from gram.data.series import score
from gram.model.readout import predict
from gram.utils import STATE


def objective_metric(kind, final, pred, true, cell, n_cells, mae_weight):
    """MSE + mae_weight * MAE of the corrected forecast, plus |a - prior|."""
    err = final - true
    base = err.pow(2).mean() + mae_weight * err.abs().mean()
    prior = STATE.get("prior")
    if prior is not None:
        base = base + (STATE["amp"] - prior).abs().mean()
    return base


def cell_ratios(final, pred, true, cell, n_cells):
    """Per-cell squared and absolute error, each against the bare forecast."""
    def pool(x):
        return torch.zeros(n_cells, device=x.device).index_add_(0, cell, x)
    err, bare = final - true, pred - true
    return (pool(err.pow(2).mean(1)) / pool(bare.pow(2).mean(1)).clamp_min(1e-9),
            pool(err.abs().mean(1)) / pool(bare.abs().mean(1)).clamp_min(1e-9))


def target_watch_mark(model, watches, device, chunk, margin, select):
    """Constrained mark on evaluation cal cells, mirroring the corpus tail."""
    mse_g, mae_g = [], []
    for _, tables, cal in watches:
        rows = torch.arange(len(cal["pred"]))
        corr = predict([model], tables, cal, rows, chunk, device)
        mse, mae = score(cal["true"], cal["pred"], corr)
        mse_g.append(mse / 100.0)
        mae_g.append(mae / 100.0)
    mse_g = torch.tensor(mse_g)
    mae_g = torch.tensor(mae_g)
    weaker = torch.minimum(mse_g, mae_g)
    worst = float(weaker.min())
    average = float((mse_g + mae_g).mean() / 2.0)
    dual = int(((mse_g > 0) & (mae_g > 0)).sum())
    mark = worst if select == "safest" else (
        average if worst > margin else worst - 1e3)
    return mark, average, worst, dual, len(watches)
