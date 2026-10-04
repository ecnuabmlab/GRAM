"""Prototype curves: what each pattern knows about the forecast error."""
from __future__ import annotations

import numpy as np

from gram.config import (AMPLITUDE, KNOWLEDGE, LEVELS, LOTSA_CACHE,
                         PRECOMPUTE, add_offline_sources)
from gram.data.series import standardized_series, trailing_mean
from gram.config import parse_names



RATIO_CLIP = (-2.0, 4.0)
MIN_WINDOWS = 4


def pool_amplitude(labels, excursion, residual, n_prototypes):
    """Per-prototype amplitude curve, its disagreement, and its support."""
    horizon = excursion.shape[1]
    weight = np.abs(excursion)
    safe = np.where(weight > 1e-6, excursion, 1.0)
    ratio = np.clip(residual / safe, *RATIO_CLIP)
    ratio[weight <= 1e-6] = 1.0

    amplitude = np.ones((n_prototypes, horizon), np.float32)
    spread = np.zeros((n_prototypes, horizon), np.float32)
    support = np.zeros(n_prototypes, np.float32)
    for proto in range(n_prototypes):
        rows = np.flatnonzero(labels == proto)
        support[proto] = len(rows)
        if len(rows) < MIN_WINDOWS:
            continue
        for h in range(horizon):
            w = weight[rows, h]
            total = w.sum()
            if total <= 1e-8:
                continue
            order = np.argsort(ratio[rows, h])
            middle = np.searchsorted(np.cumsum(w[order]), 0.5 * total)
            centre = ratio[rows[order[min(middle, len(rows) - 1)]], h]
            amplitude[proto, h] = centre
            deviation = np.abs(ratio[rows, h] - centre)
            spread[proto, h] = float((w * deviation).sum() / total)
    return amplitude, spread, support


def target_cell(dataset, backbone, pred_len, series, span):
    add_offline_sources()
    from data.loader import load_aligned, temporal_split

    with np.load(KNOWLEDGE / f"{dataset}_pl{pred_len}_{backbone}.npz",
                 allow_pickle=False) as z:
        labels = np.asarray(z["labels"], np.int64)
        n_prototypes = len(z["centers"])
    _, pred, true, channel, time = load_aligned(
        str(PRECOMPUTE), dataset, "train", pred_len, backbone=backbone)





    rows = None
    for embargo in (pred_len, pred_len - 1):
        build, _ = temporal_split(channel, time, 0.8, embargo)
        if int(build.sum()) == len(labels):
            rows = np.flatnonzero(build)
            break
    if rows is None:
        raise SystemExit(
            f"{dataset}/{backbone}/pl{pred_len}: no embargo reproduces the "
            f"{len(labels)} windows in `labels`; refusing to attach amplitudes "
            f"to a mismatched assignment")

    level = trailing_mean(channel[rows], time[rows], series, span)
    excursion = (pred[rows] - level[:, None]).astype(np.float32)
    residual = (true[rows] - level[:, None]).astype(np.float32)
    return pool_amplitude(labels, excursion, residual, n_prototypes)


def lotsa_cell(pred_len, span, seed=42):
    """The same question, asked of the pretraining corpus."""
    add_offline_sources()
    from data.loader import split_pretrain_series
    from graph.builder import fit_prototypes

    with np.load(LOTSA_CACHE / f"sl512_pl{pred_len}" / "chronos.npz",
                 allow_pickle=False) as z:
        emb = np.asarray(z["embs"], np.float32)
        pred = np.asarray(z["preds"], np.float32)
        true = np.asarray(z["trues"], np.float32)
        sids = np.asarray(z["series_ids"], np.int64)
    with np.load(LEVELS / f"pl{pred_len}.npz", allow_pickle=False) as z:
        level = np.asarray(z[f"mean{span}"], np.float32)

    memory, _, _ = split_pretrain_series(sids, seed)
    built = fit_prototypes(emb[memory], true[memory, :, None],
                           pred[memory, :, None], 500, seed)
    excursion = (pred[memory] - level[memory, None]).astype(np.float32)
    residual = (true[memory] - level[memory, None]).astype(np.float32)
    return pool_amplitude(built["labels"], excursion, residual,
                          len(built["centers"]))


def suffix(span):
    return "" if span == 24 else f"_span{span}"
