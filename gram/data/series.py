"""Target series, frozen forecasts and the window protocol."""
from __future__ import annotations

import csv

import numpy as np

from gram.config import CSV_PATHS, FORECASTS, SEQ_LEN


def load_numeric_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        first = next(reader)
    keep = []
    for index, value in enumerate(first):
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        keep.append(index)
    data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=keep,
                         dtype=np.float32, invalid_raise=True)
    return data[:, None] if data.ndim == 1 else data


def standardized_series(dataset):
    """Replay the precompute scaler so contexts match the cached windows."""
    data = load_numeric_csv(CSV_PATHS[dataset])
    n = len(data)
    if dataset in {"ETTh1", "ETTh2"} and n >= 20 * 30 * 24:
        train_end = 12 * 30 * 24
    elif dataset in {"ETTm1", "ETTm2"} and n >= 20 * 30 * 24 * 4:
        train_end = 12 * 30 * 24 * 4
    else:
        train_end = int(n * 0.7)
    mean = data[:train_end].mean(axis=0, keepdims=True)
    scale = data[:train_end].std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (data - mean) / scale


def trailing_mean(channel, time, series, span):
    """Where the series has just been, for every window, without a loop."""
    x = series.astype(np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, x.shape[1]), np.float64), np.cumsum(x, axis=0)])
    end = time + SEQ_LEN
    return ((cumulative[end, channel] - cumulative[end - span, channel])
            / span).astype(np.float32)


def load_targets(dataset, pred_len, backbone):
    """Forecasts, labels and window indices for one cell."""
    keys = ("pred", "true", "cal_pred", "cal_true", "channel_ids",
            "time_indices", "cal_channel_ids", "cal_time_indices")
    out = {}
    with np.load(FORECASTS / f"{dataset}_pl{pred_len}_{backbone}.npz",
                 allow_pickle=False) as z:
        for key in keys:
            if key not in z.files:
                continue
            value = np.asarray(z[key])
            if value.ndim == 3:
                value = value[:, :, 0]
            out[key] = value.astype(
                np.int64 if key.endswith(("_ids", "indices")) else np.float32)
    return out


def score(true, pred, correction):
    """Percent of squared and absolute error removed from the bare forecast."""
    error = pred + correction - true
    bare = pred - true
    return (100 * (1 - error.pow(2).mean().item() / bare.pow(2).mean().item()),
            100 * (1 - error.abs().mean().item() / bare.abs().mean().item()))




RATIO_CLIP = (-2.0, 4.0)
STAR_EPS = 1e-6


def star_ratio(num, den):
    """Elementwise num/den, NaN where the denominator never moved.
"""
    w = np.abs(den)
    safe = np.where(w > STAR_EPS, den, 1.0)
    out = np.clip(num / safe, *RATIO_CLIP).astype(np.float32)
    out[w <= STAR_EPS] = np.nan
    return out
