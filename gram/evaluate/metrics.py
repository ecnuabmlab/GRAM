"""Metric helpers: absolute errors against the frozen backbone."""
from __future__ import annotations


def abs_pair(true, pred):
    """MSE and MAE of a forecast tensor."""
    err = pred - true
    return [float(err.pow(2).mean()), float(err.abs().mean())]
