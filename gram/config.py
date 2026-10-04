"""Paths, retrieval geometry and release constants.

Every path can be redirected with the matching ``GRAM_*`` environment
variable; the defaults point at the ``data`` directory shipped next to this
package.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def _path(name, default):
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else Path(default)


DATA = _path("GRAM_DATA", ROOT / "data")
SERIES = _path("GRAM_SERIES", DATA / "series")
FORECASTS = _path("GRAM_FORECASTS", DATA / "forecasts")
LEVELS = _path("GRAM_LEVELS", DATA / "levels")
KNOWLEDGE = _path("GRAM_KNOWLEDGE", DATA / "knowledge")
AMPLITUDE = _path("GRAM_AMPLITUDE", DATA / "amplitude")
CLUSTER = _path("GRAM_CLUSTER", DATA / "cluster")
CLUSTER_FT = _path("GRAM_CLUSTER_FT", DATA / "cluster_ft")
PRECOMPUTE = _path("GRAM_PRECOMPUTE", DATA / "precompute")
LOTSA_CACHE = _path("GRAM_LOTSA_CACHE", DATA / "lotsa")
TSFM = _path("GRAM_TSFM", DATA / "tsfm")

LIBRARY = _path("GRAM_LIBRARY", DATA / "library")
PACKS = LIBRARY
PRETRAIN = _path("GRAM_CORPUS", DATA / "corpus")
CHECKPOINTS = _path("GRAM_CHECKPOINTS", DATA / "checkpoints")
RESULTS = _path("GRAM_RESULTS", ROOT / "results")

N_CLUSTERS = 128
SPAN = 24
TOP_K = 3
K_EXPAND = 3
K_SHAPE = 2
K_BIAS = 2
MIN_SUPPORT = 8
SEQ_LEN = 512

ANCHOR_STRENGTH = 0.35
RESIDUAL_WEIGHT = 0.15

DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange_rate")
GRID_DATASETS = DATASETS + ("electricity",)
BACKBONES = ("chronos", "sundial", "moirai", "timesfm")
HORIZONS = (96, 192, 336, 720)

CSV_PATHS = {
    "ETTh1": SERIES / "ETT-small" / "ETTh1.csv",
    "ETTh2": SERIES / "ETT-small" / "ETTh2.csv",
    "ETTm1": SERIES / "ETT-small" / "ETTm1.csv",
    "ETTm2": SERIES / "ETT-small" / "ETTm2.csv",
    "weather": SERIES / "weather" / "weather.csv",
    "exchange_rate": SERIES / "exchange_rate" / "exchange_rate.csv",
    "electricity": SERIES / "electricity" / "electricity.csv",
}


def parse_names(raw, default):
    if not raw:
        return list(default)
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def add_offline_sources():
    """Put the offline builder that wrote the knowledge files on sys.path.

    Only needed when the prototype libraries are rebuilt from scratch; set
    ``GRAM_OFFLINE_SRC`` to one or more directories separated by colons.
    """
    for part in os.environ.get("GRAM_OFFLINE_SRC", "").split(":"):
        part = part.strip()
        if part and part not in sys.path:
            sys.path.insert(0, part)
