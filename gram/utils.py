"""Small shared helpers."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

STATE: dict = {}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def chunks(total, size):
    for start in range(0, total, size):
        yield start, min(start + size, total)
