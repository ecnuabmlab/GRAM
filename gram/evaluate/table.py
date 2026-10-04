"""Result tables: one block per dataset, one row per horizon."""
from __future__ import annotations

import json

import numpy as np

from gram.config import GRID_DATASETS, HORIZONS, PACKS
from pathlib import Path


from gram.config import GRID_DATASETS, HORIZONS, PACKS, RESULTS

METHOD = "GRAM"
BB_SHOW = (
    ("chronos", "Chronos"),
    ("sundial", "Sundial"),
    ("timesfm", "TimesFM"),
    ("moirai", "Moirai"),
)


def abs_metrics(pred, true):
    err = np.asarray(pred, np.float64) - np.asarray(true, np.float64)
    return float(np.mean(np.square(err))), float(np.mean(np.abs(err)))


def pack_abs(path, chunk=8192):
    """MSE/MAE of test pred vs true without materialising the whole pack."""
    with np.load(path, allow_pickle=False) as z:
        pred_key = "test_pred" if "test_pred" in z.files else "pred"
        true_key = "test_true" if "test_true" in z.files else "true"
        pred, true = z[pred_key], z[true_key]
        sse = sae = 0.0
        count = 0
        for start in range(0, len(pred), chunk):
            err = (pred[start:start + chunk].astype(np.float64)
                   - true[start:start + chunk].astype(np.float64))
            sse += float(np.square(err).sum())
            sae += float(np.abs(err).sum())
            count += err.size
    return sse / count, sae / count


def empty_table(datasets=GRID_DATASETS, horizons=HORIZONS):
    table = {}
    for dataset in datasets:
        table[dataset] = {}
        for horizon in horizons:
            mse, mae = {}, {}
            for _, label in BB_SHOW:
                mse[label] = None
                mae[label] = None
            for _, label in BB_SHOW:
                mse[f"{label}+{METHOD}"] = None
                mae[f"{label}+{METHOD}"] = None
            table[dataset][str(int(horizon))] = {"MSE": mse, "MAE": mae}
    return table


def put(table, dataset, horizon, backbone, bare, corrected):
    label = dict(BB_SHOW)[backbone]
    block = table[dataset][str(int(horizon))]
    block["MSE"][label] = bare[0]
    block["MAE"][label] = bare[1]
    block["MSE"][f"{label}+{METHOD}"] = corrected[0]
    block["MAE"][f"{label}+{METHOD}"] = corrected[1]


def parse_cell_name(name, default_horizon=None):
    """Accept 'chronos/ETTh1' or 'chronos/ETTh1/H96'."""
    parts = name.split("/")
    if len(parts) == 3 and parts[2].startswith("H"):
        return parts[0], parts[1], int(parts[2][1:])
    if len(parts) == 2 and default_horizon is not None:
        return parts[0], parts[1], int(default_horizon)
    raise ValueError(f"cannot parse cell name {name!r}")


def from_gains(cells, pack_dir=PACKS, default_horizon=None):
    """Invert stored percent gains against pack (pred, true)."""
    table = empty_table()
    for name, row in cells.items():
        backbone, dataset, horizon = parse_cell_name(name, default_horizon)
        if dataset not in table or backbone not in dict(BB_SHOW):
            continue
        path = pack_dir / f"{backbone}@{dataset}@pl{horizon}.npz"
        bare = pack_abs(path)
        gain = row.get("gram") or row.get("graft")
        corrected = (bare[0] * (1.0 - gain[0] / 100.0),
                     bare[1] * (1.0 - gain[1] / 100.0))
        put(table, dataset, horizon, backbone, bare, corrected)
        print(f"  {backbone}/{dataset}/H{horizon}: "
              f"{bare[0]:.4f}->{corrected[0]:.4f}", flush=True)
    return table


def from_abs(cells, bare_key, gram_key, default_horizon=None):
    table = empty_table()
    for name, row in cells.items():
        backbone, dataset, horizon = parse_cell_name(name, default_horizon)
        if dataset not in table or backbone not in dict(BB_SHOW):
            continue
        put(table, dataset, horizon, backbone, row[bare_key], row[gram_key])
    return table


def dump(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=4) + "\n")
