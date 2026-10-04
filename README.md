# GRAM: Graph-Retrieved Amplitude Memory

**Zero-shot forecasting with frozen time-series foundation models (TSFMs).**

A frozen TSFM can carry a recurring bias on a target domain. GRAM retrieves the
*prediction residuals* the TSFM made on similar historical windows — not the
ground-truth futures — distills them into a prototype memory and graph, and adds
an estimated error component back onto the frozen forecast. Both the TSFM and
the retrieval encoder stay frozen; the graph fusion head is the only trainable
part, and setting the correction to zero recovers the frozen TSFM exactly.

Offline, historical windows are clustered into prototypes whose amplitude
scaling errors and spread are stored in the Amplitude Memory (AMM); the
Prototype Graph (PGM) links prototypes by bias and shape relations. Online, the
top-K prototypes are grown one hop into a subgraph and the Graph Fusion Module
(GFM) turns it into the per-horizon correction. Evaluated on 7 datasets,
4 TSFM backbones and 4 horizons.

## Layout

```
gram/                     package
  config.py               paths, retrieval geometry, release constants
  data/series.py          target series, frozen forecasts, metric
  library/                Phase 1: prototype libraries (AMM + PGM)
  model/                  Phase 2: graph fusion read-out (GFM)
  train/                  GFM fitting on multi-domain corpus libraries
  evaluate/               grid evaluation and result tables
scripts/                  command line entry points
data/                     datasets, libraries, checkpoints (see data/README.md)
```

## Install

Requires Python >= 3.9.

```bash
pip install -r requirements.txt
```

## Quick start

Everything below runs on CPU; pass `--gpu 0` to use a device.

The repository ships the prototype libraries and the fitted read-out heads of
the Sundial, horizon 96 grid, which is all `scripts/evaluate.py` needs:

```bash
# frozen read-out on the seven datasets, Sundial, horizon 96
python scripts/evaluate.py --backbone sundial --horizon 96

# one dataset
python scripts/evaluate.py --backbone sundial --horizon 96 --datasets ETTh1
```

Each run writes `results/frozen.json`. Add `--force` to re-evaluate cells that
are already in the result file.

## Building the full grid

The evaluation grid is 4 backbones x 4 horizons x 7 datasets. The remaining
cells are rebuilt in three steps:

```bash
# 1. Phase 1: prototype library of one cell (AMM + PGM); needs
#    data/knowledge, data/forecasts and data/series, see data/README.md
python scripts/build_library.py --backbone sundial --horizon 96 --datasets ETTh1

# 2. Phase 2 (training): fit the read-out head on the multi-domain
#    corpus libraries only; no target labels take part
python scripts/train_head.py --backbone sundial --horizon 96 --seeds 3

# 3. Phase 2 (inference): evaluation over the target grid
python scripts/evaluate.py --backbone sundial --horizon 96
```

Every path in `gram/config.py` can be redirected with the matching `GRAM_*`
environment variable.

## Fine-tuned backbones

The read-out stays as fitted on the corpus; only the libraries are rebuilt from
the fine-tuned forecasts:

```bash
python scripts/evaluate_finetuned.py --horizon 96 --library data/library_ft
```
