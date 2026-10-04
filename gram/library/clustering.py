"""Prototype library of one cell: assignment, coherence and evidence."""
from __future__ import annotations

import numpy as np

from gram.config import (CLUSTER, CLUSTER_FT, KNOWLEDGE, N_CLUSTERS,
                         PRECOMPUTE, SPAN, add_offline_sources)
from sklearn.cluster import MiniBatchKMeans

from gram.library.memory import add_corpus_node
from gram.data.series import star_ratio, trailing_mean
from gram.library.amplitude import MIN_WINDOWS, pool_amplitude


def is_ft_sidecar():
    return "precompute_ft" in str(PRECOMPUTE)


def cluster_dir():
    return CLUSTER_FT if is_ft_sidecar() else CLUSTER


def cluster_once(dataset, pred_len, n_clusters=N_CLUSTERS, backbone="chronos",
                 seed=42):
    root = cluster_dir() / f"k{n_clusters}"
    if is_ft_sidecar():
        stem = root / f"{dataset}_{backbone}_pl{pred_len}_cluster.npz"
    else:
        stem = root / f"{dataset}_pl{pred_len}_cluster.npz"
        legacy = root / f"{dataset}_cluster.npz"
        if not stem.is_file() and pred_len == 96 and legacy.is_file():
            stem = legacy
    if stem.is_file():
        with np.load(stem) as z:
            return {k: z[k] for k in z.files}

    pass
    from data.loader import load_aligned, normalize_rows, temporal_split

    print(f"    kmeans K={n_clusters} {backbone}/{dataset} pl{pred_len}",
          flush=True)
    emb, _, _, channel, time = load_aligned(
        str(PRECOMPUTE), dataset, "train", pred_len, backbone=backbone)
    rows = None
    for embargo in (pred_len, pred_len - 1):
        build, _ = temporal_split(channel, time, 0.8, embargo)
        if int(build.sum()) > 0:
            rows = np.flatnonzero(build)
            break
    if rows is None:
        raise SystemExit(f"{dataset}/pl{pred_len}: empty build split")
    points = normalize_rows(emb[rows])
    k = min(int(n_clusters), len(points))
    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed, batch_size=min(4096, len(points)),
        n_init=3, max_iter=150, reassignment_ratio=0.0)
    km.fit(points)
    raw = km.labels_.astype(np.int32)
    counts = np.bincount(raw, minlength=k)
    active = np.flatnonzero(counts > 0)
    remap = np.full(k, -1, np.int32)
    remap[active] = np.arange(len(active), dtype=np.int32)
    labels = remap[raw]
    centers = normalize_rows(km.cluster_centers_[active])
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem, centers=centers, labels=labels, rows=rows,
                        counts=counts[active].astype(np.int64))
    return {"centers": centers, "labels": labels, "rows": rows,
            "counts": counts[active].astype(np.int64)}


def production_memory(dataset, backbone, pred_len, series,
                      n_clusters=N_CLUSTERS, span=SPAN):
    """Prototype memory for one deployment cell under production knobs."""
    cluster_bb = backbone if is_ft_sidecar() else "chronos"
    clustered = cluster_once(dataset, pred_len, n_clusters, backbone=cluster_bb)
    cache = (cluster_dir() / f"k{n_clusters}"
             / f"{dataset}_{backbone}_pl{pred_len}_span{span}_amp.npz")
    legacy = (cluster_dir() / f"k{n_clusters}"
              / f"{dataset}_{backbone}_span{span}_amp.npz")
    if not cache.is_file() and pred_len == 96 and legacy.is_file():
        cache = legacy
    if cache.is_file():
        with np.load(cache) as z:
            amplitude = np.asarray(z["amplitude"], np.float32)
            spread = np.asarray(z["spread"], np.float32)
            support = np.asarray(z["support"], np.float32)
            coherence = np.asarray(z["coherence"], np.float32)
    else:
        pass
        from data.loader import load_aligned

        print(f"    pooling K={n_clusters} span={span} "
              f"{backbone}/{dataset}/pl{pred_len}", flush=True)
        _, pred, true, channel, time = load_aligned(
            str(PRECOMPUTE), dataset, "train", pred_len, backbone=backbone)
        rows = clustered["rows"]
        if int(rows.max()) >= len(pred):
            raise SystemExit(
                f"{dataset}/{backbone}/pl{pred_len}: cluster rows "
                f"exceed train length {len(pred)}")
        labels = clustered["labels"]
        n_proto = len(clustered["centers"])
        level = trailing_mean(channel[rows], time[rows], series, span)
        excursion = (pred[rows] - level[:, None]).astype(np.float32)
        residual = (true[rows] - level[:, None]).astype(np.float32)
        amplitude, spread, support = pool_amplitude(
            labels, excursion, residual, n_proto)
        flat = (true[rows] - pred[rows]).astype(np.float64).reshape(len(rows), -1)
        counts = np.bincount(labels, minlength=n_proto).astype(np.float64)
        sums = np.zeros((n_proto, flat.shape[1]), np.float64)
        sq = np.zeros_like(sums)
        np.add.at(sums, labels, flat)
        np.add.at(sq, labels, flat ** 2)
        mean = sums / np.maximum(counts[:, None], 1.0)
        var = np.maximum(sq / np.maximum(counts[:, None], 1.0) - mean ** 2, 0.0)
        coherence = np.clip(
            (mean ** 2 / (mean ** 2 + var + 1e-6)).mean(1), 0.0, 1.0
        ).astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, amplitude=amplitude, spread=spread,
                            support=support, coherence=coherence)
    with np.load(KNOWLEDGE / f"{dataset}_pl{pred_len}_{backbone}.npz") as z:
        memory = {
            "centers": np.asarray(clustered["centers"], np.float32),
            "coherence": coherence,
            "count": clustered["counts"].astype(np.float32),
            "emb": np.asarray(z["test_emb"], np.float32),
            "cal_emb": np.asarray(z["cal_emb"], np.float32),
            "amplitude": amplitude,
            "spread": spread,
            "support": support,
        }
    return add_corpus_node(memory)






def pool_curves(labels, res, nom, n_prototypes):
    """Mean residual curve and mean nominal-ratio curve per prototype.
"""
    horizon = res.shape[1]
    res_curve = np.zeros((n_prototypes, horizon), np.float32)
    nom_curve = np.ones((n_prototypes, horizon), np.float32)
    for proto in range(n_prototypes):
        rows = np.flatnonzero(labels == proto)
        if len(rows) < MIN_WINDOWS:
            continue
        res_curve[proto] = res[rows].mean(0)
        mean = np.nanmean(nom[rows], axis=0)
        nom_curve[proto] = np.where(np.isfinite(mean), mean, 1.0)
    return res_curve, nom_curve
