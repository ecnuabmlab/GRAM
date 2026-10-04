"""Reading a prototype library and stamping the retrieval geometry."""
from __future__ import annotations

import numpy as np

from gram.config import AMPLITUDE, KNOWLEDGE
from gram.library.amplitude import suffix


def unit(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
def attach_amplitude(memory, stem):
    with np.load(AMPLITUDE / f"{stem}.npz", allow_pickle=False) as z:
        memory["amplitude"] = np.asarray(z["amplitude"], np.float32)
        memory["spread"] = np.asarray(z["spread"], np.float32)
        memory["support"] = np.asarray(z["support"], np.float32)
    if len(memory["amplitude"]) != len(memory["centers"]):
        raise SystemExit(f"{stem}: {len(memory['amplitude'])} amplitude rows "
                         f"against {len(memory['centers'])} prototypes")
    return memory
def add_corpus_node(memory):
    """One extra node holding the whole library's support-weighted answer.
"""
    weight = memory["support"] / max(float(memory["support"].sum()), 1.0)
    memory["node_amplitude"] = np.vstack([
        memory["amplitude"],
        (weight[:, None] * memory["amplitude"]).sum(0)[None]]).astype(np.float32)
    memory["node_spread"] = np.vstack([
        memory["spread"],
        (weight[:, None] * memory["spread"]).sum(0)[None]]).astype(np.float32)
    memory["node_coherence"] = np.append(
        memory["coherence"],
        float(weight @ memory["coherence"])).astype(np.float32)
    memory["node_count"] = np.append(
        memory["count"], float(memory["count"].sum())).astype(np.float32)
    return memory


def target_memory(dataset, backbone, pred_len, span):
    """Prototype memory for one cell, plus its calibration and test queries."""
    with np.load(KNOWLEDGE / f"{dataset}_pl{pred_len}_{backbone}.npz",
                 allow_pickle=False) as z:
        memory = {
            "centers": np.asarray(z["centers"], np.float32),
            "coherence": np.asarray(z["coherences"], np.float32),
            "count": np.asarray(z["counts"], np.float64).astype(np.float32),
            "emb": np.asarray(z["test_emb"], np.float32),
            "cal_emb": np.asarray(z["cal_emb"], np.float32),
        }
    attach_amplitude(memory,
                     f"{dataset}_pl{pred_len}_{backbone}{suffix(span)}")
    return add_corpus_node(memory)





PACK_GEOM = 3


def pack_is_current(path):
    """True iff this pack was built with the current retrieval geometry."""
    with np.load(path) as z:
        return "geom" in z.files and int(z["geom"]) == PACK_GEOM


def pack_is_own_only(path):
    """True iff this pack's graph has no foreign nodes (legacy packs do)."""
    with np.load(path) as z:
        return float(np.max(z["foreign"])) == 0.0


def load_pack(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def stamp_geom(blob):
    blob["geom"] = np.int32(PACK_GEOM)
    return blob
