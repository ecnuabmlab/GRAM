"""Read-out over the retrieved subgraph.

The amplitude applied to the backbone's recent excursion is

    a(h) = 1 + ANCHOR_STRENGTH * (anchor(h) - 1)
              + RESIDUAL_WEIGHT * [ (prior(h) - 1) + phi(h) * lam(h) * v(h) ]

`anchor` is the pooled retrieval answer of the library, taken at a fixed
strength; `prior` is the evidence weighted average of the prototype curves;
`v` is the attentive vote over the subgraph nodes and `phi` the evidence gate
that scales it.  Only the residual is learned, so the correction can never
leave the range the retrieved evidence supports.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gram.config import ANCHOR_STRENGTH, RESIDUAL_WEIGHT

C0 = 0.541324854612918          # softplus(C0) = 1


class GraphFusion(nn.Module):
    """Relational propagation, then cross-attention driven by the query."""

    def __init__(self, horizon, width=96, heads=4, layers=2, dropout=0.2,
                 extra_dim=5, use_sim_bias=True):
        super().__init__()
        self.horizon = horizon
        self.heads = heads
        self.extra_dim = extra_dim
        self.use_sim_bias = use_sim_bias
        self.support_idx = -extra_dim
        self.coherence_idx = -extra_dim + 1
        self.sim_idx = -2
        node_dim = 2 * horizon + extra_dim
        query_dim = 2 * horizon + 3
        self.node_in = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, width), nn.GELU(),
            nn.Linear(width, width))
        self.query_in = nn.Sequential(
            nn.LayerNorm(query_dim), nn.Linear(query_dim, width), nn.GELU(),
            nn.Linear(width, width))
        self.props = nn.ModuleList([
            nn.ModuleDict({
                "shape": nn.Linear(width, width, bias=False),
                "bias": nn.Linear(width, width, bias=False),
                "self": nn.Linear(width, width),
                "norm": nn.LayerNorm(width),
                "ff": nn.Sequential(nn.LayerNorm(width),
                                    nn.Linear(width, 2 * width), nn.GELU(),
                                    nn.Linear(2 * width, width)),
            }) for _ in range(layers)])
        self.sim_gain = nn.Parameter(torch.zeros(1))
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(width)
        self.ceiling_gain = nn.Parameter(torch.zeros(4))
        self.ceiling_bias = nn.Parameter(torch.full((1,), 2.0))
        self.drop = nn.Dropout(dropout)
        self.scale_out = nn.Linear(width, horizon)
        nn.init.zeros_(self.scale_out.weight)
        nn.init.zeros_(self.scale_out.bias)

    def forward(self, node, adj_shape, adj_bias, query):
        horizon = self.horizon
        prior = query[:, horizon + 3:2 * horizon + 3]
        anchor = node[:, -1, :horizon]
        h = self.node_in(node)
        context = self.query_in(query)
        for layer in self.props:
            message = (layer["shape"](adj_shape @ h)
                       + layer["bias"](adj_bias @ h) + layer["self"](h))
            h = layer["norm"](h + message)
            h = h + layer["ff"](h)
        attn_mask = None
        if self.use_sim_bias:
            mask = (self.sim_gain * node[:, :, self.sim_idx])[:, None]
            attn_mask = mask.repeat_interleave(self.heads, dim=0)
        pooled, weights = self.attn(context[:, None], h, h, need_weights=True,
                                    attn_mask=attn_mask)
        pooled = self.attn_norm(pooled.squeeze(1))
        weights = weights.squeeze(1)
        amplitudes = node[:, :, :horizon]
        voted = torch.einsum("bn,bnh->bh", weights, amplitudes)
        gain = nn.functional.softplus(self.ceiling_gain)
        disagreement = torch.einsum(
            "bn,bnh->bh", weights,
            (amplitudes - 1.0 - voted[:, None]).abs())
        ceiling = torch.sigmoid(
            self.ceiling_bias
            + gain[0] * (weights * node[:, :, self.support_idx]).sum(
                1, keepdim=True)
            + gain[1] * (weights * node[:, :, self.coherence_idx]).sum(
                1, keepdim=True)
            - gain[2] * torch.einsum("bn,bnh->bh", weights,
                                     node[:, :, horizon:2 * horizon])
            - gain[3] * disagreement)
        lam = nn.functional.softplus(self.scale_out(pooled) + C0)
        residual = (prior - 1.0) + ceiling * lam * voted
        amplitude = 1.0 + ANCHOR_STRENGTH * anchor + RESIDUAL_WEIGHT * residual
        return amplitude, weights, ceiling

    def report_ceiling(self):
        gain = nn.functional.softplus(self.ceiling_gain).detach().cpu().numpy()
        return {"support": float(gain[0]), "coherence": float(gain[1]),
                "spread": float(-gain[2]), "disagreement": float(-gain[3]),
                "bias": float(self.ceiling_bias)}
